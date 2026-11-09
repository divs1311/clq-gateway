"""Eval harness: run policies over the golden set, grade, aggregate, calibrate, ablate."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np

from .cache import SemanticCache
from .cascade import Gateway, Trace
from .judge import MockJudge, SimulatedHuman, is_grounded_oracle, swap_consistent
from .providers import MockProvider
from .router import LABELS, TIERS, FixedProbaRouter, choose_tier, oof_probabilities, risk_flags
from .stats import kappa_with_ci, mcnemar_exact, percentile, weighted_percentile, wilson
from .util import ROOT

POLICIES = ["single:small", "single:mid", "single:frontier", "static_router", "cheap_first", "cascade", "cascade+cache"]
BASELINE = "single:frontier"


# --------------------------------------------------------------------------- grading
def grade(case: dict, tr: Trace, judge, human=None) -> dict:
    c = case["contract"]
    rec = {"case_id": case["case_id"], "stratum": case["stratum"], "difficulty": case["difficulty"],
           "scenario": case["scenario"], "outcome": tr.outcome, "cache_hit": tr.cache_hit,
           "routed_tier": tr.routed_tier, "risk_flags": tr.risk_flags, "tiers_visited": tr.tiers_visited,
           "attempt_flags": [a.flags for a in tr.attempts],
           "latency_ms": round(tr.latency_ms, 1), "cost_usd": tr.cost_usd}
    if tr.outcome != "delivered":
        rec.update(category="HANDOFF", passed=False, true_pass=False, deterministic_pass=None, true_error=None,
                   judge=None, human_faithfulness=None, error=None, human_handoff=True)
        return rec

    o = tr.final_output
    exp_tool = c["expected_tool_call"]
    got_tool = o["tool_call"]
    checks = {
        "security": not any(re.search(p, json.dumps(o)) for p in c["forbidden_patterns_regex"]),
        "escalation": o["escalate_to_human"] == c["must_escalate"],
        "root_cause": o["root_cause_code"] == c["expected_root_cause"],
        "tool_name": (exp_tool is None and got_tool is None) or (
            exp_tool is not None and got_tool is not None and exp_tool["name"] == got_tool["name"]),
        "tool_args": (exp_tool is None and got_tool is None) or (
            exp_tool is not None and got_tool is not None and exp_tool["args"] == got_tool["args"]),
        "citations": set(c["required_citations"]) <= set(o["cited_chunks"]),
    }
    det = all(checks.values())
    j = judge.grade(case, o["customer_reply"])
    judge_pass = j["factual_faithfulness"] == 2
    passed = det and judge_pass
    error = None
    if not passed:
        for name, label in [("security", "SECURITY_VIOLATION"), ("escalation", "ESCALATION_ERROR"),
                            ("root_cause", "WRONG_ROOT_CAUSE"), ("tool_name", "WRONG_TOOL"),
                            ("tool_args", "WRONG_TOOL_ARGS"), ("citations", "CITATION_ERROR")]:
            if not checks[name]:
                error = label
                break
        error = error or "UNGROUNDED_REPLY"
    hf = human.faithfulness(case, o["customer_reply"]) if human else None
    det_error = next((lbl for name, lbl in [("security", "SECURITY_VIOLATION"), ("escalation", "ESCALATION_ERROR"),
                                            ("root_cause", "WRONG_ROOT_CAUSE"), ("tool_name", "WRONG_TOOL"),
                                            ("tool_args", "WRONG_TOOL_ARGS"), ("citations", "CITATION_ERROR")]
                      if not checks[name]), None)
    true_error = None if hf is None else (det_error or ("UNGROUNDED_REPLY" if hf != 2 else None))
    rec.update(category="PASS" if passed else "SILENT_ERROR", passed=passed, true_error=true_error,
               true_pass=(det and hf == 2) if human else None, deterministic_pass=det, checks=checks,
               judge=j, human_faithfulness=hf, error=error, reply=o["customer_reply"],
               human_handoff=bool(o["escalate_to_human"]), output=o)
    return rec


# --------------------------------------------------------------------------- aggregation
def stratum_weights(cases: list[dict], prod_mix: dict) -> dict:
    n = len(cases)
    share = Counter(c["difficulty"] for c in cases)
    return {d: prod_mix[d] / (share[d] / n) for d in share}


def aggregate(records: list[dict], weights: dict | None = None, business: dict | None = None) -> dict:
    n = len(records)
    w = np.array([weights[r["difficulty"]] if weights else 1.0 for r in records])
    W = w.sum()

    def rate(pred):
        return float(sum(wi for wi, r in zip(w, records) if pred(r)) / W)

    lat = [r["latency_ms"] for r in records]
    k = sum(r["passed"] for r in records)
    out = {
        "n": n,
        "pass_rate": rate(lambda r: r["passed"]),
        "silent_error_rate": rate(lambda r: r["category"] == "SILENT_ERROR"),
        "validator_handoff_rate": rate(lambda r: r["category"] == "HANDOFF"),
        "human_handoff_rate": rate(lambda r: r["human_handoff"]),
        "escalation_rate": rate(lambda r: len(r["tiers_visited"]) > 1),
        "cache_hit_rate": rate(lambda r: r["cache_hit"]),
        "cost_per_1k_usd": float(sum(wi * r["cost_usd"] for wi, r in zip(w, records)) / W * 1000),
        "latency_ms": {f"p{q}": weighted_percentile(lat, w, q) for q in (50, 90, 95, 99)},
        "final_tier_share": {t: rate(lambda r, t=t: bool(r["tiers_visited"]) and r["tiers_visited"][-1] == t) for t in TIERS},
        "attempts_per_request": float(sum(wi * len(r["tiers_visited"]) for wi, r in zip(w, records)) / W),
    }
    if not weights:
        out["pass_ci95"] = wilson(k, n)
    simulated = bool(records) and all(r["true_pass"] is not None for r in records)
    if simulated:
        out["true_pass_rate_simulated"] = rate(lambda r: bool(r["true_pass"]))
        out["true_silent_error_rate_simulated"] = rate(lambda r: r["category"] != "HANDOFF" and not r["true_pass"])
    if business:
        # Prefer human-grounded silent errors when available (the judge failed its calibration gate).
        silent = out["true_silent_error_rate_simulated"] if simulated else out["silent_error_rate"]
        out["fully_loaded_per_1k_usd"] = (out["cost_per_1k_usd"]
                                          + 1000 * out["validator_handoff_rate"] * business["human_cost_per_handoff"]
                                          + 1000 * silent * business["silent_error_cost"])
    return out


def by_stratum(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r["stratum"]].append(r)
    return {s: {"n": len(g), "pass_rate": sum(r["passed"] for r in g) / len(g),
                "silent_error_rate": sum(r["category"] == "SILENT_ERROR" for r in g) / len(g),
                "handoff_rate": sum(r["category"] == "HANDOFF" for r in g) / len(g)} for s, g in groups.items()}


# --------------------------------------------------------------------------- runner
class Harness:
    def __init__(self, cfg: dict, cases: list[dict], mode: str = "mock"):
        self.cfg, self.cases = cfg, cases
        self.by_id = {c["case_id"]: c for c in cases}
        if mode == "mock":
            self.provider = MockProvider(cfg, self.by_id)
            self.judge, self.human = MockJudge(), SimulatedHuman()
        else:  # pragma: no cover
            import os
            from .judge import LLMJudge
            from .providers import LiteLLMProvider
            self.provider = LiteLLMProvider(cfg)
            self.judge, self.human = LLMJudge(os.environ["CLQ_JUDGE_MODEL"]), None
        self.mode = mode
        self.oof = {use_ctx: dict(zip([c["case_id"] for c in cases],
                                      oof_probabilities(cases, use_context=use_ctx, folds=cfg["router"]["cv_folds"])))
                    for use_ctx in (True, False)}

    def make_cache(self, threshold: float | None = None) -> SemanticCache:
        return SemanticCache.from_seed(ROOT / "data" / "faq_cache_seed.jsonl",
                                       threshold if threshold is not None else self.cfg["cache"]["similarity_threshold"])

    def run_policy(self, policy: str, cfg: dict | None = None, alpha: float | None = None,
                   use_context: bool = True, cache_threshold: float | None = None,
                   use_risk_rules: bool = True) -> list[dict]:
        cfg = cfg or self.cfg
        base = policy.replace("+cache", "")
        router = FixedProbaRouter(self.oof[use_context], alpha if alpha is not None else cfg["router"]["alpha"],
                                  use_risk_rules=use_risk_rules)
        cache = self.make_cache(cache_threshold) if policy.endswith("+cache") else None
        gw = Gateway(cfg, self.provider, base, router=router if base in {"static_router", "cascade"} else None, cache=cache)
        return [grade(c, gw.handle(c["case_id"], c["input"]), self.judge, self.human) for c in self.cases]

    # ------------------------------------------------------------------ full run
    def run(self) -> dict:
        prod_w = stratum_weights(self.cases, self.cfg["business"]["production_mix"])
        runs = {p: self.run_policy(p) for p in POLICIES}
        base_pass = [r["passed"] for r in runs[BASELINE]]
        summary = {}
        for p, recs in runs.items():
            summary[p] = {"golden_mix": aggregate(recs), "production_mix": aggregate(recs, prod_w, self.cfg["business"]),
                          "by_stratum": by_stratum(recs),
                          "vs_frontier_mcnemar": mcnemar_exact([r["passed"] for r in recs], base_pass)}
        return {
            "mode": self.mode,
            "simulated": self.mode == "mock",
            "alpha": self.cfg["router"]["alpha"],
            "production_weights": prod_w,
            "summary": summary,
            "records": runs,
            "router": self.router_report(),
            "validator": self.validator_report(runs) if self.mode == "mock" else None,
            "calibration": self.calibration_report(runs),
            "pairwise": self.pairwise_report(runs["cascade+cache"], runs[BASELINE]),
            "error_analysis": self.error_report(runs),
            "ablations": self.ablations(prod_w),
        }

    # ------------------------------------------------------------------ reports
    def router_report(self) -> dict:
        out = {}
        for use_ctx, key in ((True, "query_plus_context"), (False, "query_only")):
            conf = np.zeros((3, 3))
            for c in self.cases:
                pred = int(np.argmax(self.oof[use_ctx][c["case_id"]]))
                conf[LABELS.index(c["difficulty"]), pred] += 1
            routed = Counter("frontier" if risk_flags(c["input"]) else
                             choose_tier(self.oof[use_ctx][c["case_id"]], self.cfg["router"]["alpha"]) for c in self.cases)
            out[key] = {"accuracy": float(np.trace(conf) / conf.sum()),
                        "confusion_rows_true_cols_pred": conf.astype(int).tolist(),
                        "row_normalised": (conf / conf.sum(axis=1, keepdims=True)).round(4).tolist(),
                        "routed_tier_counts_at_alpha": {t: routed.get(t, 0) for t in TIERS}}
        hard_routes = Counter(choose_tier(self.oof[True][c["case_id"]], self.cfg["router"]["alpha"])
                              for c in self.cases if c["difficulty"] == "hard")
        out["hard_routing_without_risk_rules"] = {t: hard_routes.get(t, 0) for t in TIERS}
        flagged = Counter((c["stratum"], bool(risk_flags(c["input"]))) for c in self.cases)
        out["risk_rules"] = {
            "adversarial_flagged": flagged[("adversarial", True)],
            "adversarial_total": flagged[("adversarial", True)] + flagged[("adversarial", False)],
            "non_adversarial_flagged": flagged[("routine", True)] + flagged[("multi_hop", True)],
            "non_adversarial_total": sum(v for (s, _), v in flagged.items() if s != "adversarial"),
            "caveat": "Rules were written with knowledge of the golden-set attack categories; validate on a held-out red-team set."}
        return out

    def validator_report(self, runs: dict) -> dict:
        out = {}
        for t in TIERS:
            wrong = flagged_wrong = right = flagged_right = 0
            reasons = Counter()
            for r in runs[f"single:{t}"]:
                case = self.by_id[r["case_id"]]
                correct = self.provider.is_correct(case, t)
                flags = r["attempt_flags"][0]
                if correct:
                    right += 1
                    flagged_right += bool(flags)
                else:
                    wrong += 1
                    flagged_wrong += bool(flags)
                for f in flags:
                    reasons[f] += 1
            out[t] = {"recall_on_wrong": flagged_wrong / wrong if wrong else None, "n_wrong": wrong,
                      "false_flag_rate_on_correct": flagged_right / right if right else None, "n_correct": right,
                      "flag_reasons": dict(reasons.most_common())}
        return out

    def calibration_report(self, runs: dict) -> dict | None:
        if self.human is None:
            return None  # real mode: run `python -m clq.cli calibrate` with your labels
        seen, j, h, lengths = set(), [], [], []
        for p in ("single:small", "single:mid", "single:frontier"):
            for r in runs[p]:
                if r["category"] == "HANDOFF":
                    continue
                key = (r["case_id"], r["reply"])
                if key in seen:
                    continue
                seen.add(key)
                j.append(int(r["judge"]["factual_faithfulness"] == 2))
                h.append(int(r["human_faithfulness"] == 2))
                lengths.append(len(r["reply"]))
        j, h, lengths = np.array(j), np.array(h), np.array(lengths)
        long = lengths > 250
        res = kappa_with_ci(h, j)
        neg = h == 0
        res["judge_false_pass_rate_on_ungrounded"] = {
            "short_replies": float(j[neg & ~long].mean()) if (neg & ~long).any() else None,
            "n_short": int((neg & ~long).sum()),
            "long_replies": float(j[neg & long].mean()) if (neg & long).any() else None,
            "n_long": int((neg & long).sum())}
        res["judge_false_fail_rate_on_grounded"] = float((1 - j[~neg]).mean())
        slo = self.cfg["slo"]
        res["gate"] = {"min_point": slo["min_judge_kappa_point"], "min_ci_low": slo["min_judge_kappa_ci_low"],
                       "passed": bool(res["kappa"] >= slo["min_judge_kappa_point"] and res["ci_low"] >= slo["min_judge_kappa_ci_low"])}
        res["note"] = "SIMULATED judge vs SIMULATED human labels; replace with real labels before quoting."
        return res

    def pairwise_report(self, a: list[dict], b: list[dict]) -> dict:
        consistent = inconsistent = 0
        verdicts = Counter()
        for ra, rb in zip(a, b):
            if ra["category"] == "HANDOFF" or rb["category"] == "HANDOFF" or ra["reply"] == rb["reply"]:
                continue
            s = swap_consistent(self.judge, self.by_id[ra["case_id"]], ra["reply"], rb["reply"])
            if s["consistent"]:
                consistent += 1
                verdicts[s["verdict"]] += 1
            else:
                inconsistent += 1
        total = consistent + inconsistent
        return {"compared": total, "swap_consistent": consistent,
                "swap_consistency_rate": consistent / total if total else None,
                "candidate_wins": verdicts["x"], "baseline_wins": verdicts["y"], "ties": verdicts["tie"],
                "candidate": "cascade+cache", "baseline": BASELINE}

    def error_report(self, runs: dict) -> dict:
        out = {}
        for p, recs in runs.items():
            silent = Counter(r["error"] for r in recs if r["category"] == "SILENT_ERROR")
            caught = Counter(f for r in recs for fl in r["attempt_flags"] for f in fl)
            out[p] = {"silent_errors": dict(silent.most_common()), "n_silent": sum(silent.values()),
                      "validator_flags": dict(caught.most_common())}
            if self.human is not None:
                true = Counter(r["true_error"] for r in recs if r["category"] != "HANDOFF" and r["true_error"])
                out[p]["true_silent_errors"] = dict(true.most_common())
                out[p]["n_true_silent"] = sum(true.values())
                out[p]["judge_false_fails"] = sum(1 for r in recs if r["category"] == "SILENT_ERROR" and r["true_error"] is None)
        return out

    def ablations(self, prod_w: dict) -> dict:
        def row(recs, **extra):
            g, pm = aggregate(recs), aggregate(recs, prod_w, self.cfg["business"])
            return {**extra, "pass_rate": g["pass_rate"], "silent_error_rate": g["silent_error_rate"],
                    "true_silent_error_rate": g.get("true_silent_error_rate_simulated"),
                    "handoff_rate": g["validator_handoff_rate"], "cost_per_1k_prod": pm["cost_per_1k_usd"],
                    "fully_loaded_per_1k_prod": pm["fully_loaded_per_1k_usd"],
                    "p90_ms_prod": pm["latency_ms"]["p90"], "p95_ms_prod": pm["latency_ms"]["p95"]}

        out = {"alpha": [row(self.run_policy("cascade+cache", alpha=a), alpha=a) for a in (0.5, 0.6, 0.7, 0.8, 0.9)]}
        out["router_features"] = [row(self.run_policy("cascade+cache", use_context=u),
                                      features="query+context" if u else "query only") for u in (True, False)]
        conf_rows = []
        for tau in (0.5, 0.6, 0.7, 0.8):
            cfg = deepcopy(self.cfg)
            for t in TIERS:
                cfg["tiers"][t]["confidence_threshold"] = tau
            conf_rows.append(row(self.run_policy("cascade+cache", cfg=cfg), confidence_threshold=tau))
        out["confidence_threshold"] = conf_rows
        out["risk_rules"] = [row(self.run_policy("cascade+cache", use_risk_rules=u), risk_rules="on" if u else "off")
                             for u in (True, False)]
        out["cache_threshold"] = self.cache_sweep()
        return out

    def cache_sweep(self) -> list[dict]:
        cache = self.make_cache(0.0)
        sims = [(c, cache.best(c["input"]["user_query"])) for c in self.cases]
        rows = []
        for th in (0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.70, 0.75):
            howto = [(c, i, s) for c, (i, s) in sims if c["scenario"].startswith("howto:")]
            other = [(c, i, s) for c, (i, s) in sims if not c["scenario"].startswith("howto:")]
            hits = [(c, i) for c, i, s in howto if s >= th]
            correct_hits = sum(cache.entries[i]["faq_id"] == c["scenario"].split(":")[1] for c, i in hits)
            false_hits = sum(s >= th for _, _, s in other)
            rows.append({"threshold": th, "howto_hit_rate": len(hits) / len(howto),
                         "howto_hits_correct": correct_hits, "howto_hits_wrong_faq": len(hits) - correct_hits,
                         "false_hits_on_non_howto": false_hits, "n_howto": len(howto), "n_other": len(other)})
        return rows
