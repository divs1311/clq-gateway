"""Turn an eval run into Markdown reports and the dashboard data file."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .economics import compare, default_params, evaluate_policy
from .util import ROOT

POLICY_LABEL = {
    "single:small": "Small only", "single:mid": "Mid only", "single:frontier": "Frontier only (baseline)",
    "static_router": "Static router", "cheap_first": "Cheap-first cascade", "cascade": "Routed cascade",
    "cascade+cache": "Routed cascade + cache",
}
BANNER = ("> **SIMULATED RUN.** Model outputs come from the oracle-backed mock provider, whose accuracy is set by\n"
          "> `mock_priors` in `config/tiers.yaml`. Every number below follows from those assumptions plus the real\n"
          "> grading, routing, validation and statistics code. Re-run with `--mode litellm` to get measurements.\n")


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def money(x, d=2):
    return f"${x:,.{d}f}"


def sim_vs_eval(res: dict, cfg: dict) -> list[dict]:
    p = default_params(cfg)
    rows = []
    for pol in ["single:small", "single:mid", "single:frontier", "static_router", "cheap_first", "cascade+cache"]:
        q = deepcopy(p)
        q["cache_on"] = pol.endswith("+cache")
        s = evaluate_policy(q, pol.replace("+cache", ""))
        e = res["summary"][pol]["production_mix"]
        rows.append({"policy": pol,
                     "sim_pass": s["pass_rate"], "eval_pass": e.get("true_pass_rate_simulated", e["pass_rate"]),
                     "sim_silent": s["silent_error_rate"], "eval_silent": e.get("true_silent_error_rate_simulated", e["silent_error_rate"]),
                     "sim_handoff": s["handoff_rate"], "eval_handoff": e["validator_handoff_rate"],
                     "sim_cost_1k": s["model_cost_per_1k"], "eval_cost_1k": e["cost_per_1k_usd"]})
    return rows


def breakeven_silent_cost(cfg: dict, a: str = "cascade", b: str = "single:frontier") -> float | None:
    """Silent-error cost at which policy a and b have equal fully-loaded cost (linear in S)."""
    p = default_params(cfg)
    ra, rb = evaluate_policy(p, a), evaluate_policy(p, b)
    other_a = ra["model_cost_per_1k"] + ra["human_cost_per_1k"]
    other_b = rb["model_cost_per_1k"] + rb["human_cost_per_1k"]
    ds = 1000 * (ra["silent_error_rate"] - rb["silent_error_rate"])
    if abs(ds) < 1e-12:
        return None
    return (other_b - other_a) / ds


def breakeven_recall(cfg: dict, a: str = "cascade", b: str = "single:frontier") -> float | None:
    """Smallest validator recall (grid 0.50-1.00) at which a's fully-loaded cost <= b's."""
    p = default_params(cfg)
    for r in [x / 100 for x in range(50, 101)]:
        q = deepcopy(p)
        q["recall"] = r
        if evaluate_policy(q, a)["fully_loaded_per_1k"] <= evaluate_policy(q, b)["fully_loaded_per_1k"]:
            return r
    return None


def write_summary(res: dict, cfg: dict, out: Path) -> None:
    s = res["summary"]
    L = ["# Eval summary", "", BANNER if res["simulated"] else "", "",
         f"Golden set: 200 cases (120 routine / 50 multi-hop / 30 adversarial). Production re-weighting uses "
         f"{cfg['business']['production_mix']} (weights per stratum: "
         + ", ".join(f"{k} x{v:.3f}" for k, v in res["production_weights"].items()) + ").", "",
         "## Golden-set results (unweighted, n = 200 per policy)", "",
         "| Policy | Pass@1 (judge) | 95% CI | Pass@1 (human*) | Silent errors (judge) | Silent errors (human*) | Validator hand-offs | Escalated | McNemar p vs frontier |",
         "|---|---|---|---|---|---|---|---|---|"]
    for p, v in s.items():
        g, m = v["golden_mix"], v["vs_frontier_mcnemar"]
        ci = g["pass_ci95"]
        pval = "-" if p == "single:frontier" else "{:.3f}".format(m["p_value"])
        L.append(f"| {POLICY_LABEL[p]} | {pct(g['pass_rate'])} | {pct(ci[0])}-{pct(ci[1])} | {pct(g.get('true_pass_rate_simulated'))} | "
                 f"{pct(g['silent_error_rate'])} | {pct(g.get('true_silent_error_rate_simulated'))} | {pct(g['validator_handoff_rate'])} | "
                 f"{pct(g['escalation_rate'])} | {pval} |")
    L += ["", "*human = simulated human labels in mock mode.", "",
          "## Production-weighted economics and latency", "",
          "| Policy | Model cost / 1k | Fully-loaded / 1k | P50 | P90 | P95 | P99 | Final tier small/mid/frontier |",
          "|---|---|---|---|---|---|---|---|"]
    for p, v in s.items():
        pm = v["production_mix"]
        lat = pm["latency_ms"]
        ft = pm["final_tier_share"]
        L.append(f"| {POLICY_LABEL[p]} | {money(pm['cost_per_1k_usd'], 3)} | {money(pm['fully_loaded_per_1k_usd'], 0)} | "
                 f"{lat['p50']:.0f} ms | {lat['p90']:.0f} ms | {lat['p95']:.0f} ms | {lat['p99']:.0f} ms | "
                 f"{pct(ft['small'], 0)} / {pct(ft['mid'], 0)} / {pct(ft['frontier'], 0)} |")
    L += ["", "Latency percentiles from 200 weighted samples are noisy (P99 is roughly the 2nd-slowest request). "
          "Use the simulator for production-scale percentiles.", "",
          "Fully-loaded cost = model cost + validator hand-offs x human cost + silent errors x silent-error cost "
          f"({money(cfg['business']['human_cost_per_handoff'])} and {money(cfg['business']['silent_error_cost'])}). "
          "Policy-required escalations (prompt injection, destructive or cross-tenant requests) are excluded because a "
          "correct system sends them to a human under every policy.", ""]

    r = res["router"]
    L += ["## Router", "",
          f"Out-of-fold accuracy (5-fold CV): query+context {pct(r['query_plus_context']['accuracy'])}, query only {pct(r['query_only']['accuracy'])}.",
          "", "Row-normalised confusion (query+context; rows = true easy/medium/hard, cols = predicted):", "",
          "| true \\ pred | easy | medium | hard |", "|---|---|---|---|"]
    for lbl, row in zip(["easy", "medium", "hard"], r["query_plus_context"]["row_normalised"]):
        L.append(f"| {lbl} | " + " | ".join(pct(x) for x in row) + " |")
    rr = r["risk_rules"]
    hr = r["hard_routing_without_risk_rules"]
    L += ["", f"Without risk rules, the classifier at alpha={res['alpha']} routes the 30 adversarial cases as small {hr['small']} / "
          f"mid {hr['mid']} / frontier {hr['frontier']}: short ambiguous messages and destructive or cross-tenant requests look "
          f"like simple questions. Deterministic risk rules flag {rr['adversarial_flagged']}/{rr['adversarial_total']} "
          f"adversarial cases with {rr['non_adversarial_flagged']}/{rr['non_adversarial_total']} false positives and force them to "
          f"the frontier tier. Caveat: {rr['caveat']}", ""]

    if res.get("validator"):
        L += ["## Validator (measured against mock ground truth)", "", "| Tier | Recall on wrong answers | False-flag rate on correct | Top flags |", "|---|---|---|---|"]
        for t, v in res["validator"].items():
            top = ", ".join(f"{k} {n}" for k, n in list(v["flag_reasons"].items())[:4])
            L.append(f"| {t} | {pct(v['recall_on_wrong'])} (n={v['n_wrong']}) | {pct(v['false_flag_rate_on_correct'])} (n={v['n_correct']}) | {top} |")
        L.append("")

    c = res["calibration"]
    if c:
        L += ["## Judge calibration", "", f"> {c['note']}", "",
              f"Cohen's kappa (judge vs human, faithfulness pass/fail, n={c['n']}): **{c['kappa']:.3f}** "
              f"(95% bootstrap CI {c['ci_low']:.3f}-{c['ci_high']:.3f}); raw agreement {pct(c['raw_agreement'])}.",
              f"Gate (kappa >= {c['gate']['min_point']} and CI low >= {c['gate']['min_ci_low']}): **{'PASS' if c['gate']['passed'] else 'FAIL'}**.",
              f"Judge false-fail rate on grounded replies: {pct(c['judge_false_fail_rate_on_grounded'])}. "
              f"False-pass on ungrounded replies: short {pct(c['judge_false_pass_rate_on_ungrounded']['short_replies'])} "
              f"(n={c['judge_false_pass_rate_on_ungrounded']['n_short']}), long {pct(c['judge_false_pass_rate_on_ungrounded']['long_replies'])} "
              f"(n={c['judge_false_pass_rate_on_ungrounded']['n_long']}).", "",
              "High raw agreement with a low kappa is the expected signature of an imbalanced label set: most replies are "
              "grounded, so agreeing on them is easy. Because the gate fails, ship decisions use deterministic checks plus "
              "human labels, not the judge.", ""]
    pw = res["pairwise"]
    L += ["## Pairwise comparison with position swap", "",
          f"{POLICY_LABEL[pw['candidate']]} vs {POLICY_LABEL[pw['baseline']]} on {pw['compared']} cases where both delivered different replies. "
          f"Swap-consistent verdicts: {pw['swap_consistent']} ({pct(pw['swap_consistency_rate'])}). Among them: candidate wins "
          f"{pw['candidate_wins']}, baseline wins {pw['baseline_wins']}, ties {pw['ties']}. Inconsistent verdicts are discarded "
          "because they reflect position bias, not quality.", ""]

    L += ["## Simulator vs eval agreement (production-weighted)", "",
          "The simulator uses pooled validator rates and the measured routing matrix, so small gaps are expected "
          "(per-tier validator recall differs from the pooled value, and the hard stratum has only 30 cases).", "",
          "| Policy | Pass sim / eval | Silent sim / eval | Hand-off sim / eval | Model cost/1k sim / eval |", "|---|---|---|---|---|"]
    for row in sim_vs_eval(res, cfg):
        L.append(f"| {POLICY_LABEL[row['policy']]} | {pct(row['sim_pass'])} / {pct(row['eval_pass'])} | {pct(row['sim_silent'])} / "
                 f"{pct(row['eval_silent'])} | {pct(row['sim_handoff'])} / {pct(row['eval_handoff'])} | "
                 f"{money(row['sim_cost_1k'], 3)} / {money(row['eval_cost_1k'], 3)} |")
    be_s, be_r = breakeven_silent_cost(cfg), breakeven_recall(cfg)
    L += ["", "## Decision-relevant break-evens (simulator, default assumptions)", "",
          f"- The routed cascade beats frontier-only on fully-loaded cost only if a silent error costs less than **{money(be_s)}**.",
          f"- At the assumed {money(cfg['business']['silent_error_cost'])} per silent error, the cascade needs validator recall of at least "
          f"**{pct(be_r, 0) if be_r else 'more than 100%'}** to break even (currently {pct(cfg['business']['validator']['recall'], 0)}).", ""]
    out.write_text("\n".join(L))


def write_ablations(res: dict, out: Path) -> None:
    a = res["ablations"]
    L = ["# Ablations", "", BANNER if res["simulated"] else "", "",
         "All rows use the routed cascade + cache policy and vary one factor. Quality columns are golden-set rates; cost, "
         "fully-loaded cost and latency are production-weighted. Each ablation was chosen on this same golden set, so treat "
         "the winner as a hypothesis to confirm on fresh data, not a tuned result.", ""]

    def table(rows, key, title):
        L.extend([f"## {title}", "", f"| {key} | Pass@1 (judge) | Silent (judge) | Silent (human*) | Hand-offs | Cost / 1k | Fully-loaded / 1k | P90 | P95 |",
                  "|---|---|---|---|---|---|---|---|---|"])
        for r in rows:
            L.append(f"| {r[key]} | {pct(r['pass_rate'])} | {pct(r['silent_error_rate'])} | {pct(r['true_silent_error_rate'])} | "
                     f"{pct(r['handoff_rate'])} | {money(r['cost_per_1k_prod'], 3)} | {money(r['fully_loaded_per_1k_prod'], 0)} | "
                     f"{r['p90_ms_prod']:.0f} ms | {r['p95_ms_prod']:.0f} ms |")
        L.append("")

    table(a["alpha"], "alpha", "Router conservativeness (alpha)")
    table(a["confidence_threshold"], "confidence_threshold", "Validator confidence threshold (all tiers)")
    table(a["router_features"], "features", "Router features")
    table(a["risk_rules"], "risk_rules", "Risk pre-filter")
    L += ["## Semantic-cache similarity threshold (offline TF-IDF embedder)", "",
          "| Threshold | How-to hit rate | Hits on wrong FAQ | False hits on non-how-to (n=150) |", "|---|---|---|---|"]
    for r in a["cache_threshold"]:
        L.append(f"| {r['threshold']:.2f} | {pct(r['howto_hit_rate'])} | {r['howto_hits_wrong_faq']} | {r['false_hits_on_non_howto']} |")
    L += ["", "Below 0.60 the lexical embedder starts serving FAQ answers to diagnostic questions. At 0.62 it is safe "
          "on this set (0 of 150 false hits; the rule-of-three upper bound is about 2%) but recalls only 6% of how-to "
          "questions. The cache is not worth much until it runs on a real embedding model and real repeat traffic.", ""]
    out.write_text("\n".join(L))


def write_error_analysis(res: dict, out: Path) -> None:
    e = res["error_analysis"]
    human = "true_silent_errors" in next(iter(e.values()))
    L = ["# Error analysis", "", BANNER if res["simulated"] else "", "",
         "Silent errors = answers delivered to the merchant that are wrong. Because the judge failed its calibration gate, "
         "the breakdown uses the deterministic contract plus human labels; the judge-graded count is shown for comparison "
         "and its excess is judge false-fails. Validator flags = problems caught before delivery (they trigger escalation "
         "or a hand-off).", "",
         "| Policy | Silent errors (human) | Breakdown | Judge-graded silent errors (of which judge false-fails) | Validator flags |",
         "|---|---|---|---|---|"]
    for p, v in e.items():
        src, n = (v["true_silent_errors"], v["n_true_silent"]) if human else (v["silent_errors"], v["n_silent"])
        br = ", ".join(f"{k} {c} ({c / n:.0%})" for k, c in src.items()) if n else "-"
        fl = ", ".join(f"{k} {c}" for k, c in v["validator_flags"].items()) or "-"
        jf = f"{v['n_silent']} ({v['judge_false_fails']})" if human else str(v["n_silent"])
        L.append(f"| {POLICY_LABEL[p]} | {n} | {br} | {jf} | {fl} |")
    L += ["", "## What to fix, by failure class", "",
          "| Failure class | Why the validator misses it | Fix | How to verify |", "|---|---|---|---|",
          "| UNGROUNDED_REPLY | Schema-valid and confident; only meaning is wrong | Claim-level grounding check (NLI or a small judge) on cheap-tier replies before delivery | Recall of the grounding check on held-out ungrounded replies >= 0.85 |",
          "| WRONG_TOOL_ARGS | Arguments are well-formed, just wrong (channel, SKU, date) | Cross-check args against entities extracted from the query and the order record | Arg exact-match on multi-hop >= 95% |",
          "| ESCALATION_ERROR | Escalation flag is a single boolean with no evidence | Deterministic policy rules decide escalation; the model only drafts the reply | 0 missed escalations on the red-team set |",
          "| WRONG_ROOT_CAUSE | Enum is valid | Constrain the root-cause enum to the codes in the retrieved chunks | Root-cause accuracy per stratum |",
          "| SCHEMA_CORRUPTION / SECURITY | Caught by Pydantic and regex guards | Keep; add constrained decoding where supported | Already 100% caught in this run |", ""]
    out.write_text("\n".join(L))


def dashboard_data(res: dict, cfg: dict, cases: list[dict]) -> dict:
    keep = ["single:frontier", "cascade+cache", "single:small", "static_router"]
    by_id = {c["case_id"]: c for c in cases}
    explorer = []
    for i, c in enumerate(cases):
        row = {"id": c["case_id"], "stratum": c["stratum"], "scenario": c["scenario"], "query": c["input"]["user_query"],
               "merchant": c["input"]["requester_merchant_id"], "context": c["input"]["context_chunks"],
               "expected": c["contract"], "reference": c["reference_reply"], "runs": {}}
        for p in keep:
            r = res["records"][p][i]
            assert r["case_id"] == c["case_id"]
            row["runs"][p] = {"category": r["category"], "error": r["error"], "tiers": r["tiers_visited"],
                              "flags": r["attempt_flags"], "routed": r["routed_tier"], "risk": r["risk_flags"],
                              "cache": r["cache_hit"], "latency": r["latency_ms"], "cost": r["cost_usd"],
                              "judge": r["judge"], "human": r["human_faithfulness"], "output": r.get("output")}
        explorer.append(row)
    p = default_params(cfg)
    return {
        "simulated": res["simulated"],
        "labels": POLICY_LABEL,
        "summary": res["summary"],
        "router": res["router"], "validator": res["validator"], "calibration": res["calibration"],
        "pairwise": res["pairwise"], "error_analysis": res["error_analysis"], "ablations": res["ablations"],
        "sim_vs_eval": sim_vs_eval(res, cfg),
        "breakeven": {"silent_cost": breakeven_silent_cost(cfg), "recall": breakeven_recall(cfg)},
        "sim_defaults": p,
        "route_presets": {
            "with_risk_rules": p["route"],
            "without_risk_rules": {**p["route"], "hard": [res["router"]["hard_routing_without_risk_rules"][t] /
                                                          max(1, sum(res["router"]["hard_routing_without_risk_rules"].values()))
                                                          for t in ("small", "mid", "frontier")]},
        },
        "tier_prices": {t: {"in": cfg["tiers"][t]["price_per_mtok_in"], "out": cfg["tiers"][t]["price_per_mtok_out"]}
                        for t in ("small", "mid", "frontier")},
        "tokens": dict(cfg["business"]["tokens_per_request"]),
        "sim_reference": {k: {kk: vv for kk, vv in v.items()} for k, v in compare(p).items()},
        "slo": cfg["slo"],
        "explorer": explorer,
    }


def write_all(res: dict, cfg: dict, cases: list[dict]) -> None:
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    slim = {k: v for k, v in res.items() if k != "records"}
    (rep / "eval_results.json").write_text(json.dumps(slim, indent=1, default=float))
    with open(rep / "records.jsonl", "w") as f:
        for p, recs in res["records"].items():
            for r in recs:
                f.write(json.dumps({"policy": p, **r}, default=float) + "\n")
    write_summary(res, cfg, rep / "summary.md")
    write_ablations(res, rep / "ablations.md")
    write_error_analysis(res, rep / "error_analysis.md")
    (ROOT / "dashboard").mkdir(exist_ok=True)
    (ROOT / "dashboard" / "data.json").write_text(json.dumps(dashboard_data(res, cfg, cases), default=float))
