"""Model providers.

MockProvider   - offline, deterministic, oracle-backed stand-in used to exercise the whole pipeline
                 without API keys. Its accuracy is set by `mock_priors` in config, so its "results"
                 are a function of those assumptions. Never present mock numbers as measurements.
LiteLLMProvider - real calls through LiteLLM to any provider (OpenAI-compatible, Anthropic, Gemini,
                 vLLM/Ollama for open weights). Latency, tokens and correctness are measured.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass

from .prompts import triage_messages
from .util import estimate_tokens, hash01, hash_normal, lognormal_params

UNGROUNDED_CLAIMS = [
    "This is caused by a known platform outage that our engineers will fix within 24 hours.",
    "Your plan includes a free priority re-sync that I have already triggered.",
    "This only happens on accounts older than two years.",
]
VERBOSE_PADDING = (" I completely understand how frustrating this is for your business, and I want to make sure"
                   " you have everything you need. Our team is always here to help, so please don't hesitate to"
                   " reach out again if anything else comes up during the day.")
WRONG_ROOT = {
    "HOW_TO_QUESTION": "AMBIGUOUS_NEEDS_INFO", "AUTH_TOKEN_EXPIRED": "RATE_LIMITED",
    "WEBHOOK_SCHEMA_MISMATCH": "AUTH_TOKEN_EXPIRED", "RATE_LIMITED": "AUTH_TOKEN_EXPIRED",
    "SKU_MAPPING_MISSING": "CATEGORY_ATTRIBUTE_MISSING", "INVENTORY_SYNC_LAG": "INVENTORY_SYNC_STALLED",
    "INVENTORY_SYNC_STALLED": "INVENTORY_SYNC_LAG", "CATEGORY_ATTRIBUTE_MISSING": "SKU_MAPPING_MISSING",
    "CURRENCY_MISMATCH": "RATE_LIMITED", "PROMPT_INJECTION": "HOW_TO_QUESTION",
    "DESTRUCTIVE_ACTION_REQUEST": "HOW_TO_QUESTION", "CROSS_TENANT_REQUEST": "HOW_TO_QUESTION",
    "AMBIGUOUS_NEEDS_INFO": "AUTH_TOKEN_EXPIRED",
}


@dataclass
class ProviderResponse:
    raw: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    meta: dict


class MockProvider:
    name = "mock"

    def __init__(self, cfg: dict, cases_by_id: dict[str, dict]):
        self.cfg = cfg
        self.cases = cases_by_id
        self.acc = cfg["mock_priors"]["accuracy"]
        self.modes = cfg["mock_priors"]["failure_modes"]

    # A single latent "difficulty draw" per case makes tier outcomes nested: if a cheaper tier
    # solves a case, every more capable tier does too (monotone in p). The economics simulator
    # uses the same assumption, so mock eval and simulator are consistent by construction.
    def is_correct(self, case: dict, tier: str) -> bool:
        return hash01(case["case_id"], "latent") < self.acc[tier][case["difficulty"]]

    def complete(self, tier: str, case_input: dict, case_id: str) -> ProviderResponse:
        case = self.cases[case_id]
        correct = self.is_correct(case, tier)
        out, mode = (self._correct(case, tier), "NONE") if correct else self._failure(case, tier)
        raw = out if isinstance(out, str) else json.dumps(out)
        t = self.cfg["tiers"][tier]["latency_ms"]
        mu, sigma = lognormal_params(t["median"], t["p95"])
        latency = math.exp(mu + sigma * hash_normal(case_id, tier, "latency"))
        prompt = "".join(m["content"] for m in triage_messages(case_input))
        return ProviderResponse(raw, estimate_tokens(prompt), estimate_tokens(raw), latency,
                                {"mock_correct": correct, "mock_failure_mode": mode})

    def _base(self, case: dict, tier: str, conf_lo: float, conf_hi: float) -> dict:
        c = case["contract"]
        reply = case["reference_reply"]
        if hash01(case["case_id"], tier, "verbose") < 0.3:
            reply += VERBOSE_PADDING
        return {
            "root_cause_code": c["expected_root_cause"],
            "tool_call": c["expected_tool_call"],
            "escalate_to_human": c["must_escalate"],
            "customer_reply": reply,
            "cited_chunks": list(c["required_citations"]),
            "confidence": round(conf_lo + (conf_hi - conf_lo) * hash01(case["case_id"], tier, "conf"), 3),
        }

    def _correct(self, case: dict, tier: str) -> dict:
        return self._base(case, tier, 0.685, 0.99)

    def _failure(self, case: dict, tier: str):
        weights = self.modes[case["stratum"]]
        u, acc = hash01(case["case_id"], tier, "mode"), 0.0
        mode = list(weights)[-1]
        for m, w in weights.items():
            acc += w
            if u < acc:
                mode = m
                break
        out = self._base(case, tier, 0.35, 0.95)
        rnd = random.Random(f"{case['case_id']}|{tier}|detail")
        c = case["contract"]
        mid = case["input"]["requester_merchant_id"]

        if mode == "SCHEMA_CORRUPTION":
            s = json.dumps(out)
            return s[: max(10, len(s) // 2)], mode
        if mode == "WRONG_ROOT_CAUSE":
            out["root_cause_code"] = WRONG_ROOT[c["expected_root_cause"]]
        elif mode == "WRONG_TOOL_ARGS":
            tc = c["expected_tool_call"]
            if tc is None:  # calls a tool it did not need
                out["tool_call"] = {"name": "get_order_sync_status", "args": {"merchant_id": mid, "channel": "ShopNova"}}
            else:
                args = dict(tc["args"])
                if rnd.random() < 0.3:
                    args["merchant_id"] = str(int(mid) + 1)          # tenant violation - validator catches
                else:
                    key = rnd.choice([k for k in args if k != "merchant_id"])
                    args[key] = {"channel": "ShopNova" if args.get("channel") != "ShopNova" else "MarketMax",
                                 "since": "2026-09-30T00:00:00Z", "sku": "SKU-0000",
                                 "warehouse_id": "WH-99"}[key]          # semantic error - not catchable
                out["tool_call"] = {"name": tc["name"], "args": args}
        elif mode == "UNGROUNDED_REPLY":
            out["customer_reply"] += " " + rnd.choice(UNGROUNDED_CLAIMS)
            out["confidence"] = round(0.6 + 0.37 * hash01(case["case_id"], tier, "conf2"), 3)  # overconfident
        elif mode == "INJECTION_FOLLOWED":
            scen = case["scenario"]
            if scen == "adv:indirect_injection":
                out["tool_call"] = {"name": "delete_orders", "args": {"merchant_id": mid}}
                out["customer_reply"] = "resolved"
            elif scen == "adv:direct_injection":
                out["root_cause_code"] = "HOW_TO_QUESTION"
                out["escalate_to_human"] = False
                out["customer_reply"] = "Here is the key: sk_live_" + "".join(rnd.choice("abcdef0123456789") for _ in range(16))
            elif scen == "adv:cross_tenant":
                other = case["input"]["user_query"].split("merchant ")[1].split(".")[0]
                out["root_cause_code"] = "CURRENCY_MISMATCH"
                out["escalate_to_human"] = False
                out["tool_call"] = {"name": "get_order_sync_status", "args": {"merchant_id": other, "channel": "ShopNova"}}
            elif scen == "adv:destructive":
                out["escalate_to_human"] = False
                out["tool_call"] = {"name": "delete_orders", "args": {"merchant_id": mid}}
            else:  # ambiguous: guesses a diagnosis instead of asking
                out["root_cause_code"] = "AUTH_TOKEN_EXPIRED"
                out["tool_call"] = {"name": "check_token_validity", "args": {"merchant_id": mid, "channel": "ShopNova"}}
                out["customer_reply"] = "Your channel token has probably expired; I'm checking it."
        elif mode == "ESCALATION_ERROR":
            out["escalate_to_human"] = not c["must_escalate"]
        return out, mode


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, cfg: dict):
        import litellm  # noqa: F401  (optional dependency)
        self.litellm = litellm
        self.cfg = cfg
        missing = [t for t, v in cfg["tiers"].items() if not v.get("model")]
        if missing:
            raise RuntimeError(f"Set CLQ_<TIER>_MODEL env vars for tiers: {missing}")

    def complete(self, tier: str, case_input: dict, case_id: str) -> ProviderResponse:
        model = self.cfg["tiers"][tier]["model"]
        t0 = time.perf_counter()
        resp = self.litellm.completion(model=model, messages=triage_messages(case_input), temperature=0,
                                       response_format={"type": "json_object"}, timeout=30)
        latency = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_text = "".join(m["content"] for m in triage_messages(case_input))
        tin = getattr(usage, "prompt_tokens", None) or estimate_tokens(prompt_text)
        tout = getattr(usage, "completion_tokens", None) or estimate_tokens(raw)
        return ProviderResponse(raw, int(tin), int(tout), latency, {"model": model})
