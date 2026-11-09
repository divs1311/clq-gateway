import json

import pytest

from clq.cascade import Gateway, attempt_cost
from clq.providers import ProviderResponse
from clq.util import load_config

CFG = load_config()
CI = {"requester_merchant_id": "12345", "user_query": "q", "context_chunks": [{"id": "kb_auth_01", "text": "..."}]}


class FakeProvider:
    """Returns low confidence on every tier listed in `weak`."""

    def __init__(self, weak):
        self.weak = set(weak)

    def complete(self, tier, case_input, case_id):
        conf = 0.1 if tier in self.weak else 0.95
        raw = json.dumps({"root_cause_code": "AUTH_TOKEN_EXPIRED", "tool_call": None, "escalate_to_human": False,
                          "customer_reply": "x", "cited_chunks": ["kb_auth_01"], "confidence": conf})
        return ProviderResponse(raw, 1000, 100, {"small": 100.0, "mid": 500.0, "frontier": 2000.0}[tier], {})


def test_cheap_first_accumulates_cost_and_latency_across_escalations():
    tr = Gateway(CFG, FakeProvider(["small", "mid"]), "cheap_first").handle("c1", CI)
    assert tr.tiers_visited == ["small", "mid", "frontier"] and tr.outcome == "delivered"
    expected_cost = sum(attempt_cost(CFG, t, 1000, 100) for t in ["small", "mid", "frontier"])
    assert tr.cost_usd == pytest.approx(expected_cost)
    assert tr.latency_ms == pytest.approx(2600 + 3 * CFG["overheads"]["validator_ms"])


def test_frontier_failure_hands_off():
    tr = Gateway(CFG, FakeProvider(["small", "mid", "frontier"]), "cheap_first").handle("c1", CI)
    assert tr.outcome == "validator_handoff" and len(tr.attempts) == 3


def test_single_tier_never_escalates():
    tr = Gateway(CFG, FakeProvider(["small"]), "single:small").handle("c1", CI)
    assert tr.tiers_visited == ["small"] and tr.outcome == "validator_handoff"


def test_router_policies_require_router():
    with pytest.raises(ValueError):
        Gateway(CFG, FakeProvider([]), "cascade")
