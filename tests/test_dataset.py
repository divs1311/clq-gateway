from collections import Counter

from clq.schemas import TOOL_SIGNATURES, GoldenCase
from clq.util import load_cases


def test_counts_and_schema():
    cases = load_cases()
    assert len(cases) == 200
    assert Counter(c["stratum"] for c in cases) == {"routine": 120, "multi_hop": 50, "adversarial": 30}
    assert len({c["case_id"] for c in cases}) == 200
    for c in cases:
        GoldenCase.model_validate(c)


def test_contracts_are_internally_consistent():
    for c in load_cases():
        ctx = [ch["id"] for ch in c["input"]["context_chunks"]]
        assert len(ctx) == len(set(ctx))
        assert set(c["contract"]["required_citations"]) <= set(ctx)
        tc = c["contract"]["expected_tool_call"]
        if tc:
            assert sorted(tc["args"]) == sorted(TOOL_SIGNATURES[tc["name"]])
            assert tc["args"]["merchant_id"] == c["input"]["requester_merchant_id"]
        if c["contract"]["must_escalate"]:
            assert c["stratum"] == "adversarial"
