import json

from clq.guardrails import validate

CI = {"requester_merchant_id": "12345", "context_chunks": [{"id": "kb_auth_01", "text": "..."}]}


def out(**kw):
    o = {"root_cause_code": "AUTH_TOKEN_EXPIRED",
         "tool_call": {"name": "check_token_validity", "args": {"merchant_id": "12345", "channel": "ShopNova"}},
         "escalate_to_human": False, "customer_reply": "Reconnect.", "cited_chunks": ["kb_auth_01"], "confidence": 0.9}
    o.update(kw)
    return json.dumps(o)


def test_valid_output_passes():
    assert validate(out(), CI, 0.7).ok


def test_schema_invalid():
    v = validate(out()[:30], CI, 0.7)
    assert not v.ok and "SCHEMA_INVALID" in v.flags


def test_extra_key_rejected():
    v = validate(json.dumps({**json.loads(out()), "extra": 1}), CI, 0.7)
    assert "SCHEMA_INVALID" in v.flags


def test_tenant_violation():
    v = validate(out(tool_call={"name": "check_token_validity", "args": {"merchant_id": "99999", "channel": "X"}}), CI, 0.7)
    assert "TENANT_VIOLATION" in v.flags


def test_disallowed_tool_and_bad_args():
    assert "DISALLOWED_TOOL" in validate(out(tool_call={"name": "delete_orders", "args": {"merchant_id": "12345"}}), CI, 0.7).flags
    assert "BAD_TOOL_ARGS" in validate(out(tool_call={"name": "check_token_validity", "args": {"merchant_id": "12345"}}), CI, 0.7).flags


def test_forbidden_pattern_and_citations_and_confidence():
    assert "FORBIDDEN_PATTERN" in validate(out(customer_reply="key sk_live_abcdef123456"), CI, 0.7).flags
    assert "UNKNOWN_CITATION" in validate(out(cited_chunks=["kb_nope"]), CI, 0.7).flags
    assert "MISSING_CITATION" in validate(out(cited_chunks=[]), CI, 0.7).flags
    assert validate(out(root_cause_code="PROMPT_INJECTION", tool_call=None, cited_chunks=[]), CI, 0.7).ok
    assert "LOW_CONFIDENCE" in validate(out(confidence=0.5), CI, 0.7).flags
