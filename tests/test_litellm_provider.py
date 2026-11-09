"""The real provider path, exercised against a stub litellm module (no network)."""
import json
import sys
import types
from copy import deepcopy

from clq.cascade import Gateway
from clq.util import load_config


def test_litellm_provider_with_stub(monkeypatch):
    calls = []

    def completion(model, messages, **kw):
        calls.append(model)
        content = json.dumps({"root_cause_code": "HOW_TO_QUESTION", "tool_call": None, "escalate_to_human": False,
                              "customer_reply": "Go to Settings.", "cited_chunks": ["k"], "confidence": 0.9})
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=800, completion_tokens=60))

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    from clq.providers import LiteLLMProvider
    cfg = deepcopy(load_config())
    for t, m in zip(["small", "mid", "frontier"], ["s-model", "m-model", "f-model"]):
        cfg["tiers"][t]["model"] = m
    gw = Gateway(cfg, LiteLLMProvider(cfg), "single:mid")
    tr = gw.handle("x", {"user_query": "how?", "requester_merchant_id": "1", "merchant_tier": "standard",
                         "context_chunks": [{"id": "k", "text": "Settings."}]})
    assert calls == ["m-model"] and tr.outcome == "delivered"
    assert tr.attempts[0].tokens_in == 800 and tr.attempts[0].tokens_out == 60
