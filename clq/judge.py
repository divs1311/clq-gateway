"""LLM-as-a-judge (real) and a SIMULATED judge/human pair for the offline demo.

The simulated judge has deliberately injected flaws (5% false-fail, verbosity-biased false-pass,
20% first-position bias) so the calibration and swap-consistency machinery has something real to
detect. The simulated "human" is an oracle that knows which replies contain an unsupported claim.
Replace both with LLMJudge + your own labels (clq label) before quoting any kappa.
"""
from __future__ import annotations

import json

from .prompts import judge_messages, pairwise_messages
from .providers import UNGROUNDED_CLAIMS
from .util import hash01

LONG_REPLY_CHARS = 250


def is_grounded_oracle(reply: str) -> bool:
    return not any(c in reply for c in UNGROUNDED_CLAIMS)


class SimulatedHuman:
    name = "simulated_human"

    def faithfulness(self, case: dict, reply: str) -> int:
        return 2 if is_grounded_oracle(reply) else 0


class MockJudge:
    name = "mock_judge"

    def __init__(self, false_fail=0.05, false_pass_short=0.10, false_pass_long=0.35, position_bias=0.20):
        self.ff, self.fps, self.fpl, self.pb = false_fail, false_pass_short, false_pass_long, position_bias

    def grade(self, case: dict, reply: str) -> dict:
        u = hash01(case["case_id"], reply, "judge")
        if is_grounded_oracle(reply):
            f = 0 if u < self.ff else 2
        else:
            f = 2 if u < (self.fpl if len(reply) > LONG_REPLY_CHARS else self.fps) else 0
        return {"factual_faithfulness": f, "helpfulness": 2 if f == 2 else 1,
                "rationale": "simulated judge verdict"}

    def pairwise(self, case: dict, a: str, b: str) -> str:
        if hash01(case["case_id"], a, b, "pos") < self.pb:
            return "A"  # position bias: picks whatever is shown first
        ga, gb = is_grounded_oracle(a), is_grounded_oracle(b)
        return "tie" if ga == gb else ("A" if ga else "B")


class LLMJudge:  # pragma: no cover - needs network + keys
    name = "llm_judge"

    def __init__(self, model: str):
        import litellm
        self.litellm, self.model = litellm, model

    def _call(self, messages) -> dict:
        r = self.litellm.completion(model=self.model, messages=messages, temperature=0,
                                    response_format={"type": "json_object"})
        return json.loads(r.choices[0].message.content)

    def grade(self, case: dict, reply: str) -> dict:
        return self._call(judge_messages(case, reply))

    def pairwise(self, case: dict, a: str, b: str) -> str:
        return self._call(pairwise_messages(case, a, b)).get("winner", "tie")


def swap_consistent(judge, case: dict, x: str, y: str) -> dict:
    """Ask in both orders; keep the verdict only if it survives the swap."""
    first = judge.pairwise(case, x, y)            # A=x, B=y
    second = judge.pairwise(case, y, x)           # A=y, B=x
    map1 = {"A": "x", "B": "y", "tie": "tie"}[first]
    map2 = {"A": "y", "B": "x", "tie": "tie"}[second]
    return {"consistent": map1 == map2, "verdict": map1 if map1 == map2 else None,
            "order_xy": first, "order_yx": second}
