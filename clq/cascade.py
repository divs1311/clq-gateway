"""The gateway: semantic cache -> complexity router -> tiered model pool -> validator -> escalate.

Policies
  single:<tier>  every request goes to one tier; a flagged answer goes to a human
  static_router  router picks a tier; a flagged answer goes to a human (no model escalation)
  cascade        router picks a starting tier; a flagged answer escalates to the next tier;
                 a flagged frontier answer goes to a human
  cheap_first    like cascade but always starts at the small tier (no router)

Cost and latency ACCUMULATE across every attempt, so an escalated request pays for the cheap
attempt(s) it went through. That is the main difference from a naive "static split" model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .guardrails import validate
from .router import TIERS
from .util import hash_normal, lognormal_params

import math


@dataclass
class Attempt:
    tier: str
    latency_ms: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
    flags: list[str]
    raw: str
    meta: dict


@dataclass
class Trace:
    case_id: str
    policy: str
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    routed_tier: Optional[str] = None
    router_proba: Optional[list[float]] = None
    risk_flags: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    overhead_ms: float = 0.0
    overhead_cost_usd: float = 0.0
    final_output: Optional[dict] = None
    outcome: str = ""                      # delivered | validator_handoff

    @property
    def latency_ms(self) -> float:
        return self.overhead_ms + sum(a.latency_ms for a in self.attempts)

    @property
    def cost_usd(self) -> float:
        return self.overhead_cost_usd + sum(a.cost_usd for a in self.attempts)

    @property
    def tiers_visited(self) -> list[str]:
        return [a.tier for a in self.attempts]


def attempt_cost(cfg: dict, tier: str, tin: int, tout: int) -> float:
    t = cfg["tiers"][tier]
    return tin * t["price_per_mtok_in"] / 1e6 + tout * t["price_per_mtok_out"] / 1e6


class Gateway:
    def __init__(self, cfg: dict, provider, policy: str, router=None, cache=None):
        if policy.startswith("single:"):
            assert policy.split(":")[1] in TIERS
        else:
            assert policy in {"static_router", "cascade", "cheap_first"}, policy
        if policy in {"static_router", "cascade"} and router is None:
            raise ValueError(f"{policy} needs a router")
        self.cfg, self.provider, self.policy, self.router, self.cache = cfg, provider, policy, router, cache

    def handle(self, case_id: str, case_input: dict) -> Trace:
        cfg, ov = self.cfg, self.cfg["overheads"]
        tr = Trace(case_id, self.policy + ("+cache" if self.cache else ""))

        if self.cache is not None:
            lk = ov["cache_lookup_ms"]
            mu, sg = lognormal_params(lk["median"], lk["p95"])
            tr.overhead_ms += math.exp(mu + sg * hash_normal(case_id, "cache_latency"))
            tr.overhead_cost_usd += ov["cache_cost_per_1k"] / 1000
            hit = self.cache.lookup(case_input["user_query"], {c["id"] for c in case_input["context_chunks"]})
            if hit is not None:
                tr.cache_hit, tr.cache_similarity = True, hit.similarity
                tr.final_output, tr.outcome = hit.output, "delivered"
                return tr
            tr.cache_similarity = self.cache.best(case_input["user_query"])[1]

        if self.policy.startswith("single:"):
            start, escalate = self.policy.split(":")[1], False
        elif self.policy == "cheap_first":
            start, escalate = "small", True
        else:
            tr.overhead_ms += ov["router_ms"]
            tr.overhead_cost_usd += ov["router_cost_per_1k"] / 1000
            start, p, risk = (self.router.route_case(case_id, case_input) if hasattr(self.router, "route_case")
                              else self.router.route(case_input))
            tr.routed_tier, tr.router_proba, tr.risk_flags = start, [round(float(x), 4) for x in p], risk
            escalate = self.policy == "cascade"

        tier_idx = TIERS.index(start)
        while True:
            tier = TIERS[tier_idx]
            r = self.provider.complete(tier, case_input, case_id)
            v = validate(r.raw, case_input, cfg["tiers"][tier]["confidence_threshold"])
            tr.overhead_ms += ov["validator_ms"]
            tr.attempts.append(Attempt(tier, r.latency_ms, attempt_cost(cfg, tier, r.tokens_in, r.tokens_out),
                                       r.tokens_in, r.tokens_out, v.flags, r.raw, r.meta))
            if v.ok:
                tr.final_output, tr.outcome = v.output.model_dump(mode="json"), "delivered"
                return tr
            if escalate and tier_idx < len(TIERS) - 1:
                tier_idx += 1
                continue
            tr.outcome = "validator_handoff"
            return tr
