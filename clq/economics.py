"""Unit-economics model for routing policies at production scale.

Expected values (accuracy, silent errors, hand-offs, cost) are computed EXACTLY by enumerating
every path a request can take. Latency percentiles come from a seeded Monte-Carlo over those
paths, because a sum of log-normal latencies has no closed form. The JavaScript simulator in the
dashboard implements the same algorithm with the same PRNG (mulberry32), so both produce the same
numbers for the same inputs (checked in tests/test_parity.py).

Outcome model (same as the mock provider): one latent draw u ~ U(0,1) per request; tier t answers
correctly iff u < acc[t][difficulty]. So if a cheap tier solves a request, stronger tiers do too.
The validator flags a wrong answer with probability `recall` and a correct one with `false_flag`.
Flagged answers escalate to the next tier (cascade policies) or go to a human.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TIERS = ["small", "mid", "frontier"]
DIFFS = ["easy", "medium", "hard"]
POLICIES = ["single:small", "single:mid", "single:frontier", "static_router", "cheap_first", "cascade"]


# --------------------------------------------------------------------------- PRNG (parity with JS)
def mulberry32(seed: int):
    a = seed & 0xFFFFFFFF
    M = 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & M

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & M) ^ t
        return ((t ^ (t >> 14)) & M) / 4294967296

    return rnd


def normal(rnd) -> float:
    u1 = 1.0 - rnd()   # (0, 1]
    u2 = rnd()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


Z95 = 1.6448536269514722


def lognormal_sample(rnd, median: float, p95: float) -> float:
    sigma = (math.log(p95) - math.log(median)) / Z95
    return math.exp(math.log(median) + sigma * normal(rnd))


# --------------------------------------------------------------------------- params
def default_params(cfg: dict) -> dict:
    b = cfg["business"]
    tok = b["tokens_per_request"]
    return {
        "mix": dict(b["production_mix"]),
        "acc": {t: dict(cfg["mock_priors"]["accuracy"][t]) for t in TIERS},
        "route": {d: list(v) for d, v in cfg["simulator_route_defaults"].items()},
        "cost_per_call": {t: tok["input"] * cfg["tiers"][t]["price_per_mtok_in"] / 1e6
                          + tok["output"] * cfg["tiers"][t]["price_per_mtok_out"] / 1e6 for t in TIERS},
        "latency": {t: dict(cfg["tiers"][t]["latency_ms"]) for t in TIERS},
        "cache_latency": dict(cfg["overheads"]["cache_lookup_ms"]),
        "router_ms": cfg["overheads"]["router_ms"],
        "validator_ms": cfg["overheads"]["validator_ms"],
        "cache_cost": cfg["overheads"]["cache_cost_per_1k"] / 1000,
        "router_cost": cfg["overheads"]["router_cost_per_1k"] / 1000,
        "cache_on": True,
        "cache_hit_easy": b["cache_hit_rate_on_easy"],
        "cache_precision": b["cache_precision"],
        "recall": b["validator"]["recall"],
        "false_flag": b["validator"]["false_flag_rate"],
        "volume": b["monthly_tickets"],
        "human_cost": b["human_cost_per_handoff"],
        "silent_cost": b["silent_error_cost"],
        "mc_samples": 20000,
        "seed": 12345,
    }


# --------------------------------------------------------------------------- model
@dataclass
class Path:
    prob: float
    tiers: tuple
    cache_hit: bool
    router: bool
    outcome: str          # pass | silent | handoff


def enumerate_paths(p: dict, policy: str) -> list[Path]:
    paths: list[Path] = []
    use_router = policy in ("static_router", "cascade")
    escalate = policy in ("cascade", "cheap_first")
    cache_on = bool(p["cache_on"])
    for d in DIFFS:
        md = p["mix"][d]
        if md <= 0:
            continue
        h = p["cache_hit_easy"] if (cache_on and d == "easy") else 0.0
        if h > 0:
            paths.append(Path(md * h * p["cache_precision"], (), True, False, "pass"))
            paths.append(Path(md * h * (1 - p["cache_precision"]), (), True, False, "silent"))
        miss = md * (1 - h)
        if policy.startswith("single:"):
            start = {t: (1.0 if t == policy.split(":")[1] else 0.0) for t in TIERS}
        elif policy == "cheap_first":
            start = {"small": 1.0, "mid": 0.0, "frontier": 0.0}
        else:
            start = dict(zip(TIERS, p["route"][d]))
        cuts = sorted(set([0.0, 1.0] + [min(max(p["acc"][t][d], 0.0), 1.0) for t in TIERS]))
        for si, s in enumerate(TIERS):
            ps = start[s]
            if ps <= 0:
                continue
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                w = hi - lo
                if w <= 0:
                    continue
                mid_u = (lo + hi) / 2
                mass = miss * ps * w
                visited = []
                for ti in range(si, len(TIERS)):
                    t = TIERS[ti]
                    visited.append(t)
                    correct = mid_u < p["acc"][t][d]
                    f = p["false_flag"] if correct else p["recall"]
                    deliver = mass * (1 - f)
                    if deliver > 0:
                        paths.append(Path(deliver, tuple(visited), False, use_router, "pass" if correct else "silent"))
                    mass *= f
                    if mass <= 0:
                        break
                    if not (escalate and ti < len(TIERS) - 1):
                        paths.append(Path(mass, tuple(visited), False, use_router, "handoff"))
                        break
    return paths


def evaluate_policy(p: dict, policy: str) -> dict:
    paths = enumerate_paths(p, policy)
    total = sum(x.prob for x in paths)
    use_router = policy in ("static_router", "cascade")
    out = {"policy": policy, "prob_mass": total}
    rate = lambda pred: sum(x.prob for x in paths if pred(x))  # noqa: E731
    out["pass_rate"] = rate(lambda x: x.outcome == "pass")
    out["silent_error_rate"] = rate(lambda x: x.outcome == "silent")
    out["handoff_rate"] = rate(lambda x: x.outcome == "handoff")
    out["escalation_rate"] = rate(lambda x: len(x.tiers) > 1)
    out["cache_hit_rate"] = rate(lambda x: x.cache_hit)
    out["final_tier_share"] = {t: rate(lambda x, t=t: bool(x.tiers) and x.tiers[-1] == t) for t in TIERS}
    model = sum(x.prob * sum(p["cost_per_call"][t] for t in x.tiers) for x in paths)
    overhead = (p["cache_cost"] if p["cache_on"] else 0.0) + (p["router_cost"] * (1 - out["cache_hit_rate"]) if use_router else 0.0)
    out["model_cost_per_1k"] = 1000 * (model + overhead)
    out["human_cost_per_1k"] = 1000 * out["handoff_rate"] * p["human_cost"]
    out["silent_cost_per_1k"] = 1000 * out["silent_error_rate"] * p["silent_cost"]
    out["fully_loaded_per_1k"] = out["model_cost_per_1k"] + out["human_cost_per_1k"] + out["silent_cost_per_1k"]
    out["monthly_model_cost"] = out["model_cost_per_1k"] * p["volume"] / 1000
    out["monthly_fully_loaded"] = out["fully_loaded_per_1k"] * p["volume"] / 1000
    out["latency_ms"] = latency_percentiles(p, paths)
    return out


def latency_percentiles(p: dict, paths: list[Path]) -> dict:
    rnd = mulberry32(p["seed"])
    n = int(p["mc_samples"])
    total = sum(x.prob for x in paths)
    cum, acc = [], 0.0
    for x in paths:
        acc += x.prob / total
        cum.append(acc)
    samples = []
    for _ in range(n):
        u = rnd()
        k = 0
        while k < len(cum) - 1 and u >= cum[k]:
            k += 1
        x = paths[k]
        ms = 0.0
        if p["cache_on"]:
            ms += lognormal_sample(rnd, p["cache_latency"]["median"], p["cache_latency"]["p95"])
        if x.router:
            ms += p["router_ms"]
        for t in x.tiers:
            ms += lognormal_sample(rnd, p["latency"][t]["median"], p["latency"][t]["p95"]) + p["validator_ms"]
        samples.append(ms)
    samples.sort()

    def pct(q):
        return samples[max(1, math.ceil(q / 100 * n)) - 1]

    return {"p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99)}


def compare(p: dict) -> dict:
    return {pol: evaluate_policy(p, pol) for pol in POLICIES}
