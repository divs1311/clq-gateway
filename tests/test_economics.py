from copy import deepcopy

import pytest

from clq.economics import POLICIES, default_params, evaluate_policy
from clq.util import load_config

P = default_params(load_config())


@pytest.mark.parametrize("policy", POLICIES)
def test_outcomes_sum_to_one(policy):
    r = evaluate_policy(P, policy)
    assert r["prob_mass"] == pytest.approx(1.0)
    assert r["pass_rate"] + r["silent_error_rate"] + r["handoff_rate"] == pytest.approx(1.0)


def test_single_frontier_closed_form():
    p = deepcopy(P)
    p["cache_on"] = False
    r = evaluate_policy(p, "single:frontier")
    exp_pass = sum(p["mix"][d] * p["acc"]["frontier"][d] * (1 - p["false_flag"]) for d in p["mix"])
    exp_silent = sum(p["mix"][d] * (1 - p["acc"]["frontier"][d]) * (1 - p["recall"]) for d in p["mix"])
    assert r["pass_rate"] == pytest.approx(exp_pass)
    assert r["silent_error_rate"] == pytest.approx(exp_silent)
    assert r["model_cost_per_1k"] == pytest.approx(1000 * p["cost_per_call"]["frontier"])


def test_escalated_requests_pay_for_every_attempt():
    p = deepcopy(P)
    p.update(cache_on=False, recall=1.0, false_flag=0.0)
    p["acc"] = {t: {"easy": 0.0, "medium": 0.0, "hard": 0.0} for t in p["acc"]}  # every tier wrong
    r = evaluate_policy(p, "cheap_first")
    assert r["handoff_rate"] == pytest.approx(1.0)
    assert r["model_cost_per_1k"] == pytest.approx(1000 * sum(p["cost_per_call"].values()))


def test_p95_reflects_the_slow_tail():
    # 15% of traffic on a ~3 s tier cannot have a P95 near the fast tier's latency.
    p = deepcopy(P)
    p.update(cache_on=False, recall=0.0, false_flag=0.0, mix={"easy": 0.85, "medium": 0.0, "hard": 0.15})
    p["route"] = {"easy": [1, 0, 0], "medium": [0, 1, 0], "hard": [0, 0, 1]}
    r = evaluate_policy(p, "static_router")
    assert r["latency_ms"]["p95"] > 2000
