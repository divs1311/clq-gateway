"""Complexity router.

A multinomial logistic regression over TF-IDF features predicts P(difficulty = easy/medium/hard).
Routing rule (RouteLLM-style threshold): send the request to the cheapest tier t such that
P(difficulty <= t) >= alpha. Higher alpha = more conservative = more traffic to expensive tiers.

Evaluation uses out-of-fold (cross-validated) probabilities so the router is never scored on
cases it was trained on.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

LABELS = ["easy", "medium", "hard"]

# --------------------------------------------------------------------------- risk pre-filter
# Deterministic risk signals force the frontier tier regardless of predicted complexity.
# Caveat: these rules were written knowing the adversarial categories in the golden set, so their
# perfect score there is NOT evidence of production recall. Validate on a held-out red-team set.
INJECTION_MARKERS = [r"(?i)ignore (all )?(previous|prior) instructions", r"(?i)system override",
                     r"(?i)admin mode", r"(?i)note to assistant", r"(?i)developer console", r"(?i)sk_live"]
SENSITIVE_REQUESTS = [r"(?i)\b(reveal|print|show|output|give|send)\b.{0,40}\b(secret|credentials?|api key)\b",
                      r"(?i)\b(delete|wipe|remove|purge)\b.{0,25}\b(all|every)\b"]
MERCHANT_REF = re.compile(r"(?i)\b(?:merchant|account|acct|shop)(?:\s+id)?\s*(?:is\s*)?#?:?\s*(\d{5})\b")


def risk_flags(case_input: dict) -> list[str]:
    q = case_input["user_query"]
    everything = q + " " + " ".join(c["text"] for c in case_input["context_chunks"])
    flags = []
    if any(re.search(p, everything) for p in INJECTION_MARKERS):   # direct or indirect (in retrieved text)
        flags.append("injection_marker")
    if any(re.search(p, q) for p in SENSITIVE_REQUESTS):
        flags.append("sensitive_request")
    if set(MERCHANT_REF.findall(q)) - {case_input["requester_merchant_id"]}:
        flags.append("other_merchant_id")
    if len(q.split()) < 6 and not re.search(r"\d", q):
        flags.append("low_information")
    return flags


TIERS = ["small", "mid", "frontier"]


def router_text(case_input: dict, use_context: bool = True) -> str:
    text = case_input["user_query"]
    if use_context:
        text += " || " + " | ".join(c["text"] for c in case_input["context_chunks"])
    return text


def make_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                         LogisticRegression(max_iter=5000, C=1.0))


def oof_probabilities(cases: list[dict], use_context: bool = True, folds: int = 5, seed: int = 0) -> np.ndarray:
    x = [router_text(c["input"], use_context) for c in cases]
    y = [c["difficulty"] for c in cases]
    model = make_model()
    proba = cross_val_predict(model, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                              method="predict_proba")
    classes = sorted(set(y))  # sklearn orders classes alphabetically: easy, hard, medium
    order = [classes.index(lbl) for lbl in LABELS]
    return proba[:, order]


def choose_tier(p: np.ndarray, alpha: float) -> str:
    """p = [P(easy), P(medium), P(hard)]."""
    cum = np.cumsum(p)
    for i, t in enumerate(TIERS):
        if cum[i] >= alpha - 1e-12:
            return t
    return TIERS[-1]


class ComplexityRouter:
    def __init__(self, alpha: float, use_context: bool = True, use_risk_rules: bool = True):
        self.alpha, self.use_context, self.use_risk_rules = alpha, use_context, use_risk_rules
        self.model = make_model()

    def fit(self, cases: list[dict]) -> "ComplexityRouter":
        self.model.fit([router_text(c["input"], self.use_context) for c in cases], [c["difficulty"] for c in cases])
        return self

    def proba(self, case_input: dict) -> np.ndarray:
        p = self.model.predict_proba([router_text(case_input, self.use_context)])[0]
        classes = list(self.model.classes_)
        return np.array([p[classes.index(lbl)] for lbl in LABELS])

    def route(self, case_input: dict) -> tuple[str, np.ndarray, list[str]]:
        p = self.proba(case_input)
        flags = risk_flags(case_input) if self.use_risk_rules else []
        return ("frontier" if flags else choose_tier(p, self.alpha)), p, flags


class FixedProbaRouter:
    """Wraps precomputed out-of-fold probabilities for leakage-free evaluation."""

    def __init__(self, proba_by_case: dict[str, np.ndarray], alpha: float, use_risk_rules: bool = True):
        self.p, self.alpha, self.use_risk_rules = proba_by_case, alpha, use_risk_rules

    def route_case(self, case_id: str, case_input: dict) -> tuple[str, np.ndarray, list[str]]:
        p = self.p[case_id]
        flags = risk_flags(case_input) if self.use_risk_rules else []
        return ("frontier" if flags else choose_tier(p, self.alpha)), p, flags
