"""Statistics used by the eval harness. Kept small and explicit so it can be audited."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy.stats import binom
from sklearn.metrics import cohen_kappa_score


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired pass/fail vectors.

    b01 = cases A passed and B failed; b10 = A failed and B passed.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape:
        raise ValueError("paired vectors must align")
    b01 = int(np.sum(a & ~b))
    b10 = int(np.sum(~a & b))
    n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b01, b10), n, 0.5))
    return {"a_only": b01, "b_only": b10, "discordant": n, "p_value": float(p)}


def kappa(y1: Sequence, y2: Sequence) -> float:
    y1, y2 = list(y1), list(y2)
    if len(set(y1) | set(y2)) < 2:
        return float("nan")  # undefined when both raters use a single label
    return float(cohen_kappa_score(y1, y2))


def bootstrap_ci(stat: Callable[[np.ndarray], float], idx_n: int, b: int = 2000,
                 seed: int = 7, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over case indices. `stat` receives an index array."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        s = stat(rng.integers(0, idx_n, idx_n))
        if not math.isnan(s):
            vals.append(s)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def kappa_with_ci(y1: Sequence, y2: Sequence, b: int = 2000, seed: int = 7) -> dict:
    y1, y2 = np.asarray(y1), np.asarray(y2)
    point = kappa(y1, y2)
    lo, hi = bootstrap_ci(lambda i: kappa(y1[i], y2[i]), len(y1), b=b, seed=seed)
    agree = float(np.mean(y1 == y2)) if len(y1) else float("nan")
    return {"kappa": point, "ci_low": lo, "ci_high": hi, "raw_agreement": agree, "n": int(len(y1))}


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]). Same definition as the JS simulator."""
    v = sorted(values)
    if not v:
        return float("nan")
    k = max(1, math.ceil(q / 100 * len(v)))
    return float(v[k - 1])


def weighted_percentile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Nearest-rank percentile where each value carries a weight (used for production re-weighting)."""
    if not len(values):
        return float("nan")
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, float)[order]
    w = np.asarray(weights, float)[order]
    cum = np.cumsum(w) / np.sum(w)
    k = int(np.searchsorted(cum, q / 100 - 1e-12, side="left"))
    return float(v[min(k, len(v) - 1)])
