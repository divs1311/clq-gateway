import pytest

from clq.stats import kappa, mcnemar_exact, percentile, weighted_percentile, wilson


def test_wilson_known_value():
    lo, hi = wilson(90, 100)
    assert lo == pytest.approx(0.8256, abs=1e-4) and hi == pytest.approx(0.9448, abs=1e-4)


def test_mcnemar_exact():
    a = [True] * 9 + [False] * 1 + [True] * 5
    b = [False] * 9 + [True] * 1 + [True] * 5
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (9, 1)
    assert r["p_value"] == pytest.approx(2 * 11 / 1024)


def test_kappa():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # 90% raw agreement on an imbalanced set can still be a weak kappa
    h = [1] * 18 + [0, 0]
    j = [1] * 17 + [0] + [1, 0]
    assert kappa(h, j) < 0.65


def test_percentiles_agree():
    v = list(range(1, 101))
    for q in (50, 90, 95, 99):
        assert percentile(v, q) == weighted_percentile(v, [1] * 100, q) == q
