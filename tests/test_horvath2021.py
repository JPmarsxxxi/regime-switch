import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from regime_switch import horvath2021 as hv


def test_lift_overlap_and_membership():
    x = np.arange(100.0)
    segs, starts = hv.lift(x, 35, 28)
    assert segs.shape[1] == 35 and np.all(np.diff(starts) == 7)
    y_off, _ = hv.label_counts(100, starts, 35, np.zeros(len(starts), dtype=bool))
    assert y_off.max() == 5  # "a single return can potentially belong to 5 different empirical measures"


def test_w1_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.standard_normal(35), rng.standard_t(3, 35)
        assert hv.wasserstein(np.sort(a), np.sort(b), 1) == pytest.approx(wasserstein_distance(a, b), abs=1e-12)


def test_median_barycenter_minimises_sum_w1():
    rng = np.random.default_rng(1)
    S = np.sort(rng.standard_normal((40, 20)) * rng.uniform(0.5, 2, (40, 1)), axis=1)
    bc = hv.barycenter(S, 1)
    best = hv.wasserstein(S, bc, 1).sum()
    for _ in range(1000):
        cand = np.sort(bc + rng.normal(0, 0.05, bc.size))
        assert hv.wasserstein(S, cand, 1).sum() >= best - 1e-12


def test_accuracy_hand_example():
    truth = np.array([False, False, True, True])
    y_off = np.array([2.0, 1.0, 1.0, 0.0])
    y_on = np.array([0.0, 1.0, 1.0, 2.0])
    a = hv.accuracy(y_off, y_on, truth)
    assert a["ROFS"] == pytest.approx(3 / 4)
    assert a["RONS"] == pytest.approx(3 / 4)
    assert a["TA"] == pytest.approx(6 / 8)


def test_wk_means_separates_two_volatility_levels():
    rng = np.random.default_rng(2)
    segs = np.vstack([rng.normal(0, 1, (100, 35)), rng.normal(0, 3, (100, 35))])
    labels, _ = hv.wk_means(segs, seed=0)
    assert len(set(labels[:100])) == 1 and len(set(labels[100:])) == 1 and labels[0] != labels[-1]


def test_mmd_zero_for_identical_and_positive_otherwise():
    x = np.linspace(-1, 1, 30)
    assert hv.mmd2_biased(x, x) == pytest.approx(0.0, abs=1e-12)
    assert hv.mmd2_biased(x, x + 0.5) > 0
