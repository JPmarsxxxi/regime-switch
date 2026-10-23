import numpy as np
import pandas as pd
import pytest

from regime_switch import hmm, wk_hmm
from regime_switch import horvath2021 as hv
from regime_switch.sim import markov_switching_gaussian


def _returns(n, seed=0):
    p = markov_switching_gaussian(n, seed)
    return pd.Series(p.log_returns, index=pd.bdate_range("2000-01-03", periods=n))


@pytest.mark.parametrize("variant", ["A", "B"])
def test_online_has_no_lookahead(variant):
    x = _returns(1700)
    a = wk_hmm.online(x, variant, train=1000)
    x2 = x.copy()
    x2.iloc[1400:] = -x2.iloc[1400:] * 3
    b = wk_hmm.online(x2, variant, train=1000)
    pd.testing.assert_series_equal(a.iloc[:1400], b.iloc[:1400])
    first = wk_hmm.refit_dates(x.index, x.index[999], 6)[0]
    assert a.loc[first:].notna().all() and a.loc[: first - pd.Timedelta(days=1)].isna().all()


def test_tau_is_mean_within_cluster_distance():
    x = _returns(1500).to_numpy()
    p = wk_hmm.fit_a(x, seed=3)
    _, segs, lab, cent, on = wk_hmm.wk_daily_labels(x, 3)
    S = np.sort(segs, axis=1)
    for state, k in enumerate([1 - on, on]):
        assert p["tau"][state] == pytest.approx(np.mean([hv.wasserstein(s, cent[k], 1) for s in S[lab == k]]))


def test_weighted_kde_integrates_to_one():
    rng = np.random.default_rng(0)
    c, w = rng.standard_normal(300) * 0.01, rng.random(300)
    h = wk_hmm.scott_bandwidth(c, w)
    grid = np.linspace(-0.1, 0.1, 20001)
    assert np.trapezoid(np.exp(wk_hmm.kde_logpdf(grid, c, w, h)), grid) == pytest.approx(1.0, abs=1e-3)


def test_forward_filter_matches_gaussian_hmm_filter():
    x = _returns(800).to_numpy()
    mu, var = np.array([0.0005, -0.001]), np.array([0.007, 0.02]) ** 2
    G, d = np.array([[0.99, 0.01], [0.03, 0.97]]), np.array([0.6, 0.4])
    logE = -0.5 * ((x[:, None] - mu) ** 2 / var + np.log(2 * np.pi * var))
    assert wk_hmm.forward_filter(logE, G, d) == pytest.approx(hmm.forward(x, mu, var, G, d)[0], abs=1e-12)


def test_transition_counts_with_smoothing():
    G = wk_hmm.transition_counts(np.array([0, 0, 0, 1, 1, 0]))
    assert G == pytest.approx(np.array([[3 / 5, 2 / 5], [2 / 4, 2 / 4]]))
