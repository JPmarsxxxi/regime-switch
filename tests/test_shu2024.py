import numpy as np
import pandas as pd

from regime_switch import shu2024
from regime_switch.sim import markov_switching_gaussian


def reference_features(ret):
    # the authors' example code (jump-models/examples/nasdaq/feature.py), with the paper's Table 2 halflives
    def dd(hl):
        return np.sqrt(np.minimum(ret, 0.0).pow(2).ewm(halflife=hl).mean())

    return pd.DataFrame({"DD_10": dd(10), "Sortino_20": ret.ewm(halflife=20).mean().div(dd(20)),
                         "Sortino_60": ret.ewm(halflife=60).mean().div(dd(60))})


def _series(n=6000, seed=0):
    p = markov_switching_gaussian(n, seed)
    idx = pd.bdate_range("1990-01-01", periods=n)
    return pd.Series(np.expm1(p.log_returns), index=idx), p


def test_features_match_authors_definition():
    r, _ = _series(3000)
    pd.testing.assert_frame_equal(shu2024.features(r), reference_features(r), rtol=1e-12, atol=0)


def test_jm_online_has_no_lookahead():
    r, _ = _series(4200)
    cfg = shu2024.Config(train=1500, lookback=1500, n_init=3)
    a = shu2024.jm_online(shu2024.features(r), r, 35.0, cfg)
    r2 = r.copy()
    cut = 3500
    r2.iloc[cut:] = r2.iloc[cut:] * -3 + 0.02  # rewrite the future
    b = shu2024.jm_online(shu2024.features(r2), r2, 35.0, cfg)
    pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])


def test_hmm_online_has_no_lookahead():
    r, _ = _series(1700)
    cfg = shu2024.Config(train=1500, n_init=2, workers=1)
    logr = np.log1p(r)
    a = shu2024.hmm_online_raw(logr, cfg)
    logr2 = logr.copy()
    logr2.iloc[1650:] = 0.05
    b = shu2024.hmm_online_raw(logr2, cfg)
    pd.testing.assert_series_equal(a.iloc[:1650], b.iloc[:1650])


def test_median_filter():
    raw = pd.Series([0, 1, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    out = shu2024.median_filter(raw, 3)
    assert out.isna().sum() == 2
    assert list(out.iloc[2:]) == [0, 1, 1, 1, 1, 0, 0]


def test_cv_uses_only_trailing_window():
    r, _ = _series(3000)
    rf = pd.Series(0.0, index=r.index)
    good = (r.rolling(5).mean() < 0).astype(float).shift(-3)  # cheats by peeking 3 days ahead
    bad = 1 - good
    cands = {"good": good.fillna(0), "bad": bad.fillna(0)}
    cfg = shu2024.Config(val_years=2)
    bear, choice = shu2024.cross_validate(cands, r, rf, cfg, r.index[600])
    assert (choice["choice"] == "good").mean() > 0.9
    assert bear.loc[choice.index[0]:].notna().all()
