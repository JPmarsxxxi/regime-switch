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


def test_default_test_start_waits_for_the_latest_candidate():
    idx = pd.bdate_range("2000-01-03", periods=3000)
    early = pd.Series(np.nan, index=idx)
    early.iloc[100:] = 0.0
    late = pd.Series(np.nan, index=idx)
    late.iloc[400:] = 1.0
    cfg = shu2024.Config(val_years=2)
    start = shu2024.default_test_start({"HMM": {0: early}, "JM": {0.0: early, 5.0: late}}, cfg)
    assert start == (idx[400] + pd.DateOffset(years=2, months=1)).to_period("M").to_timestamp()
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 3000), index=pd.bdate_range("2000-01-03", periods=3000))
    late_r = late.copy()
    late_r.index = r.index
    late_r.iloc[400:] = np.random.default_rng(1).integers(0, 2, 2600)
    bear, _ = shu2024.cross_validate({0.0: late_r}, r, r * 0, cfg, start)
    assert bear.loc[start:].notna().all()
