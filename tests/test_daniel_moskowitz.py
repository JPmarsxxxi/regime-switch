import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from regime_switch import daniel_moskowitz as dm


def test_gjr_matches_arch():
    rng = np.random.default_rng(0)
    n, om, al, ga, be, mu = 6000, 1e-6, 0.08, 0.05, 0.88, 5e-4
    x = np.empty(n)
    h, e = om / (1 - al - ga / 2 - be), 0.0
    for t in range(n):
        h = om + be * h + (al + ga * (e < 0)) * e**2
        e = np.sqrt(h) * rng.standard_normal()
        x[t] = mu + e
    ours = dm.gjr_fit(x)
    ref = arch_model(x * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, rescale=False).fit(disp="off")
    p, se = ref.params, ref.std_err
    pairs = [("mu", "mu", 100), ("omega", "omega", 1e4), ("alpha", "alpha[1]", 1), ("gamma", "gamma[1]", 1),
             ("beta", "beta[1]", 1)]
    for ours_k, ref_k, scale in pairs:
        assert abs(ours[ours_k] * scale - p[ref_k]) <= 2 * se[ref_k], ours_k
    # arch starts the variance recursion from a backcast, we start it from the sample variance; compare both parameter
    # sets under our likelihood, where ours must be at least as good
    def loglik(mu_, om_, al_, ga_, be_):
        h = dm.gjr_variance(x, mu_, om_, al_, ga_, be_)[:-1]
        return -0.5 * np.sum(np.log(2 * np.pi * h) + (x - mu_) ** 2 / h)

    at_ref = loglik(p["mu"] / 100, p["omega"] / 1e4, p["alpha[1]"], p["gamma[1]"], p["beta[1]"])
    assert ours["loglik"] == pytest.approx(loglik(*(ours[k] for k in ("mu", "omega", "alpha", "gamma", "beta"))))
    assert ours["loglik"] >= at_ref - 1e-6
    assert ours["loglik"] == pytest.approx(ref.loglikelihood + n * np.log(100), abs=2.0)  # arch fits 100 x returns


def test_bear_indicator_and_market_variance_by_hand():
    months = pd.period_range("2000-01", periods=30, freq="M")
    mkt_m = pd.Series(0.01, index=months)
    mkt_m.iloc[3] = -0.5  # drags the 24-month cumulative return negative while it is in the window
    ib = dm.bear_indicator(mkt_m)
    assert ib.iloc[:24].isna().all()
    assert ib.iloc[24] == 1.0  # window months 0..23 contains month 3
    assert ib.iloc[28] == 0.0  # window months 4..27 does not
    rng = np.random.default_rng(1)
    month = np.repeat(np.arange(10), 21)
    daily = pd.DataFrame({"mkt": rng.standard_normal(210) * 0.01, "month": month})
    v = dm.market_variance(daily)
    assert np.isnan(v.iloc[5])
    assert v.iloc[6] == pytest.approx(daily.mkt.iloc[0:126].var(ddof=1) * 252)
    assert v.iloc[9] == pytest.approx(daily.mkt.iloc[63:189].var(ddof=1) * 252)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    res, _ = dm.ols(1.0 + 2.0 * x + 0.1 * rng.standard_normal(5000), [x], ["x"])
    assert res.loc["const", "coef"] == pytest.approx(1.0, abs=0.01)
    assert res.loc["x", "coef"] == pytest.approx(2.0, abs=0.01)
