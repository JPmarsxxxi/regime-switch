"""Daniel & Moskowitz, "Momentum Crashes" (2013 working paper; JFE 2016).

Bear indicator and ex-ante market variance (Sec. 3.5) -> Table 5 regressions -> mean forecast; GJR-GARCH on daily WML
plus 126-day realised vol -> vol forecast (Appendix D); dynamic weight w = mu / (2 lambda sigma^2) (eq. 5) scaled to
19% annual vol (Sec. 4.1); constant-volatility comparison (Barroso & Santa-Clara); spanning tests (Table 7).

All functions take daily data with a `month` label column so the same code runs on Ken French data and on simulations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DAYS = 252
TARGET_VOL = 0.19


def ols(y, X, names):
    """OLS with classical standard errors. Returns DataFrame [coef, t]."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, dtype=float) for c in X])
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return pd.DataFrame({"coef": beta, "t": beta / se}, index=["const"] + names), X @ beta


def bear_indicator(monthly_mkt):
    """I_B,t-1 = 1 if the cumulative market return over months t-24..t-1 is negative."""
    cum = (1 + monthly_mkt).rolling(24).apply(np.prod, raw=True) - 1
    return (cum.shift(1) < 0).astype(float).where(cum.shift(1).notna())


def monthly_from_daily(daily, cols=("wml", "mkt")):
    """Compound daily simple returns into monthly returns, one row per `month` label."""
    return daily.groupby("month")[list(cols)].agg(lambda v: np.prod(1 + v) - 1)


def pre_month_stat(daily, col, n, fn):
    """fn over the `n` daily observations before the first day of each month (NaN until n are available)."""
    first = daily.groupby("month").apply(lambda g: daily.index.get_loc(g.index[0]))
    vals = daily[col].to_numpy()
    return pd.Series([fn(vals[i - n : i]) if i >= n else np.nan for i in first], index=first.index)


def market_variance(daily):
    """sigma^2_m,t-1: variance of the 126 daily market returns before month t, annualised. See DEVIATIONS.md#D3."""
    return pre_month_stat(daily, "mkt", 126, lambda v: v.var(ddof=1) * DAYS)


def table5(wml_m, bear, mvar):
    """The five regressions of Table 5 (monthly WML on I_B, sigma^2_m and their interaction)."""
    df = pd.DataFrame({"wml": wml_m, "IB": bear, "var": mvar}).dropna()
    inter = df.IB * df["var"]
    specs = {"(1)": [("IB", df.IB)], "(2)": [("var", df["var"])], "(3)": [("IB", df.IB), ("var", df["var"])],
             "(4)": [("int", inter)], "(5)": [("IB", df.IB), ("var", df["var"]), ("int", inter)]}
    out, fitted = {}, {}
    for k, cols in specs.items():
        res, fit = ols(df.wml, [c for _, c in cols], [n for n, _ in cols])
        out[k] = res
        fitted[k] = pd.Series(fit, index=df.index)
    return out, fitted


def gjr_variance(x, mu, omega, alpha, gamma, beta):
    """Eq. 7: sigma^2_t given information to t-1; sigma^2_0 = sample variance of eps."""
    eps = x - mu
    h = np.empty(x.size + 1)
    h[0] = eps.var()
    for t in range(x.size):
        h[t + 1] = omega + beta * h[t] + (alpha + gamma * (eps[t] < 0)) * eps[t] ** 2
    return h  # h[t] is the variance of x[t]; h[-1] is the one-step forecast after the sample


def gjr_fit(x):
    """Gaussian ML for eqs. 6-7. Returns dict(mu, omega, alpha, gamma, beta, loglik)."""
    x = np.asarray(x, dtype=float)
    scale = x.std()
    z = x / scale

    def nll(p):
        mu, omega, alpha, gamma, beta = p
        h = gjr_variance(z, mu, omega, alpha, gamma, beta)[:-1]
        if np.any(h <= 0):
            return 1e10
        return 0.5 * np.mean(np.log(2 * np.pi * h) + (z - mu) ** 2 / h)  # mean keeps SLSQP well scaled

    cons = [{"type": "ineq", "fun": lambda p: 1 - p[2] - p[3] / 2 - p[4] - 1e-6},
            {"type": "ineq", "fun": lambda p: p[2] + p[3]}]
    best = None
    for a0, b0 in ((0.1, 0.85), (0.05, 0.9), (0.15, 0.8)):
        res = minimize(nll, [z.mean(), 0.02, a0, 0.0, b0], method="SLSQP", constraints=cons,
                       bounds=[(-1, 1), (1e-8, 1), (0, 1), (-1, 1), (0, 1)], options={"maxiter": 1000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    if not best.success:
        raise RuntimeError(f"GJR-GARCH did not converge: {best.message}")
    mu, omega, alpha, gamma, beta = best.x
    loglik = -best.fun * x.size - x.size * np.log(scale)
    return {"mu": mu * scale, "omega": omega * scale**2, "alpha": alpha, "gamma": gamma, "beta": beta,
            "loglik": loglik}


def vol_forecast(daily, gjr):
    """Appendix D: regress next-month realised daily vol on the GJR-GARCH vol at the month start and the trailing
    126-day realised vol; return the regression, the fitted forecast (daily vol) and both regressors."""
    # sigma_GARCH,t is the one-step-ahead conditional vol for the month's first day; see DEVIATIONS.md#D5
    h = gjr_variance(daily.wml.to_numpy(), gjr["mu"], gjr["omega"], gjr["alpha"], gjr["gamma"], gjr["beta"])[:-1]
    d = daily.assign(h=h)
    first = d.groupby("month").head(1)
    s_garch = pd.Series(np.sqrt(first.h.to_numpy()), index=first.month.to_numpy())
    s126 = pre_month_stat(daily, "wml", 126, lambda v: v.std(ddof=1))
    s22 = daily.groupby("month").wml.std(ddof=1)
    df = pd.DataFrame({"s22": s22, "garch": s_garch, "s126": s126}).dropna()
    res, fit = ols(df.s22, [df.garch, df.s126], ["sigma_GARCH", "sigma_126"])
    return res, pd.Series(fit, index=df.index), df


def scale_to_vol(w, daily_wml, month, target=TARGET_VOL):
    """Multiply monthly weights so the daily strategy has annualised vol `target` in-sample (fixes lambda)."""
    r = w.reindex(month).to_numpy() * daily_wml
    return w * target / (np.nanstd(r, ddof=1) * np.sqrt(DAYS))


@dataclass
class Result:
    table5: dict
    gjr: dict
    vol_reg: pd.DataFrame
    weights: pd.DataFrame  # monthly weights: dyn, cvol, static
    daily: pd.DataFrame  # daily returns of each strategy (zero investment, excess of rf)
    sharpe: pd.Series


def run(daily: pd.DataFrame, monthly: pd.DataFrame, mean_col="(4)", target=TARGET_VOL):
    """`daily`: columns wml, mkt (total market return), month. `monthly`: columns wml, mkt, indexed by month.
    Every estimate is full-sample, as in the paper."""
    bear = bear_indicator(monthly.mkt)
    mvar = market_variance(daily)
    t5, fitted = table5(monthly.wml, bear, mvar)
    mu = fitted[mean_col]  # see DEVIATIONS.md#D2
    gjr = gjr_fit(daily.wml.to_numpy())
    vreg, sig, regs = vol_forecast(daily, gjr)
    var_m = (sig**2) * daily.groupby("month").size().reindex(sig.index)
    common = mu.index.intersection(var_m.index)
    w_dyn = (mu[common] / var_m[common]).rename("dyn")
    w_cvol = (1 / regs.s126[common]).rename("cvol")
    w_static = pd.Series(1.0, index=common, name="static")
    month = daily.month.to_numpy()
    x = daily.wml.to_numpy()
    keep = np.isin(month, common)
    W = pd.DataFrame({k: scale_to_vol(w, x[keep], month[keep], target) for k, w in
                      (("dyn", w_dyn), ("cvol", w_cvol), ("static", w_static))})
    rets = pd.DataFrame({k: W[k].reindex(month[keep]).to_numpy() * x[keep] for k in W}, index=daily.index[keep])
    sr = rets.mean() / rets.std(ddof=1) * np.sqrt(DAYS)
    return Result(t5, gjr, vreg, W, rets, sr)


def spanning(y, X, names):
    """Table 7: daily OLS of a strategy on factors; alpha annualised in percent (x 252 x 100)."""
    res, _ = ols(y, X, names)
    res.loc["const", "coef"] *= DAYS * 100
    return res
