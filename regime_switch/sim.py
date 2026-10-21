"""Ground-truth simulators. Every generator returns the observed series together with the truth it was built from."""

from dataclasses import dataclass

import numpy as np

BENCH_MU = (0.0006, -0.0008)
BENCH_SIGMA = (0.007, 0.018)
BENCH_STAY = (0.996, 0.985)


def markov_chain(p_stay, n, rng, s0=None):
    p_stay = np.asarray(p_stay)
    if s0 is None:
        pi0 = (1 - p_stay[1]) / (2 - p_stay[0] - p_stay[1])
        s0 = 0 if rng.random() < pi0 else 1
    u = rng.random(n)
    s = np.empty(n, dtype=np.int64)
    s[0] = s0
    for t in range(1, n):
        s[t] = s[t - 1] if u[t] < p_stay[s[t - 1]] else 1 - s[t - 1]
    return s


@dataclass
class RegimePath:
    log_returns: np.ndarray
    states: np.ndarray  # 0 = favourable (bull / low vol), 1 = unfavourable
    mu: np.ndarray  # per-day true mean of the log-return, shape (n, 2) for drifting paths, (2,) otherwise
    sigma: np.ndarray
    p_stay: np.ndarray


def markov_switching_gaussian(n, seed, mu=BENCH_MU, sigma=BENCH_SIGMA, p_stay=BENCH_STAY):
    rng = np.random.default_rng(seed)
    s = markov_chain(p_stay, n, rng)
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    r = mu[s] + sigma[s] * rng.standard_normal(n)
    return RegimePath(r, s, mu, sigma, np.asarray(p_stay))


def markov_switching_t(n, seed, df=4, mu=BENCH_MU, sigma=BENCH_SIGMA, p_stay=BENCH_STAY):
    rng = np.random.default_rng(seed)
    s = markov_chain(p_stay, n, rng)
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    z = rng.standard_t(df, n) * np.sqrt((df - 2) / df)
    r = mu[s] + sigma[s] * z
    return RegimePath(r, s, mu, sigma, np.asarray(p_stay))


def drifting_hmm(n, seed, period_years=8.0):
    """Gaussian 2-state HMM whose means and vols move along a known sinusoid (one cycle per `period_years`)."""
    rng = np.random.default_rng(seed)
    s = markov_chain(BENCH_STAY, n, rng)
    phase = 2 * np.pi * np.arange(n) / (252 * period_years)
    mu = np.column_stack([0.0006 + 0.0004 * np.sin(phase), -0.0008 + 0.0006 * np.cos(phase)])
    sigma = np.column_stack([0.007 * (1 + 0.3 * np.sin(phase)), 0.018 * (1 + 0.3 * np.cos(phase))])
    idx = np.arange(n)
    r = mu[idx, s] + sigma[idx, s] * rng.standard_normal(n)
    return RegimePath(r, s, mu, sigma, np.asarray(BENCH_STAY))


# Horvath, Issa & Muguruza (2021), Section 3.3

HOURS_PER_YEAR = 252 * 7
GBM_BULL, GBM_BEAR = (0.02, 0.2), (-0.02, 0.3)
MJD_BULL, MJD_BEAR = (0.05, 0.2, 5, 0.02, 0.0125), (-0.05, 0.4, 10, -0.04, 0.1)


def horvath_episodes(n, r, length, rng):
    """r disjoint regime-change intervals [s_i, s_i + l_i] with s_i + l_i + 2 < s_{i+1}, starts uniform at random."""
    block = length + 3
    free = n - r * block
    if free < 0:
        raise ValueError("episodes do not fit in the path")
    starts = np.sort(rng.choice(free + 1, size=r, replace=True)) + np.arange(r) * block
    on = np.zeros(n, dtype=bool)
    for s0 in starts:
        on[s0 : s0 + length + 1] = True
    return on


def _horvath_mask(years, r, seed):
    rng = np.random.default_rng(seed)
    n = HOURS_PER_YEAR * years
    return horvath_episodes(n, r, int(0.5 * HOURS_PER_YEAR), rng), rng


def horvath_gbm(seed, years=20, r=10, bull=GBM_BULL, bear=GBM_BEAR):
    on, rng = _horvath_mask(years, r, seed)
    dt = 1 / HOURS_PER_YEAR
    mu = np.where(on, bear[0], bull[0])
    sig = np.where(on, bear[1], bull[1])
    x = (mu - sig**2 / 2) * dt + sig * np.sqrt(dt) * rng.standard_normal(on.size)  # eq. 34
    return x, on


def horvath_merton(seed, years=20, r=10, bull=MJD_BULL, bear=MJD_BEAR):
    on, rng = _horvath_mask(years, r, seed)
    dt = 1 / HOURS_PER_YEAR
    p = np.where(on[:, None], np.array(bear)[None, :], np.array(bull)[None, :])
    mu, sig, lam, gam, dlt = p.T
    n_jumps = rng.poisson(lam * dt)
    jumps = gam * n_jumps + dlt * np.sqrt(n_jumps) * rng.standard_normal(on.size)
    x = (mu - sig**2 / 2) * dt + sig * np.sqrt(dt) * rng.standard_normal(on.size) + jumps  # eq. 38
    return x, on


def horvath_daily(model, seed, years=30):
    """The paper's hourly gBm / Merton generator aggregated to daily log-returns (7 hours a day), with 1.5 bear
    episodes per 2 years as in the paper's 10 per 20. A day is regime-on if most of its hours are."""
    gen = horvath_gbm if model == "gbm" else horvath_merton
    x, on = gen(seed, years=years, r=years // 2)
    return x.reshape(-1, 7).sum(axis=1), on.reshape(-1, 7).mean(axis=1) > 0.5


def merton_moments(theta, dt=1 / HOURS_PER_YEAR):
    mu, sig, lam, gam, dlt = theta
    return ((mu - sig**2 / 2) + lam * gam) * dt, (sig**2 + lam * (dlt**2 + gam**2)) * dt  # eqs. 39-40


# Daniel & Moskowitz: market with calm/turbulent regimes, WML with the paper's conditional mean and GJR-GARCH variance

DAYS_PER_MONTH = 21
DM_GAMMA0, DM_GAMMA_INT = 0.01973, -0.397  # Table 5 col. 4 (gamma0 in decimal per month)
DM_GJR = {"omega": 1.17e-6, "alpha": 0.111, "gamma": -0.016, "beta": 0.896}  # Appendix D


@dataclass
class MomentumPath:
    mkt: np.ndarray  # daily simple market excess returns
    wml: np.ndarray  # daily simple WML returns
    month: np.ndarray  # month index of each day
    mu_month: np.ndarray  # true conditional WML mean for each month (decimal per month)
    var_month: np.ndarray  # true conditional variance of the monthly WML return, as of the month start
    bear: np.ndarray  # true I_B,t-1 for each month
    mkt_var: np.ndarray  # true sigma^2_m,t-1 (annualised) used in the mean
    regime: np.ndarray  # true daily market regime, 0 = calm, 1 = turbulent


def momentum_panic(seed, years=80, gamma0=DM_GAMMA0, gamma_int=DM_GAMMA_INT, gjr=DM_GJR):
    rng = np.random.default_rng(seed)
    n_months = 12 * years
    n = n_months * DAYS_PER_MONTH
    regime = markov_chain((0.998, 0.99), n, rng)
    mkt = np.where(regime == 0, 0.0005, -0.0008) + np.where(regime == 0, 0.008, 0.022) * rng.standard_normal(n)
    month = np.repeat(np.arange(n_months), DAYS_PER_MONTH)
    lookback_bear, lookback_var = 24 * DAYS_PER_MONTH, 126
    mu_month = np.full(n_months, np.nan)
    var_month = np.full(n_months, np.nan)
    bear = np.full(n_months, np.nan)
    mkt_var = np.full(n_months, np.nan)
    om, al, ga, be = gjr["omega"], gjr["alpha"], gjr["gamma"], gjr["beta"]
    persist = al + ga / 2 + be
    long_var = om / (1 - persist)
    wml = np.empty(n)
    h = long_var
    eps_prev = 0.0
    for m in range(n_months):
        d0 = m * DAYS_PER_MONTH
        if d0 >= lookback_bear:
            bear[m] = float(np.prod(1 + mkt[d0 - lookback_bear : d0]) - 1 < 0)
            mkt_var[m] = mkt[d0 - lookback_var : d0].var(ddof=1) * 252
            mu_month[m] = gamma0 + gamma_int * bear[m] * mkt_var[m]
        mu_d = (mu_month[m] if np.isfinite(mu_month[m]) else gamma0) / DAYS_PER_MONTH
        for d in range(d0, d0 + DAYS_PER_MONTH):
            h = om + be * h + (al + ga * (eps_prev < 0)) * eps_prev**2  # eq. 7
            if d == d0:
                var_month[m] = sum(long_var + persist**j * (h - long_var) for j in range(DAYS_PER_MONTH))
            eps_prev = np.sqrt(h) * rng.standard_normal()
            wml[d] = mu_d + eps_prev
    return MomentumPath(mkt, wml, month, mu_month, var_month, bear, mkt_var, regime)
