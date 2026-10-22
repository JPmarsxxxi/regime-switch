"""Nystrup, Madsen & Lindström (2018), "Dynamic portfolio optimization across hidden market regimes".

Adaptive 2-state Gaussian HMM (Sec. 2.1, eq. 7) -> mixture forecasts of mean and variance K steps ahead
(Sec. 2.2, eqs. 8-13) -> truncated MPC over stock and zero-interest cash (Sec. 3, eq. 25, Algorithm 1).
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from numba import njit

from . import hmm
from .metrics import nystrup_metrics

P = 6  # theta = (mu1, mu2, log var1, log var2, logit gamma11, logit gamma22); see DEVIATIONS.md#N2


@njit(cache=True)
def unpack(theta):
    mu = theta[0:2].copy()
    var = np.exp(theta[2:4])
    g = 1.0 / (1.0 + np.exp(-theta[4:6]))
    return mu, var, g


def pack(mu, var, g11, g22):
    g = np.array([g11, g22])
    return np.concatenate([mu, np.log(var), np.log(g / (1 - g))])


@njit(cache=True)
def score_pass(x, theta, f):
    """One forward pass of the exact derivative recursion (Lystig & Hughes 2002 in forward mode) at fixed theta.
    Returns the weighted score sum_n f^(t-n) d/dtheta log Pr(y_n | y_1..n-1), the last observation's score, the
    per-observation scores and the conditional log-likelihood terms."""
    mu, var, g = unpack(theta)
    T = x.size
    G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    dG = np.zeros((P, 2, 2))
    d0, d1 = g[0] * (1 - g[0]), g[1] * (1 - g[1])
    dG[4, 0, 0], dG[4, 0, 1] = d0, -d0
    dG[5, 1, 0], dG[5, 1, 1] = -d1, d1
    D = 2 - g[0] - g[1]
    pi = np.array([(1 - g[1]) / D, (1 - g[0]) / D])
    dpi = np.zeros((P, 2))
    dpi[4, 0] = (1 - g[1]) / D**2 * d0
    dpi[5, 0] = -(1 - g[0]) / D**2 * d1
    dpi[4, 1], dpi[5, 1] = -dpi[4, 0], -dpi[5, 0]
    a = pi.copy()
    da = dpi.copy()
    wsum = np.zeros(P)
    scores = np.empty((T, P))
    lls = np.empty(T)
    lp = np.empty(2)
    dlp = np.zeros((P, 2))
    for n in range(T):
        if n == 0:
            pr, dpr = pi.copy(), dpi.copy()
        else:
            pr = np.empty(2)
            dpr = np.empty((P, 2))
            for j in range(2):
                pr[j] = a[0] * G[0, j] + a[1] * G[1, j]
                for q in range(P):
                    dpr[q, j] = da[q, 0] * G[0, j] + da[q, 1] * G[1, j] + a[0] * dG[q, 0, j] + a[1] * dG[q, 1, j]
        for k in range(2):
            z = x[n] - mu[k]
            lp[k] = -0.5 * (z * z / var[k] + np.log(var[k]) + np.log(2 * np.pi))
        m = max(lp[0], lp[1])
        dlp[:] = 0.0
        for k in range(2):
            z = x[n] - mu[k]
            dlp[k, k] = z / var[k]
            dlp[2 + k, k] = 0.5 * (z * z / var[k] - 1.0)
        u = np.empty(2)
        du = np.empty((P, 2))
        for k in range(2):
            pk = np.exp(lp[k] - m)
            u[k] = pr[k] * pk
            for q in range(P):
                du[q, k] = dpr[q, k] * pk + pr[k] * pk * dlp[q, k]
        c = u[0] + u[1]
        lls[n] = np.log(c) + m
        for q in range(P):
            dc = du[q, 0] + du[q, 1]
            scores[n, q] = dc / c
            for k in range(2):
                da[q, k] = (du[q, k] - u[k] / c * dc) / c
        a = u / c
        wsum = f * wsum + scores[n]
    return wsum, scores[T - 1], scores, lls


LO = np.array([-0.01, -0.01, np.log(1e-8), np.log(1e-8), 0.0, 0.0])  # see DEVIATIONS.md#N4
HI = np.array([0.01, 0.01, np.log(1e-2), np.log(1e-2), np.log(0.9999 / 0.0001), np.log(0.9999 / 0.0001)])


@njit(cache=True)
def _adapt(x, theta0, R0, n0, f, A, lo, hi):
    T = x.size
    thetas = np.full((T, P), np.nan)
    theta = theta0.copy()
    R = R0.copy()
    thetas[n0 - 1] = theta
    for t in range(n0, T):
        g, s, _, _ = score_pass(x[: t + 1], theta, f)
        R = f * R + np.outer(s, s)  # see DEVIATIONS.md#N4
        theta = np.minimum(np.maximum(theta + A * np.linalg.solve(R, g), lo), hi)  # eq. 7, projected
        thetas[t] = theta
    return thetas


def adaptive_estimates(x, n_eff=260, n_init_obs=260):
    """Eq. 7 with exponential forgetting, f = 1 - 1/N_eff and A = 1/N_eff, run from day `n_init_obs`. The Fisher
    information is accumulated with the same forgetting, R_t = f R_{t-1} + s_t s_t' (the recursive form the paper's
    matrix-inversion-lemma remark implies). The first `n_init_obs` observations initialise it: a neutral persistent
    start (both means at the sample mean, variances at half and twice the sample variance, P(stay) = 0.98) refined by
    the recursion over those days, with R started at N_eff times the mean outer product of their scores.
    Returns (T, 6) theta_t, NaN before n_init_obs - 1. See DEVIATIONS.md#N3."""
    x = np.ascontiguousarray(x, dtype=float)
    f = 1 - 1 / n_eff
    x0 = x[:n_init_obs]
    theta0 = pack(np.full(2, x0.mean()), x0.var() * np.array([0.5, 2.0]), 0.98, 0.98)
    _, _, sc, _ = score_pass(x0, theta0, f)
    R0 = n_eff * sc.T @ sc / n_init_obs
    warm = _adapt(x0, theta0, R0, 20, f, 1 / n_eff, LO, HI)[-1]
    _, _, sc, _ = score_pass(x0, warm, f)
    return _adapt(x, warm, n_eff * sc.T @ sc / n_init_obs, n_init_obs, f, 1 / n_eff, LO, HI)


def filtered_probs(x, thetas):
    """Eq. 8 with time-varying parameters: alpha_t ∝ alpha_{t-1} Gamma_t P_t(y_t), starting where thetas begins."""
    start = int(np.argmax(~np.isnan(thetas[:, 0])))
    out = np.full((x.size, 2), np.nan)
    mu, var, g = unpack(thetas[start])
    G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    a, _ = hmm.forward(np.ascontiguousarray(x[: start + 1]), mu, var, G, np.array([0.5, 0.5]))
    alpha = a[-1]
    out[start] = alpha
    for t in range(start + 1, x.size):
        mu, var, g = unpack(thetas[t])
        G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
        lp = -0.5 * (x[t] - mu) ** 2 / var - 0.5 * np.log(var)
        u = (alpha @ G) * np.exp(lp - lp.max())
        alpha = u / u.sum()
        out[t] = alpha
    return out


def forecast_moments(alpha, theta, K):
    """Eqs. 9-13: mean and variance of the simple return 1..K days ahead."""
    mu, var, g = unpack(theta)
    G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    m_i = np.exp(mu + var / 2) - 1  # eq. 12
    v_i = (np.exp(var) - 1) * np.exp(2 * mu + var)  # eq. 13
    means, variances = np.empty(K), np.empty(K)
    a = np.asarray(alpha, dtype=float)
    for k in range(K):
        a = a @ G  # eq. 9
        means[k] = a @ m_i  # eq. 10
        variances[k] = a @ (m_i**2 + v_i) - means[k] ** 2  # eq. 11
    return means, variances


class MPC:
    """Eq. 25 with stock + cash, holdings scaled so V_t = 1. Risk penalty eq. 22, trading penalty eq. 23, costs eq. 16
    on the stock leg only, long-only post-trade holdings."""

    def __init__(self, K=100, gamma=0.0, rho=0.0, kappa=0.001):
        self.K = K
        self.r = cp.Parameter(K)
        self.sd = cp.Parameter(K, nonneg=True)
        self.h0 = cp.Parameter(2, nonneg=True)
        h = cp.Variable((K + 1, 2))
        u = cp.Variable((K, 2))
        hp = h[:-1] + u
        cons = [h[0] == self.h0, hp >= 0,
                cp.sum(u, axis=1) + kappa * cp.abs(u[:, 0]) <= 0,  # eq. 16
                h[1:, 0] == cp.multiply(1 + self.r, hp[:, 0]), h[1:, 1] == hp[:, 1]]
        cost = 0
        if gamma > 0:
            cost += gamma * sum(cp.quad_over_lin(self.sd[k] * hp[k, 0], cp.sum(hp[k])) for k in range(K))  # eq. 22
        if rho > 0:
            cost += rho * cp.sum(cp.abs(u[:, 0]))  # eq. 23
        self.u = u
        self.prob = cp.Problem(cp.Maximize(cp.sum(h[K]) - cost), cons)

    def first_trade(self, h, means, variances):
        self.h0.value = np.maximum(h, 0)
        self.r.value = means[: self.K]
        self.sd.value = np.sqrt(np.maximum(variances[: self.K], 0))
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except cp.SolverError:
            return None
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        return float(self.u.value[0, 0])


@dataclass
class Result:
    returns: pd.Series
    weights: pd.Series  # stock weight held over each day
    turnover: pd.Series  # |u_stock| / V per day
    failures: int


def simulate_mpc(simple_returns, means, variances, start, gamma=0.0, rho=0.0, kappa=0.001, K=100, delay=0):
    """Algorithm 1 from day `start`: decide u_t after observing day t, hold to t+1. With delay = 1 the decision of
    day t is executed at the end of day t+1 as a fraction of the holding it trades out of (Sec. 4.5)."""
    r = np.asarray(simple_returns, dtype=float)
    mpc = MPC(K, gamma, rho, kappa)
    h = np.array([0.0, 1.0])
    rets, wts, turn = [], [], []
    pending = None
    fails = 0
    for t in range(start, r.size - 1):
        V = h.sum()
        hn = h / V
        trade = 0.0
        if delay == 0:
            u = mpc.first_trade(hn, means[t], variances[t])
            fails += u is None
            trade = u or 0.0
        else:
            if pending is not None:
                frac, side = pending
                trade = -frac * hn[0] if side == "sell" else frac * hn[1] / (1 + kappa)
            u = mpc.first_trade(hn, means[t], variances[t])  # decided on today's pre-trade holdings, DEVIATIONS.md#N6
            fails += u is None
            u = u or 0.0
            pending = None
            if u < -1e-9 and hn[0] > 0:
                pending = (min(-u / hn[0], 1.0), "sell")
            elif u > 1e-9 and hn[1] > 0:
                pending = (min(u * (1 + kappa) / hn[1], 1.0), "buy")
        trade = float(np.clip(trade, -hn[0], hn[1] / (1 + kappa)))
        hp = np.array([hn[0] + trade, hn[1] - trade - kappa * abs(trade)])  # eq. 15 with equality
        h_next = np.array([hp[0] * (1 + r[t + 1]), hp[1]])
        rets.append(h_next.sum() - 1)  # V_t = 1 before trading, so this is net of costs
        wts.append(hp[0] / hp.sum())
        turn.append(abs(trade))
        h = h_next
    return np.array(rets), np.array(wts), np.array(turn), fails


@dataclass
class Config:
    n_eff: int = 260
    n_init_obs: int = 260
    K: int = 100
    kappa: float = 0.001
    seed: int = 0


def run(log_returns: pd.Series, cfg: Config | None = None, test_start=None, configs=((0.0, 0.0), (2.0, 0.0), (0.0, 0.02)),
        delays=(0,)):
    """Returns (thetas DataFrame, dict[(gamma, rho, delay)] -> Result, metrics DataFrame incl. buy-and-hold)."""
    cfg = cfg or Config()
    x = log_returns.to_numpy()
    thetas = adaptive_estimates(x, cfg.n_eff, cfg.n_init_obs)
    alpha = filtered_probs(x, thetas)
    start = cfg.n_init_obs - 1 if test_start is None else int(log_returns.index.searchsorted(test_start)) - 1
    means = np.full((x.size, cfg.K), np.nan)
    variances = np.full((x.size, cfg.K), np.nan)
    for t in range(start, x.size):
        means[t], variances[t] = forecast_moments(alpha[t], thetas[t], cfg.K)
    simple = np.expm1(x)
    idx = log_returns.index[start + 1 :]
    out, met = {}, {}
    for gamma, rho in configs:
        for d in delays:
            rets, w, turn, fails = simulate_mpc(simple, means, variances, start, gamma, rho, cfg.kappa, cfg.K, d)
            key = (gamma, rho, d)
            out[key] = Result(pd.Series(rets, idx), pd.Series(w, idx), pd.Series(turn, idx), fails)
            met[f"MPC(g={gamma:g}, rho={rho:g}, delay={d})"] = nystrup_metrics(rets, turn)
    met["Buy and hold"] = nystrup_metrics(simple[start + 1 :], np.zeros(idx.size))
    cols = ["mu1", "mu2", "var1", "var2", "gamma11", "gamma22"]
    th = pd.DataFrame(thetas, index=log_returns.index, columns=cols)
    th[["var1", "var2"]] = np.exp(th[["var1", "var2"]])
    th[["gamma11", "gamma22"]] = 1 / (1 + np.exp(-th[["gamma11", "gamma22"]]))
    return th, out, pd.DataFrame(met), means, variances, alpha
