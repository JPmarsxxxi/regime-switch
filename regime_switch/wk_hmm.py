"""WK-means (Horvath et al. 2021) with an HMM head: an extension, not a paper replication (SPEC: Extension).

Both variants refit on Shu et al.'s calendar (end of June / December, trailing `train` days) and return, for every day,
P(regime-on | data to that day) from a forward filter.
  A: emissions = exponential likelihood of the W1 distance between the last 35-day return distribution and each centroid.
  B: emissions = weighted Gaussian KDEs of the training returns, initialised from WK-means and refined by EM.
"""

import numpy as np
import pandas as pd
from numba import njit

from . import horvath2021 as hv
from .shu2024 import refit_dates

H1, H2 = 35, 28


@njit(cache=True)
def forward_filter(logE, G, delta):
    """Filtered probabilities from log emission values (T, 2)."""
    T = logE.shape[0]
    a = np.empty((T, 2))
    prev = delta.copy()
    for t in range(T):
        m = max(logE[t, 0], logE[t, 1])
        u = np.empty(2)
        for j in range(2):
            pr = prev[j] if t == 0 else a[t - 1, 0] * G[0, j] + a[t - 1, 1] * G[1, j]
            u[j] = pr * np.exp(logE[t, j] - m)
        a[t] = u / (u[0] + u[1])
    return a


@njit(cache=True)
def forward_backward(logE, G, delta):
    """State posteriors and expected transition counts for emission values given in logs."""
    T = logE.shape[0]
    p = np.empty((T, 2))
    for t in range(T):
        m = max(logE[t, 0], logE[t, 1])
        p[t, 0], p[t, 1] = np.exp(logE[t, 0] - m), np.exp(logE[t, 1] - m)
    a = np.empty((T, 2))
    c = np.empty(T)
    u0, u1 = delta[0] * p[0, 0], delta[1] * p[0, 1]
    c[0] = u0 + u1
    a[0, 0], a[0, 1] = u0 / c[0], u1 / c[0]
    for t in range(1, T):
        u0 = (a[t - 1, 0] * G[0, 0] + a[t - 1, 1] * G[1, 0]) * p[t, 0]
        u1 = (a[t - 1, 0] * G[0, 1] + a[t - 1, 1] * G[1, 1]) * p[t, 1]
        c[t] = u0 + u1
        a[t, 0], a[t, 1] = u0 / c[t], u1 / c[t]
    b = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        for i in range(2):
            b[t, i] = (G[i, 0] * p[t + 1, 0] * b[t + 1, 0] + G[i, 1] * p[t + 1, 1] * b[t + 1, 1]) / c[t + 1]
    xi = np.zeros((2, 2))
    for t in range(T - 1):
        for i in range(2):
            for j in range(2):
                xi[i, j] += a[t, i] * G[i, j] * p[t + 1, j] * b[t + 1, j] / c[t + 1]
    return a * b, xi


def stationary(G):
    return np.array([1 - G[1, 1], 1 - G[0, 0]]) / (2 - G[0, 0] - G[1, 1])


def wk_daily_labels(x, seed):
    """WK-means on the window, then per-day regime-on labels by majority vote of the segments covering each day
    (days after the last segment take the last covered label). Returns (labels, segments, segment labels, centroids,
    regime-on cluster)."""
    segs, starts = hv.lift(x, H1, H2)
    lab, cent = hv.wk_means(segs, seed=seed)
    on = hv.regime_on_cluster(segs, lab)
    y_off, y_on = hv.label_counts(x.size, starts, H1, lab == on)
    daily = pd.Series(np.where(y_off + y_on > 0, (y_on > y_off).astype(float), np.nan)).ffill().bfill()
    return daily.to_numpy().astype(int), segs, lab, cent, on


def transition_counts(labels):
    """Daily transition matrix from a label sequence with +1 smoothing."""
    n = np.ones((2, 2))
    np.add.at(n, (labels[:-1], labels[1:]), 1)
    return n / n.sum(axis=1, keepdims=True)


def fit_a(x, seed):
    labels, segs, lab, cent, on = wk_daily_labels(x, seed)
    S = np.sort(segs, axis=1)
    order = [1 - on, on]  # state 0 = regime-off, state 1 = regime-on
    tau = np.array([hv.wasserstein(S[lab == k], cent[k], 1).mean() for k in order])
    return {"cent": cent[order], "tau": tau, "G": transition_counts(labels)}


def log_emissions_a(x, params):
    """log e_k for every day with a full 35-day window behind it (earlier rows are 0 = uninformative)."""
    out = np.zeros((x.size, 2))
    if x.size < H1:
        return out
    S = np.sort(np.lib.stride_tricks.sliding_window_view(x, H1), axis=1)
    d = hv.wasserstein(S[:, None, :], params["cent"][None, :, :], 1)
    out[H1 - 1 :] = -d / params["tau"] - np.log(params["tau"])
    return out


def kde_logpdf(points, centers, weights, h, chunk=2048):
    """log of sum_i w_i N(points; centers_i, h^2) / sum_i w_i."""
    w = weights / weights.sum()
    out = np.empty(points.size)
    for i in range(0, points.size, chunk):
        z = (points[i : i + chunk, None] - centers[None, :]) / h
        out[i : i + chunk] = np.log((np.exp(-0.5 * z * z) * w[None, :]).sum(axis=1) + 1e-300) - np.log(h * np.sqrt(2 * np.pi))
    return out


def scott_bandwidth(x, w):
    w = w / w.sum()
    mu = (w * x).sum()
    sd = np.sqrt((w * (x - mu) ** 2).sum())
    n_eff = 1 / (w**2).sum()
    return sd * n_eff ** (-1 / 5)


def _loo_log_emissions(x, gamma):
    """Leave-one-out weighted KDE log densities at the training points, so a point cannot explain itself in EM."""
    out = np.empty((x.size, 2))
    for k in range(2):
        w = gamma[:, k]
        h = scott_bandwidth(x, w)
        z = (x[:, None] - x[None, :]) / h
        K = np.exp(-0.5 * z * z) * w[None, :]
        np.fill_diagonal(K, 0.0)
        out[:, k] = np.log(K.sum(axis=1) / (w.sum() - w) + 1e-300) - np.log(h * np.sqrt(2 * np.pi))
    return out


def fit_b(x, seed, n_iter=20):
    labels, *_ = wk_daily_labels(x, seed)
    gamma = np.column_stack([1.0 - labels, labels.astype(float)])
    G = transition_counts(labels)
    for _ in range(n_iter):
        gamma, xi = forward_backward(_loo_log_emissions(x, gamma), G, stationary(G))
        G = xi / xi.sum(axis=1, keepdims=True)
    var = [np.average((x - np.average(x, weights=gamma[:, k])) ** 2, weights=gamma[:, k]) for k in range(2)]
    on = int(np.argmax(var))
    order = [1 - on, on]
    gamma = gamma[:, order]
    G = G[np.ix_(order, order)]
    h = [scott_bandwidth(x, gamma[:, k]) for k in range(2)]
    return {"x": x.copy(), "w": gamma, "h": np.array(h), "G": G}


def log_emissions_b(x, params):
    return np.column_stack([kde_logpdf(x, params["x"], params["w"][:, k], params["h"][k]) for k in range(2)])


def online(log_returns: pd.Series, variant, train=3000, refit_months=6, seed=0):
    """P(regime-on | data to t) for every day after the first fit (NaN before). Each refit's parameters are used from
    its fit day until the next refit; the filter for that block runs over the trailing `train` days so it starts warm."""
    fit, emis = (fit_a, log_emissions_a) if variant == "A" else (fit_b, log_emissions_b)
    x = log_returns.to_numpy()
    prob = pd.Series(np.nan, index=log_returns.index)
    fits = refit_dates(log_returns.index, log_returns.index[train - 1], refit_months)
    for i, d in enumerate(fits):
        end = log_returns.index.get_loc(d) + 1
        stop = log_returns.index.get_loc(fits[i + 1]) + 1 if i + 1 < len(fits) else x.size
        params = fit(x[end - train : end], seed + i)
        a = forward_filter(emis(x[end - train : stop], params), params["G"], stationary(params["G"]))
        prob.iloc[end - 1 : stop] = a[train - 1 :, 1]
    return prob
