"""Univariate Gaussian HMM: scaled forward filter, Baum-Welch, Viterbi. numba ports of the textbook recursions
(Zucchini, MacDonald & Langrock 2016), cross-checked against hmmlearn in tests."""

import numpy as np
from numba import njit
from sklearn.cluster import kmeans_plusplus

LOG2PI = np.log(2 * np.pi)


@njit(cache=True)
def _logdens(x, mu, var):
    K = mu.size
    T = x.size
    lp = np.empty((T, K))
    for k in range(K):
        for t in range(T):
            lp[t, k] = -0.5 * ((x[t] - mu[k]) ** 2 / var[k] + np.log(var[k]) + LOG2PI)
    return lp


@njit(cache=True)
def _dens(x, mu, var):
    """Densities rescaled per row by exp(-m_t) so a crash day cannot underflow every state; m is added back to the
    log-likelihood and cancels everywhere else."""
    lp = _logdens(x, mu, var)
    m = np.empty(x.size)
    for t in range(x.size):
        m[t] = lp[t].max()
    return np.exp(lp - m.reshape(-1, 1)), m


@njit(cache=True)
def forward(x, mu, var, G, delta):
    """Filtered probabilities Pr(S_t | y_1..t) (eq. 8 of Nystrup et al. 2018) and the log-likelihood."""
    p, m = _dens(x, mu, var)
    T, K = p.shape
    a = np.empty((T, K))
    u = delta * p[0]
    c = u.sum()
    a[0] = u / c
    ll = np.log(c) + m[0]
    for t in range(1, T):
        u = (a[t - 1] @ G) * p[t]
        c = u.sum()
        a[t] = u / c
        ll += np.log(c) + m[t]
    return a, ll


@njit(cache=True)
def _em(x, mu, var, G, delta, max_iter, tol, var_floor):
    T = x.size
    K = mu.size
    ll_old = -np.inf
    ll = -np.inf
    a = np.empty((T, K))
    b = np.empty((T, K))
    c = np.empty(T)
    for it in range(max_iter):
        p, m = _dens(x, mu, var)
        c[0] = 0.0
        for j in range(K):
            a[0, j] = delta[j] * p[0, j]
            c[0] += a[0, j]
        for j in range(K):
            a[0, j] /= c[0]
        for t in range(1, T):
            c[t] = 0.0
            for j in range(K):
                s = 0.0
                for i in range(K):
                    s += a[t - 1, i] * G[i, j]
                a[t, j] = s * p[t, j]
                c[t] += a[t, j]
            for j in range(K):
                a[t, j] /= c[t]
        ll = np.log(c).sum() + m.sum()
        for j in range(K):
            b[T - 1, j] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                s = 0.0
                for j in range(K):
                    s += G[i, j] * p[t + 1, j] * b[t + 1, j]
                b[t, i] = s / c[t + 1]
        xi = np.zeros((K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[i, j] += a[t, i] * G[i, j] * p[t + 1, j] * b[t + 1, j] / c[t + 1]
        g0 = a[0] * b[0]
        delta = g0 / g0.sum()
        for i in range(K):
            G[i] = xi[i] / xi[i].sum()
            sw, sx = 0.0, 0.0
            for t in range(T):
                w = a[t, i] * b[t, i]
                sw += w
                sx += w * x[t]
            mu[i] = sx / sw
            sv = 0.0
            for t in range(T):
                sv += a[t, i] * b[t, i] * (x[t] - mu[i]) ** 2
            var[i] = max(sv / sw, var_floor)
        if ll - ll_old < tol:
            break
        ll_old = ll
    return mu, var, G, delta, ll


def fit(x, n_components=2, n_init=10, max_iter=500, tol=1e-8, seed=0, persistent=False):
    """Maximum likelihood by Baum-Welch from `n_init` starts whose means come from k-means++; keeps the best
    log-likelihood. Start values for the other parameters follow hmmlearn's defaults (uniform start and transition
    probabilities, pooled variance). Returns dict(mu, var, G, delta, ll).

    `persistent=True` starts every run from a low/high-variance split with sticky transitions instead, and keeps the
    best fit whose states are both persistent (P(stay) > 0.5); it falls back to the plain best fit if none is."""
    x = np.ascontiguousarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    v0 = x.var()
    best = best_any = None
    for _ in range(n_init):
        mu0 = kmeans_plusplus(x[:, None], n_components, random_state=rng)[0][:, 0].copy()
        if persistent:
            var0 = v0 * np.geomspace(0.5, 2.0, n_components)
            G0 = np.full((n_components, n_components), 0.02 / (n_components - 1))
            np.fill_diagonal(G0, 0.98)
        else:
            var0 = np.full(n_components, v0)
            G0 = np.full((n_components, n_components), 1.0 / n_components)
        d0 = np.full(n_components, 1.0 / n_components)
        mu, var, G, delta, _ = _em(x, mu0, var0, G0, d0, max_iter, tol, 1e-12 * v0)
        ll = forward(x, mu, var, G, delta)[1]
        res = {"mu": mu, "var": var, "G": G, "delta": delta, "ll": ll}
        if best_any is None or ll > best_any["ll"]:
            best_any = res
        if (not persistent or np.diag(G).min() > 0.5) and (best is None or ll > best["ll"]):
            best = res
    return best if best is not None else best_any


@njit(cache=True)
def viterbi(x, mu, var, G, delta):
    T = x.size
    K = mu.size
    lp = _logdens(x, mu, var)
    lG = np.log(G)
    v = np.empty((T, K))
    bp = np.empty((T, K), dtype=np.int64)
    v[0] = np.log(delta) + lp[0]
    for t in range(1, T):
        for j in range(K):
            best, arg = -np.inf, 0
            for i in range(K):
                c = v[t - 1, i] + lG[i, j]
                if c > best:
                    best, arg = c, i
            v[t, j] = best + lp[t, j]
            bp[t, j] = arg
    s = np.empty(T, dtype=np.int64)
    s[T - 1] = np.argmax(v[T - 1])
    for t in range(T - 1, 0, -1):
        s[t - 1] = bp[t, s[t]]
    return s
