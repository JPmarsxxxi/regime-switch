"""Statistical jump model (Bemporad et al. 2018; Nystrup et al. 2020b) as used by Shu, Yu & Mulvey (2024), eq. 1:

    min_{Theta, S}  sum_t 0.5 * ||x_t - theta_{s_t}||^2  +  lambda * sum_t 1{s_{t-1} != s_t}
"""

import numpy as np
from numba import njit
from sklearn.cluster import kmeans_plusplus


@njit(cache=True)
def dp_values(loss, lam):
    T, K = loss.shape
    v = np.empty((T, K))
    v[0] = loss[0]
    for t in range(1, T):
        for j in range(K):
            best = np.inf
            for i in range(K):
                c = v[t - 1, i] + (0.0 if i == j else lam)
                best = min(best, c)
            v[t, j] = loss[t, j] + best
    return v


@njit(cache=True)
def dp_path(loss, lam):
    """Exact minimiser of the state sequence for fixed centroids. Ties go to the lowest state index."""
    v = dp_values(loss, lam)
    T, K = loss.shape
    s = np.empty(T, dtype=np.int64)
    s[T - 1] = np.argmin(v[T - 1])
    val = v[T - 1, s[T - 1]]
    for t in range(T - 1, 0, -1):
        best, arg = np.inf, 0
        for i in range(K):
            c = v[t - 1, i] + (0.0 if i == s[t] else lam)
            if c < best:
                best, arg = c, i
        s[t - 1] = arg
    return s, val


def loss_matrix(X, centers):
    d = 0.5 * ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.where(np.isnan(d), np.inf, d)


def objective(X, centers, states, lam):
    return float(0.5 * ((X - centers[states]) ** 2).sum() + lam * np.sum(states[1:] != states[:-1]))


def fit(X, lam, n_components=2, n_init=10, max_iter=1000, tol=1e-8, seed=0):
    """Coordinate descent from `n_init` k-means++ starts; keeps the lowest objective. Returns (centers, states, value)."""
    X = np.ascontiguousarray(X, dtype=float)
    rng = np.random.RandomState(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        centers = kmeans_plusplus(X, n_components, random_state=rng)[0]
        states, val = dp_path(loss_matrix(X, centers), lam)
        prev_states, prev_val = None, np.inf
        it = 0
        while it < max_iter and (prev_states is None or not np.array_equal(states, prev_states)) and prev_val - val > tol:
            it += 1
            prev_states, prev_val = states, val
            centers = np.array([X[states == k].mean(axis=0) if np.any(states == k) else np.full(X.shape[1], np.nan)
                                for k in range(n_components)])
            states, val = dp_path(loss_matrix(X, centers), lam)
        if val < best[2]:
            best = (centers, states, val)
    return best


@njit(cache=True)
def _online(loss, lam, lookback, start):
    T, K = loss.shape
    out = np.empty(T - start, dtype=np.int64)
    for t in range(start, T):
        a = max(0, t - lookback + 1)
        v = loss[a].copy()
        nv = np.empty(K)
        for u in range(a + 1, t + 1):
            for j in range(K):
                best = np.inf
                for i in range(K):
                    c = v[i] + (0.0 if i == j else lam)
                    best = min(best, c)
                nv[j] = loss[u, j] + best
            v[:] = nv
        out[t - start] = np.argmin(v)
    return out


def online_states(X, centers, lam, lookback, start=0):
    """Shu et al. Sec. 3.4.2: for each day t >= start, solve the DP over x_{t-l+1..t} with fixed centroids and keep
    the last state of the optimal sequence (= argmin of the final DP values)."""
    return _online(loss_matrix(np.asarray(X, dtype=float), centers), float(lam), int(lookback), int(start))
