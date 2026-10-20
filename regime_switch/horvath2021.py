"""Horvath, Issa & Muguruza (2021), "Clustering Market Regimes using the Wasserstein Distance".

Stream lift (eq. 3) -> empirical measures -> WK-means (Algorithm 1: W_p assignment, Wasserstein-barycenter update,
summed-centroid-move stopping rule, eq. 23), with MK-means (eqs. 24-25, Algorithm 2) and a Gaussian HMM as
benchmarks, scored by ROFS / RONS / TA (eqs. 31-33) on the paper's synthetic paths (Sec. 3.3).
"""

import numpy as np

from . import hmm
from .sim import horvath_gbm, horvath_merton


def lift(x, h1, h2):
    """Segments of length h1 overlapping by h2 (step h1 - h2). Returns (segments (M, h1), start index of each).
    See DEVIATIONS.md#H1."""
    step = h1 - h2
    starts = np.arange(0, x.size - h1 + 1, step)
    return x[starts[:, None] + np.arange(h1)[None, :]], starts


def wasserstein(a_sorted, b_sorted, p=1):
    """Eq. 21 for equal-size empirical measures given as sorted atoms (broadcasts over leading axes)."""
    return np.mean(np.abs(a_sorted - b_sorted) ** p, axis=-1) ** (1 / p)


def barycenter(sorted_atoms, p=1):
    """Prop. 2.6 for p = 1 (coordinate-wise median of the sorted atoms); the coordinate-wise mean for p = 2, where it
    is the exact W_2 barycenter of 1-d measures. Other p are not covered by the paper."""
    if p == 1:
        return np.median(sorted_atoms, axis=0)
    if p == 2:
        return sorted_atoms.mean(axis=0)
    raise ValueError("barycenter implemented for p in {1, 2}")


def wk_means(segments, k=2, p=1, tol=1e-8, max_iter=1000, seed=0):
    """Algorithm 1. Returns (labels, centroids as sorted atoms (k, h1))."""
    rng = np.random.default_rng(seed)
    S = np.sort(segments, axis=1)
    cent = S[rng.choice(len(S), size=k, replace=False)]
    labels = np.zeros(len(S), dtype=int)
    for _ in range(max_iter):
        d = wasserstein(S[:, None, :], cent[None, :, :], p)
        labels = d.argmin(axis=1)
        new = np.array([barycenter(S[labels == j], p) if np.any(labels == j) else cent[j] for j in range(k)])
        loss = wasserstein(cent, new, p).sum()  # eq. 23
        cent = new
        if loss < tol:
            break
    labels = wasserstein(S[:, None, :], cent[None, :, :], p).argmin(axis=1)
    return labels, cent


def raw_moments(segments, p=4):
    """Eq. 24: (1/n!) * n-th raw moment, n = 1..p."""
    from math import factorial

    return np.column_stack([np.mean(segments**n, axis=1) / factorial(n) for n in range(1, p + 1)])


def mk_means(segments, k=2, p=4, tol=1e-8, max_iter=1000, seed=0):
    """Algorithm 2 on the standardised moment vectors (Def. 3.1, Remark 3.2). Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    Z = raw_moments(segments, p)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    cent = Z[rng.choice(len(Z), size=k, replace=False)]
    for _ in range(max_iter):
        labels = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        new = np.array([Z[labels == j].mean(axis=0) if np.any(labels == j) else cent[j] for j in range(k)])
        loss = np.sqrt(((cent - new) ** 2).sum(axis=1)).sum()  # eq. 44
        cent = new
        if loss < tol:
            break
    labels = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return labels, cent


def regime_on_cluster(segments, labels, k=2):
    """Regime-on (regime change) = the cluster whose members have the higher average variance."""
    return int(np.argmax([segments[labels == j].var(axis=1).mean() if np.any(labels == j) else -np.inf
                          for j in range(k)]))


def label_counts(n, starts, h1, seg_on):
    """Y^i of Sec. 3.3: for every return, how many of the segments containing it are regime-off / regime-on."""
    on = np.zeros(n)
    tot = np.zeros(n)
    for s0, flag in zip(starts, seg_on):
        tot[s0 : s0 + h1] += 1
        on[s0 : s0 + h1] += flag
    return tot - on, on


def accuracy(y_off, y_on, truth_on):
    """Eqs. 31-33 over the returns covered by at least one label."""
    tot = y_off + y_on
    m = tot > 0
    N, R = m & ~truth_on, m & truth_on
    return {
        "ROFS": y_off[N].sum() / tot[N].sum(),
        "RONS": y_on[R].sum() / tot[R].sum(),
        "TA": (y_off[N].sum() + y_on[R].sum()) / tot[m].sum(),
    }


def hmm_labels(x, seed=0):
    """Gaussian 2-state HMM by maximum likelihood, Viterbi path; regime-on = higher-variance state."""
    p = hmm.fit(x, n_init=10, seed=seed)
    s = hmm.viterbi(np.ascontiguousarray(x), p["mu"], p["var"], p["G"], p["delta"])
    return (s == np.argmax(p["var"])).astype(float)


def hmmlearn_default_labels(x, seed=0):
    """The same benchmark through hmmlearn with its default settings (n_iter=10, tol=1e-2, min_covar=1e-3,
    covars_prior=1e-2). See DEVIATIONS.md#H3."""
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(n_components=2, random_state=seed).fit(x[:, None])
    s = m.predict(x[:, None])
    return (s == np.argmax(m.covars_.ravel())).astype(float)


def mmd2_biased(x, y, sigma=0.1):
    """Biased MMD^2 with a Gaussian kernel (eq. 10 with F the unit ball of the Gaussian RKHS)."""
    def k(a, b):
        return np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * sigma**2))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def self_similarity(segs_a, segs_b, n=1000, sigma=0.1, seed=0):
    """Def. 1.9: median MMD^2 over n random pairs (one segment from each set)."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(segs_a), n)
    ib = rng.integers(0, len(segs_b), n)
    return float(np.median([mmd2_biased(np.sort(segs_a[i]), np.sort(segs_b[j]), sigma) for i, j in zip(ia, ib)]))


def synthetic_run(model, seed, h1=35, h2=28, k=2, p=1, with_hmmlearn=True):
    """One run of the Sec. 3.3 experiment. Returns accuracy dicts per method plus centroid moments."""
    x, on = (horvath_gbm if model == "gbm" else horvath_merton)(seed)
    segs, starts = lift(x, h1, h2)
    out = {}
    for name, fn in (("Wasserstein", lambda: wk_means(segs, k, p, seed=seed)),
                     ("Moment", lambda: mk_means(segs, k, seed=seed))):
        labels, cent = fn()
        j_on = regime_on_cluster(segs, labels, k)
        y_off, y_on = label_counts(x.size, starts, h1, labels == j_on)
        out[name] = accuracy(y_off, y_on, on)
        out[name]["centroid_mean_on"] = float(segs[labels == j_on].mean()) if name == "Moment" else float(cent[j_on].mean())
        out[name]["centroid_var_on"] = (float(segs[labels == j_on].var()) if name == "Moment" else float(cent[j_on].var()))
        off = 1 - j_on
        out[name]["centroid_mean_off"] = float(segs[labels == off].mean()) if name == "Moment" else float(cent[off].mean())
        out[name]["centroid_var_off"] = (float(segs[labels == off].var()) if name == "Moment" else float(cent[off].var()))
        out[name]["labels"] = labels
    hl = hmm_labels(x, seed)
    out["HMM"] = accuracy(1 - hl, hl, on)
    if with_hmmlearn:
        hd = hmmlearn_default_labels(x, seed)
        out["HMM (hmmlearn defaults)"] = accuracy(1 - hd, hd, on)
    out["segments"] = segs
    return out
