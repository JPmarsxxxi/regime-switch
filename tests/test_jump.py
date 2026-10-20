import itertools

import numpy as np
import pytest
from jumpmodels.jump import JumpModel

from regime_switch import jump


def brute_force(loss, lam):
    T, K = loss.shape
    best, arg = np.inf, None
    for s in itertools.product(range(K), repeat=T):
        s = np.array(s)
        v = loss[np.arange(T), s].sum() + lam * np.sum(s[1:] != s[:-1])
        if v < best - 1e-12:
            best, arg = v, s
    return arg, best


def test_dp_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        T, K = rng.integers(1, 11), rng.integers(2, 4)
        loss = rng.random((T, K)) * 3
        lam = rng.random() * 2
        s, v = jump.dp_path(loss, lam)
        _, bv = brute_force(loss, lam)
        assert v == pytest.approx(bv, abs=1e-12)
        assert loss[np.arange(T), s].sum() + lam * np.sum(s[1:] != s[:-1]) == pytest.approx(bv, abs=1e-12)


def test_online_is_last_state_of_window_optimum():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 2))
    centers = np.array([[0.5, 0.0], [-0.5, 0.2]])
    lam, lb = 0.7, 8
    on = jump.online_states(X, centers, lam, lb)
    loss = jump.loss_matrix(X, centers)
    for t in range(40):
        a = max(0, t - lb + 1)
        s, _ = brute_force(loss[a : t + 1], lam)
        v = jump.dp_values(loss[a : t + 1], lam)[-1]
        assert v[on[t]] == pytest.approx(v.min())
        assert on[t] == s[-1] or v[s[-1]] == pytest.approx(v[on[t]])


@pytest.mark.parametrize("seed", range(5))
def test_fit_matches_reference_package(seed):
    rng = np.random.default_rng(seed)
    s = np.repeat(rng.integers(0, 2, 20), 25)
    X = rng.standard_normal((500, 3)) + np.where(s[:, None] == 1, 1.0, -0.5)
    lam = float(rng.choice([0.0, 2.0, 10.0, 50.0]))
    ours = jump.fit(X, lam, n_init=10, seed=seed)
    ref = JumpModel(n_components=2, jump_penalty=lam, n_init=10, random_state=np.random.RandomState(seed)).fit(X)
    assert ours[2] <= ref.val_ * (1 + 1e-8) + 1e-12
    assert ours[2] == pytest.approx(jump.objective(X, ours[0], ours[1], lam))
