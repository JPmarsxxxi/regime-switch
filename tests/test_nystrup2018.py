import numpy as np
import pytest

from regime_switch import hmm
from regime_switch import nystrup2018 as ny
from regime_switch.sim import markov_switching_gaussian


def _theta():
    return ny.pack(np.array([0.0006, -0.0009]), np.array([0.008, 0.02]) ** 2, 0.98, 0.95)


def test_score_matches_finite_differences():
    x = markov_switching_gaussian(400, seed=2).log_returns
    th = _theta()
    _, last, scores, _ = ny.score_pass(x, th, 0.99)
    for q in range(ny.P):
        e = np.zeros(ny.P)
        h = 1e-6 * max(1.0, abs(th[q]))
        e[q] = h
        fd = (ny.score_pass(x, th + e, 0.99)[3] - ny.score_pass(x, th - e, 0.99)[3]) / (2 * h)
        assert scores[:, q] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    assert last == pytest.approx(scores[-1])


def test_weighted_score_is_discounted_sum():
    x = markov_switching_gaussian(300, seed=5).log_returns
    f = 0.99
    w, _, scores, _ = ny.score_pass(x, _theta(), f)
    assert w == pytest.approx(f ** np.arange(299, -1, -1) @ scores, rel=1e-10)


def test_loglik_matches_forward_with_stationary_start():
    x = markov_switching_gaussian(500, seed=6).log_returns
    th = _theta()
    mu, var, g = ny.unpack(th)
    G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    pi = np.array([1 - g[1], 1 - g[0]]) / (2 - g.sum())
    assert ny.score_pass(x, th, 0.99)[3].sum() == pytest.approx(hmm.forward(x, mu, var, G, pi)[1], rel=1e-12)


def test_forecast_moments_match_monte_carlo():
    th = _theta()
    alpha = np.array([0.3, 0.7])
    means, variances = ny.forecast_moments(alpha, th, 5)
    mu, var, g = ny.unpack(th)
    G = np.array([[g[0], 1 - g[0]], [1 - g[1], g[1]]])
    rng = np.random.default_rng(0)
    n = 1_000_000
    s = (rng.random(n) < alpha[1]).astype(int)
    for k in range(5):
        stay = rng.random(n) < G[s, s]
        s = np.where(stay, s, 1 - s)
        r = np.expm1(mu[s] + np.sqrt(var[s]) * rng.standard_normal(n))
        se = r.std() / np.sqrt(n)
        assert abs(r.mean() - means[k]) < 3 * se
        assert abs(r.var() - variances[k]) < 3 * r.var() * np.sqrt(2 / n) * 3


def test_mpc_risk_neutral_zero_cost_is_bang_bang():
    m = ny.MPC(K=20, gamma=0, rho=0, kappa=0)
    up, down = np.full(20, 0.001), np.full(20, -0.001)
    var = np.full(20, 1e-4)
    for h in (np.array([0.0, 1.0]), np.array([0.4, 0.6]), np.array([1.0, 0.0])):
        assert h[0] + m.first_trade(h, up, var) == pytest.approx(1.0, abs=1e-6)
        assert h[0] + m.first_trade(h, down, var) == pytest.approx(0.0, abs=1e-6)


def test_mpc_costs_create_no_trade_band():
    m = ny.MPC(K=100, gamma=0, rho=0, kappa=0.02)
    var = np.full(100, 1e-4)
    small = np.full(100, 0.0)
    small[:5] = -0.002  # a short bad patch is not worth two 2% trades
    # interior-point solver: trades are exact to ~1e-5 of portfolio value
    assert m.first_trade(np.array([1.0, 0.0]), small, var) == pytest.approx(0.0, abs=1e-4)
    assert m.first_trade(np.array([0.0, 1.0]), -small, var) == pytest.approx(0.0, abs=1e-4)
    big = np.full(100, -0.002)
    assert m.first_trade(np.array([1.0, 0.0]), big, var) == pytest.approx(-1.0, abs=1e-4)
