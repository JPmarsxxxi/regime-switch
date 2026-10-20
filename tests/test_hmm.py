import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from regime_switch import hmm
from regime_switch.sim import markov_switching_gaussian


def _ref(mu, var, G, delta, **kw):
    # hmmlearn's default priors and covariance floor are sized for unit-scale data; switch them off for returns
    m = GaussianHMM(n_components=2, covariance_type="diag", init_params="", params="stmc", min_covar=1e-300,
                    covars_prior=0.0, covars_weight=0.0, means_prior=0.0, means_weight=0.0, **kw)
    m.startprob_, m.transmat_ = delta, G
    m.means_, m.covars_ = mu[:, None], var[:, None]
    return m


def test_forward_matches_hmmlearn():
    x = markov_switching_gaussian(2000, seed=3).log_returns
    mu, var = np.array([0.0005, -0.001]), np.array([0.007, 0.02]) ** 2
    G, d = np.array([[0.99, 0.01], [0.03, 0.97]]), np.array([0.6, 0.4])
    a, ll = hmm.forward(x, mu, var, G, d)
    m = _ref(mu, var, G, d)
    assert ll == pytest.approx(m.score(x[:, None]), rel=1e-10)
    _, fwd = m.score_samples(x[:, None])
    # hmmlearn returns smoothed posteriors; compare filtered probs on the last observation, where they coincide
    assert a[-1] == pytest.approx(fwd[-1], abs=1e-10)
    assert np.array_equal(hmm.viterbi(x, mu, var, G, d), m.predict(x[:, None]))


def test_baum_welch_matches_hmmlearn_from_same_start():
    x = markov_switching_gaussian(3000, seed=4).log_returns
    mu0, var0 = np.array([0.001, -0.002]), np.full(2, x.var())
    G0, d0 = np.full((2, 2), 0.5), np.full(2, 0.5)
    mu, var, G, d, _ = hmm._em(x, mu0.copy(), var0.copy(), G0.copy(), d0.copy(), 2000, 1e-10, 0.0)
    m = _ref(mu0, var0, G0, d0, n_iter=2000, tol=1e-10)
    m.fit(x[:, None])
    order = np.argsort(m.means_[:, 0])[::-1]
    assert hmm.forward(x, mu, var, G, d)[1] == pytest.approx(m.score(x[:, None]), abs=1e-4)
    assert np.sort(mu)[::-1] == pytest.approx(m.means_[order, 0], abs=1e-6)
