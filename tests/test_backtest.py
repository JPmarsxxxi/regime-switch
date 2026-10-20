import numpy as np
import pytest

from regime_switch.backtest import zero_one


def test_delay_and_cost_are_exact():
    n = 12
    r = np.full(n, 0.01)
    rf = np.full(n, 0.001)
    bear = np.zeros(n)
    bear[4:] = 1.0  # regime flips at the end of day 4
    for d in (1, 5):
        strat, w = zero_one(r, rf, bear, delay=d, cost=0.001)
        first_cash = 4 + 1 + d
        assert np.all(w[:first_cash] == 1) and np.all(w[first_cash:] == 0)
        assert strat[first_cash] == pytest.approx(0.001 - 0.001)
        assert strat[first_cash - 1] == pytest.approx(0.01)
        assert np.all(strat[first_cash + 1 :] == pytest.approx(0.001))


def test_nan_signal_means_invested():
    strat, w = zero_one(np.ones(5) * 0.01, np.zeros(5), np.full(5, np.nan))
    assert np.all(w == 1) and np.allclose(strat, 0.01)
