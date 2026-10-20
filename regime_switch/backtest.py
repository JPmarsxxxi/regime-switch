"""Backtests with one-way proportional costs and an explicit trading delay."""

import numpy as np


def zero_one(returns, rf, bear, delay=1, cost=0.001):
    """Shu, Yu & Mulvey (2024) 0/1 strategy.

    `bear[t]` is the regime inferred at the end of day t (1 = unfavourable, NaN = no signal yet). With delay d the
    allocation it implies is held from day t + 1 + d: "if the prevailing regime inferred at the end of day t differs
    ... the switch in allocation can only be executed at the end of next day t + 1 ... applied starting from day t + 2".
    A NaN signal means fully invested. Returns (daily total returns after costs, daily risky weight).
    """
    r = np.asarray(returns, dtype=float)
    f = np.asarray(rf, dtype=float)
    b = np.asarray(bear, dtype=float)
    w = np.ones(r.size)
    lag = 1 + delay
    sig = np.where(np.isnan(b), 0.0, b)
    w[lag:] = 1 - sig[:-lag]
    trade = np.abs(np.diff(w, prepend=w[0]))
    return w * r + (1 - w) * f - cost * trade, w


def weighted(returns, weights, cost=0.0):
    """Daily returns of a position held at `weights[t]` over day t (weights already lagged by the caller)."""
    w = np.asarray(weights, dtype=float)
    trade = np.abs(np.diff(w, prepend=0.0))
    return w * np.asarray(returns, dtype=float) - cost * trade
