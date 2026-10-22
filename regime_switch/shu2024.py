"""Shu, Yu & Mulvey (2024), "Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach".

Pipeline: features (Table 2) -> JM refit every 6 months on 3000 days, online inference over a 3000-day lookback
(Sec. 3.4) | HMM refit daily on 3000 days, last Viterbi state, median filter (Sec. 3.3) -> monthly time-series CV of
the smoothing hyperparameter on an 8-year window (Sec. 3.4.3) -> 0/1 strategy with delay and 10 bp costs (Sec. 3.1).
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import hmm, jump
from .backtest import zero_one
from .metrics import shu_metrics

LAMBDAS = (0.0, 5.0, 15.0, 35.0, 50.0, 70.0, 100.0, 150.0)  # see DEVIATIONS.md#S2
HMM_KS = (0, 2, 4, 6, 8, 20)  # see DEVIATIONS.md#S2


@dataclass
class Config:
    train: int = 3000
    lookback: int = 3000
    refit_months: int = 6
    val_years: int = 8
    delay: int = 1
    cost: float = 0.001
    n_init: int = 10
    lambdas: tuple = LAMBDAS
    ks: tuple = HMM_KS
    seed: int = 0
    workers: int = 4
    burn_in: int = 252
    extra: dict = field(default_factory=dict)


def downside_deviation(excess, halflife):
    return np.sqrt(np.minimum(excess, 0.0).pow(2).ewm(halflife=halflife).mean())


def features(excess):
    """Table 2: EWM downside deviation (hl 10) and EWM Sortino ratios (hl 20, 60) of the daily excess return.
    Rows before the first negative excess return have DD = 0 and an undefined Sortino ratio; they are NaN."""
    return pd.DataFrame({
        "DD_10": downside_deviation(excess, 10),
        "Sortino_20": excess.ewm(halflife=20).mean() / downside_deviation(excess, 20),
        "Sortino_60": excess.ewm(halflife=60).mean() / downside_deviation(excess, 60),
    }).replace([np.inf, -np.inf], np.nan)


def refit_dates(index, first_fit, months):
    """See DEVIATIONS.md#S5. Last trading day of every `months`-month block, starting at the first block end on or after `first_fit`."""
    ends = pd.Series(index, index=index).groupby(index.to_period("M")).last()
    ends = ends[ends.index.month % months == 0]
    return [d for d in ends if d >= first_fit]


def jm_online(feat, excess, lam, cfg: Config):
    """Online JM regimes (1 = bear) for every day after the first fit; NaN before it."""
    X_all = feat.to_numpy()
    out = pd.Series(np.nan, index=feat.index)
    first = feat.index[cfg.train - 1]
    fits = refit_dates(feat.index, first, cfg.refit_months)
    for i, d in enumerate(fits):
        end = feat.index.get_loc(d) + 1
        Xtr = X_all[end - cfg.train : end]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)  # no 3-sd clipping, see DEVIATIONS.md#S6
        centers, states, _ = jump.fit((Xtr - mu) / sd, lam, n_init=cfg.n_init, seed=cfg.seed + i)
        ex_tr = excess.to_numpy()[end - cfg.train : end]
        occupied = [k for k in range(2) if np.any(states == k)]
        cum = {k: ex_tr[states == k].sum() for k in occupied}
        # a fit that collapses to one state labels that state by the sign of its own cumulative excess return
        bull = max(cum, key=cum.get) if len(occupied) == 2 or cum[occupied[0]] >= 0 else 1 - occupied[0]
        stop = feat.index.get_loc(fits[i + 1]) + 1 if i + 1 < len(fits) else len(feat)
        a = max(0, end - cfg.lookback)
        lab = jump.online_states((X_all[a:stop] - mu) / sd, centers, lam, cfg.lookback, start=end - 1 - a)
        # the fit at the end of day d is used for day d itself and every day until the next refit
        out.iloc[end - 1 : stop] = (lab != bull).astype(float)
    return out


def _hmm_chunk(args):  # see DEVIATIONS.md#S4
    x, starts, train, n_init, seed = args
    res = np.empty(len(starts))
    for j, end in enumerate(starts):
        w = x[end - train : end]
        p = hmm.fit(w, n_init=n_init, seed=seed + end)
        s = hmm.viterbi(w, p["mu"], p["var"], p["G"], p["delta"])
        res[j] = float(s[-1] == np.argmax(p["var"]))
    return res


def hmm_online_raw(log_returns, cfg: Config, start=None):
    """Sec. 3.3: daily rolling 2-state Gaussian HMM on daily log total returns, last Viterbi state, 1 = high vol."""
    x = log_returns.to_numpy()
    first = cfg.train if start is None else max(cfg.train, start)
    ends = np.arange(first, len(x) + 1)
    chunks = np.array_split(ends, cfg.workers * 8)
    args = [(x, c, cfg.train, cfg.n_init, cfg.seed) for c in chunks if len(c)]
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as ex:
            parts = list(ex.map(_hmm_chunk, args))
    else:
        parts = [_hmm_chunk(a) for a in args]
    out = pd.Series(np.nan, index=log_returns.index)
    out.iloc[ends - 1] = np.concatenate(parts)
    return out


def median_filter(raw, k):
    """Rolling mean of the online states over the trailing k days; bear only if it exceeds 0.5 (k = 0: raw)."""
    if k <= 1:
        return raw
    m = raw.rolling(k, min_periods=k).mean()
    return (m > 0.5).astype(float).where(m.notna())


def cross_validate(candidates: dict, returns, rf, cfg: Config, test_start):
    """Sec. 3.4.3. At the end of each month, pick the candidate whose 0/1 strategy had the highest Sharpe over the
    trailing `val_years`; its regime series drives the signal from that day until the next month end."""
    idx = returns.index
    ex = {}
    for c, bear in candidates.items():
        strat, _ = zero_one(returns.to_numpy(), rf.to_numpy(), bear.to_numpy(), cfg.delay, cfg.cost)
        ex[c] = pd.Series(strat - rf.to_numpy(), index=idx)
    ex = pd.DataFrame(ex)
    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    month_ends = month_ends[month_ends >= idx[idx.searchsorted(test_start) - 1]]
    choice = pd.Series(np.nan, index=idx, dtype=object)
    bear = pd.Series(np.nan, index=idx)
    chosen = []
    for i, d in enumerate(month_ends):
        win = ex.loc[d - pd.DateOffset(years=cfg.val_years) : d]
        valid = [c for c in ex.columns if candidates[c].loc[win.index].notna().all()]
        if not valid:
            continue
        sr = win[valid].mean() / win[valid].std(ddof=1)
        best = sr.idxmax()
        chosen.append((d, best))
        stop = month_ends.iloc[i + 1] if i + 1 < len(month_ends) else idx[-1] + pd.Timedelta(days=1)
        sel = (idx >= d) & (idx < stop)
        bear[sel] = candidates[best][sel]
        choice[sel] = best
    return bear, pd.DataFrame(chosen, columns=["date", "choice"]).set_index("date")


@dataclass
class Result:
    returns: pd.DataFrame  # daily total returns of B&H, JM and HMM strategies over the test period
    weights: pd.DataFrame
    bear: pd.DataFrame  # final (CV-selected) online regimes
    choices: dict
    candidates: dict  # model -> {param: regime series}
    metrics: pd.DataFrame


def candidates(price, rf_daily, cfg: Config | None = None, hmm_raw=None):
    """Online regimes for every candidate hyperparameter. Returns (simple returns, daily rf, {"JM": {lambda: bear},
    "HMM": {k: bear}}, raw HMM states). `hmm_raw` may pass a precomputed `hmm_online_raw` series (the expensive part)."""
    cfg = cfg or Config()
    r = price.pct_change().dropna()
    rf = rf_daily.reindex(r.index).ffill()
    excess = r - rf
    feat = features(excess)
    # EWM warm-up: the first rows have a near-zero DD and Sortino ratios in the hundreds, which capture a centroid
    valid = feat.notna().all(axis=1).to_numpy().argmax() + cfg.burn_in  # see DEVIATIONS.md#S3
    jm = {lam: jm_online(feat.iloc[valid:], excess.iloc[valid:], lam, cfg).reindex(r.index) for lam in cfg.lambdas}
    raw = hmm_raw if hmm_raw is not None else hmm_online_raw(np.log1p(r), cfg)
    hm = {k: median_filter(raw, k) for k in cfg.ks}
    return r, rf, {"JM": jm, "HMM": hm}, raw


def default_test_start(cands, cfg: Config):
    """Start of the first month whose selection (made at the previous month end) sees a full validation window for
    every candidate of every model."""
    first = max(s.first_valid_index() for model in cands.values() for s in model.values())
    return (first + pd.DateOffset(years=cfg.val_years, months=1)).to_period("M").to_timestamp()


def evaluate(r, rf, cands, cfg: Config, test_start):
    """CV-select each model's hyperparameter monthly and run its 0/1 strategy over the test period."""
    test = r.index >= pd.Timestamp(test_start)
    rets, wts = {"B&H": r[test]}, {"B&H": pd.Series(1.0, index=r.index[test])}
    bears, choices = {}, {}
    for name in ("HMM", "JM"):
        bear, choice = cross_validate(cands[name], r, rf, cfg, test_start)
        b = bear.copy()
        b[~test] = np.nan
        strat, w = zero_one(r.to_numpy(), rf.to_numpy(), b.to_numpy(), cfg.delay, cfg.cost)
        rets[name] = pd.Series(strat, index=r.index)[test]
        wts[name] = pd.Series(w, index=r.index)[test]
        bears[name], choices[name] = bear[test], choice
    rets, wts = pd.DataFrame(rets), pd.DataFrame(wts)
    met = pd.DataFrame({c: shu_metrics(rets[c], rf[test], wts[c]) for c in rets.columns})
    return Result(rets, wts, pd.DataFrame(bears), choices, cands, met)


def run(price, rf_daily, cfg: Config | None = None, test_start=None, hmm_raw=None):
    """`price`: index level; `rf_daily`: daily risk-free return on the same calendar. Returns `Result`."""
    cfg = cfg or Config()
    r, rf, cands, _ = candidates(price, rf_daily, cfg, hmm_raw)
    return evaluate(r, rf, cands, cfg, test_start or default_test_start(cands, cfg))
