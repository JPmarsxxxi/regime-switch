"""Every detector on every generator (SPEC: cross-generator study, criteria 26-27)."""

import sys

import numpy as np
import pandas as pd

from regime_switch import daniel_moskowitz as dm
from regime_switch import hmm, shu2024
from regime_switch import horvath2021 as hv
from regime_switch import nystrup2018 as ny
from regime_switch.backtest import zero_one
from regime_switch.metrics import balanced_accuracy, detection_latency, sharpe, shifts_per_year
from regime_switch.sim import (
    BENCH_MU,
    BENCH_SIGMA,
    BENCH_STAY,
    drifting_hmm,
    horvath_daily,
    markov_switching_gaussian,
    markov_switching_t,
    momentum_panic,
)
from scripts.common import ROOT, cached, verdict

YEARS, SEEDS = 30, 5
DEPS = ["regime_switch/shu2024.py", "regime_switch/jump.py", "regime_switch/hmm.py", "regime_switch/nystrup2018.py",
        "regime_switch/horvath2021.py", "regime_switch/daniel_moskowitz.py", "regime_switch/backtest.py",
        "regime_switch/metrics.py", "regime_switch/sim.py", "scripts/cross_generators.py"]
DETECTORS = ["Shu JM", "Shu HMM", "Nystrup adaptive HMM", "Horvath WK-means (a posteriori)", "D&M bear indicator"]


def filter_const(x, mu, sigma, stay):
    G = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    pi = np.array([1 - G[1, 1], 1 - G[0, 0]]) / (2 - G[0, 0] - G[1, 1])
    return hmm.forward(np.ascontiguousarray(x), np.asarray(mu), np.asarray(sigma) ** 2, G, pi)[0][:, 1] > 0.5


def filter_drifting(x, mu, sigma, stay):
    G = np.array([[stay[0], 1 - stay[0]], [1 - stay[1], stay[1]]])
    a = np.array([0.5, 0.5])
    out = np.empty(x.size, dtype=bool)
    for t in range(x.size):
        lp = -0.5 * ((x[t] - mu[t]) / sigma[t]) ** 2 - np.log(sigma[t])
        u = (a @ G if t else a) * np.exp(lp - lp.max())
        a = u / u.sum()
        out[t] = a[1] > 0.5
    return out


def generate(name, seed):
    """Daily log-returns, true regime (1 = unfavourable) and the oracle filter's labels (or None)."""
    n = 252 * YEARS
    if name == "Markov-switching Gaussian":
        p = markov_switching_gaussian(n, seed)
        return p.log_returns, p.states, filter_const(p.log_returns, BENCH_MU, BENCH_SIGMA, BENCH_STAY)
    if name == "Markov-switching Student-t":
        p = markov_switching_t(n, seed)
        return p.log_returns, p.states, filter_const(p.log_returns, BENCH_MU, BENCH_SIGMA, BENCH_STAY)
    if name == "Drifting-parameter HMM":
        p = drifting_hmm(n, seed)
        return p.log_returns, p.states, filter_drifting(p.log_returns, p.mu, p.sigma, BENCH_STAY)
    if name in ("Horvath gBm (daily)", "Horvath Merton (daily)"):
        x, on = horvath_daily("gbm" if "gBm" in name else "merton", seed, YEARS)
        return x, on.astype(int), None
    if name == "Momentum-panic market":
        p = momentum_panic(seed, YEARS)
        x = np.log1p(p.mkt)
        return x, p.regime, filter_const(x, (0.0005, -0.0008), (0.008, 0.022), (0.998, 0.99))
    raise ValueError(name)


GENERATORS = ["Markov-switching Gaussian", "Markov-switching Student-t", "Drifting-parameter HMM",
              "Horvath gBm (daily)", "Horvath Merton (daily)", "Momentum-panic market"]


def run_path(name, seed):
    print(name, seed, flush=True)
    x, truth, oracle = generate(name, seed)
    idx = pd.bdate_range("1980-01-01", periods=x.size + 1)
    price = pd.Series(np.exp(np.r_[0.0, np.cumsum(x)]), index=idx)
    r = price.pct_change().dropna()
    rf = pd.Series(0.0, index=idx)
    res = shu2024.run(price, rf, shu2024.Config(seed=seed))
    logr = np.log1p(r.to_numpy())
    th = ny.adaptive_estimates(logr)
    alpha = ny.filtered_probs(logr, th)
    ny_bear = np.full(logr.size, np.nan)
    for t in range(259, logr.size):
        hi = int(np.argmax(th[t, 2:4]))
        ny_bear[t] = float(alpha[t, hi] > 0.5)
    segs, starts = hv.lift(logr, 35, 28)
    labels, _ = hv.wk_means(segs, seed=seed)
    y_off, y_on = hv.label_counts(logr.size, starts, 35, labels == hv.regime_on_cluster(segs, labels))
    wk_bear = np.where(y_off + y_on > 0, (y_on > y_off).astype(float), np.nan)
    month = r.index.to_period("M")
    ib = dm.bear_indicator(dm.monthly_from_daily(pd.DataFrame({"mkt": r.to_numpy(), "month": month}), ("mkt",)).mkt)
    dm_bear = ib.reindex(month).to_numpy()
    signals = {"Shu JM": res.bear["JM"].reindex(r.index).to_numpy(), "Shu HMM": res.bear["HMM"].reindex(r.index).to_numpy(),
               "Nystrup adaptive HMM": ny_bear, "Horvath WK-means (a posteriori)": wk_bear, "D&M bear indicator": dm_bear,
               "Perfect hindsight": truth.astype(float)}
    if oracle is not None:
        signals["Oracle filter"] = oracle.astype(float)
    test = r.index.isin(res.returns.index)
    rows = []
    for k, s in signals.items():
        strat, _ = zero_one(r.to_numpy(), np.zeros(r.size), s, delay=1, cost=0.001)
        st = s[test]
        ok = ~np.isnan(st)
        rows.append({"generator": name, "seed": seed, "detector": k, "coverage": ok.mean(),
                     "bal.acc": balanced_accuracy(st[ok], truth[test][ok]),
                     "shifts/yr": shifts_per_year(st[ok]), "latency": detection_latency(st[ok], truth[test][ok]),
                     "Sharpe": sharpe(strat[test])})
    rows.append({"generator": name, "seed": seed, "detector": "Buy and hold", "Sharpe": sharpe(r.to_numpy()[test]),
                 "shifts/yr": 0.0, "truth shifts/yr": shifts_per_year(truth[test]), "test years": test.sum() / 252})
    return rows


def results():
    rows = []
    for g in GENERATORS:
        for s in range(SEEDS):
            rows += cached(f"cross_{g.replace(' ', '_').replace('(', '').replace(')', '')}_{s}",
                           lambda g=g, s=s: run_path(g, s), DEPS)
    return pd.DataFrame(rows)


def criteria(df):
    errors = df[df.detector.isin(DETECTORS) & (df.coverage < 0.99)]
    out = {"26 every detector produced a signal on >= 99% of every test window": (errors.empty,
                                                                                  f"{len(errors)} short paths")}
    for g in GENERATORS:
        d = df[df.generator == g]
        if "Oracle filter" not in set(d.detector):
            continue
        m = d.groupby("detector")["bal.acc"].mean()
        online = m.drop(["Oracle filter", "Perfect hindsight", "Horvath WK-means (a posteriori)"], errors="ignore")
        out[f"27 {g}: no online detector beats the oracle by > 0.02"] = (
            (online <= m["Oracle filter"] + 0.02).all(), f"best online {online.max():.3f} vs oracle {m['Oracle filter']:.3f}")
    return out


def main():
    df = results()
    crit = criteria(df)
    intro = (f"{SEEDS} seeds x {YEARS} years of daily returns per generator; every detector run as its module "
             "implements the paper (SPEC: cross-generator study). Scores are over the Shu out-of-sample test window of "
             "each path (about 9 years). 0/1 strategy: bear -> cash, 1-day delay, 10 bp one-way, rf = 0. WK-means "
             "labels are a posteriori (the clustering sees the whole path), every other detector is online.")
    lines = ["# Cross-generator study: every detector on every generator", "", intro, "",
             "## Acceptance criteria (SPEC 26-27)", "", "| Criterion | Result | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {verdict(ok)} | {txt} |" for k, (ok, txt) in crit.items()]
    order = DETECTORS + ["Oracle filter", "Perfect hindsight", "Buy and hold"]
    summary = []
    for g in GENERATORS:
        d = df[df.generator == g]
        bh = d[d.detector == "Buy and hold"]
        t = d.groupby("detector")[["bal.acc", "shifts/yr", "latency", "Sharpe"]].agg(["mean", "std"])
        t = t.reindex([o for o in order if o in t.index])
        cells = t.apply(lambda row: pd.Series({c: f"{row[(c, 'mean')]:.2f} ({row[(c, 'std')]:.2f})"
                                               for c in ("bal.acc", "shifts/yr", "latency", "Sharpe")}), axis=1)
        cells.loc["Buy and hold", ["bal.acc", "latency"]] = "-"
        head = (f"True shifts/yr {bh['truth shifts/yr'].mean():.2f}; test window {bh['test years'].mean():.1f} years. "
                "Mean (sd) over seeds.")
        lines += ["", f"## {g}", "", head, "", cells.to_markdown()]
        m = d.groupby("detector")[["bal.acc", "Sharpe"]].mean()
        det = m.loc[[x for x in DETECTORS if x in m.index]]
        summary.append({"generator": g, "best accuracy": det["bal.acc"].idxmax(),
                        "best Sharpe": det["Sharpe"].idxmax(),
                        "B&H Sharpe": m.loc["Buy and hold", "Sharpe"]})
    lines += ["", "## Summary", "", pd.DataFrame(summary).to_markdown(index=False, floatfmt=".2f"), ""]
    (ROOT / "reports" / "cross_generators.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return crit


if __name__ == "__main__":
    sys.exit(0 if all(ok for ok, _ in main().values()) else 1)
