"""Ground truth for Shu, Yu & Mulvey (2024): JM and HMM online regimes and 0/1 strategies on Markov-switching paths
with known states, scored against the true-parameter forward filter (the best possible online classifier)."""

import sys

import numpy as np
import pandas as pd

from regime_switch import hmm, shu2024
from regime_switch.backtest import zero_one
from regime_switch.metrics import balanced_accuracy, detection_latency, sharpe, shifts_per_year
from regime_switch.sim import markov_switching_gaussian, markov_switching_t
from scripts.common import ROOT, cached, verdict

YEARS, SEEDS, T_SEEDS = 30, 20, 10


def oracle_states(path):
    G = np.array([[path.p_stay[0], 1 - path.p_stay[0]], [1 - path.p_stay[1], path.p_stay[1]]])
    pi = np.array([1 - G[1, 1], 1 - G[0, 0]]) / (2 - G[0, 0] - G[1, 1])
    a, _ = hmm.forward(path.log_returns, path.mu, path.sigma**2, G, pi)
    return (a[:, 1] > 0.5).astype(float)


def run_seed(seed, gen):
    print("seed", seed, gen.__name__, flush=True)
    path = gen(252 * YEARS, seed)
    idx = pd.bdate_range("1980-01-01", periods=path.log_returns.size)
    price = pd.Series(np.exp(np.cumsum(path.log_returns)), index=idx)
    rf = pd.Series(0.0, index=idx)
    res = shu2024.run(price, rf, shu2024.Config(seed=seed))
    test = res.returns.index
    truth = pd.Series(path.states, index=idx).iloc[1:].loc[test].to_numpy()
    orc = pd.Series(oracle_states(path), index=idx).iloc[1:]
    r = price.pct_change().dropna()
    orc_ret, _ = zero_one(r.to_numpy(), rf.reindex(r.index).to_numpy(), orc.to_numpy())
    orc_ret = pd.Series(orc_ret, index=r.index).loc[test]
    sig = {"JM": res.bear["JM"], "HMM": res.bear["HMM"],
           "JM lam=0": res.candidates["JM"][0.0].loc[test], "HMM k=0": res.candidates["HMM"][0].loc[test],
           "Oracle filter": orc.loc[test]}
    row = {"seed": seed, "truth shifts/yr": shifts_per_year(truth)}
    for k, s in sig.items():
        s = s.to_numpy()
        row[f"{k} bal.acc"] = balanced_accuracy(s, truth)
        row[f"{k} shifts/yr"] = shifts_per_year(s)
        row[f"{k} latency"] = detection_latency(s, truth)
    for k in ("B&H", "JM", "HMM"):
        row[f"{k} Sharpe"] = sharpe(res.returns[k])
        row[f"{k} turnover"] = res.metrics.loc["Turnover", k]
    row["Oracle Sharpe"] = sharpe(orc_ret)
    row["JM lambda (median)"] = float(np.median(res.choices["JM"]["choice"].astype(float)))
    row["HMM k (median)"] = float(np.median(res.choices["HMM"]["choice"].astype(float)))
    return row


def study(gen, seeds):
    return pd.DataFrame([run_seed(s, gen) for s in range(seeds)])


def criteria(df):
    m = df.mean(numeric_only=True)
    return {
        "13a JM bal.acc >= 90% of oracle filter": (m["JM bal.acc"] >= 0.9 * m["Oracle filter bal.acc"],
                                                   f"{m['JM bal.acc']:.3f} vs 0.9 x {m['Oracle filter bal.acc']:.3f}"),
        "13b JM shifts/yr < lambda=0 shifts/yr": (m["JM shifts/yr"] < m["JM lam=0 shifts/yr"],
                                                  f"{m['JM shifts/yr']:.2f} vs {m['JM lam=0 shifts/yr']:.2f}"),
        "13c JM 0/1 Sharpe >= B&H Sharpe": (m["JM Sharpe"] >= m["B&H Sharpe"],
                                            f"{m['JM Sharpe']:.3f} vs {m['B&H Sharpe']:.3f}"),
        "14a JM shifts/yr < HMM (k=0) shifts/yr": (m["JM shifts/yr"] < m["HMM k=0 shifts/yr"],
                                                   f"{m['JM shifts/yr']:.2f} vs {m['HMM k=0 shifts/yr']:.2f}"),
        "14b JM turnover < HMM turnover": (m["JM turnover"] < m["HMM turnover"],
                                           f"{m['JM turnover']:.3f} vs {m['HMM turnover']:.3f}"),
    }


DEPS = ["regime_switch/shu2024.py", "regime_switch/jump.py", "regime_switch/hmm.py", "regime_switch/backtest.py",
        "regime_switch/metrics.py", "regime_switch/sim.py", "scripts/sim_shu2024.py"]


def results():
    g = cached("sim_shu2024_gauss", lambda: study(markov_switching_gaussian, SEEDS), DEPS)
    t = cached("sim_shu2024_t", lambda: study(markov_switching_t, T_SEEDS), DEPS)
    return g, t


def summary(df):
    cols = [c for c in df.columns if c != "seed"]
    s = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(), "min": df[cols].min(), "max": df[cols].max()})
    return s.to_markdown(floatfmt=".3f")


def main():
    g, t = results()
    crit = criteria(g)
    intro = (f"Paths: `markov_switching_gaussian` (benchmark parameters in `sim.py`), {SEEDS} seeds x {YEARS} years; "
             f"robustness: `markov_switching_t` (df=4), {T_SEEDS} seeds. Full paper pipeline: 3000-day training, "
             "6-monthly JM refits, daily HMM refits, 8-year CV, 1-day delay, 10 bp costs, rf = 0. Scores are over the "
             "out-of-sample test period only. 'Oracle filter' is the forward filter run with the true parameters.")
    lines = ["# Shu, Yu & Mulvey (2024): simulation ground truth", "", intro, "",
             "## Acceptance criteria (SPEC 13-14)", "", "| Criterion | Result | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {verdict(ok)} | {txt} |" for k, (ok, txt) in crit.items()]
    js, hs = g["JM Sharpe"].mean(), g["HMM Sharpe"].mean()
    who = "JM higher" if js > hs else "HMM higher"
    lines += ["", f"Reported (not a criterion): JM Sharpe {js:.3f} vs HMM Sharpe {hs:.3f} -> {who}.", "",
              "## Gaussian paths", "", summary(g), "", "## Student-t paths", "", summary(t), ""]
    (ROOT / "reports" / "sim_shu2024.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    return crit


if __name__ == "__main__":
    sys.exit(0 if all(ok for ok, _ in main().values()) else 1)
