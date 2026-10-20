"""Ground truth for Nystrup, Madsen & Lindström (2018): convergence and tracking of the adaptive HMM estimator,
filtered-state accuracy against the true-parameter filter, and MPC vs buy-and-hold on Markov-switching paths."""

import sys

import numpy as np
import pandas as pd

from regime_switch import hmm
from regime_switch import nystrup2018 as ny
from regime_switch.metrics import balanced_accuracy, sharpe
from regime_switch.sim import drifting_hmm, markov_switching_gaussian
from scripts.common import ROOT, cached, verdict

SEEDS, YEARS, DRIFT_YEARS = 20, 10, 16
NAMES = ["mu1", "mu2", "sd1", "sd2", "g11", "g22"]
DEPS = ["regime_switch/nystrup2018.py", "regime_switch/hmm.py", "regime_switch/metrics.py", "regime_switch/sim.py",
        "scripts/sim_nystrup2018.py"]


def ordered(theta):
    """Parameters with state 1 = the lower-variance state (the recursion does not enforce an order)."""
    mu, var, g = ny.unpack(theta)
    o = np.argsort(var)
    return np.r_[mu[o], np.sqrt(var[o]), g[o]], o


def oracle_filter(path):
    G = np.array([[path.p_stay[0], 1 - path.p_stay[0]], [1 - path.p_stay[1], path.p_stay[1]]])
    pi = np.array([1 - G[1, 1], 1 - G[0, 0]]) / (2 - G[0, 0] - G[1, 1])
    return hmm.forward(path.log_returns, path.mu, path.sigma**2, G, pi)[0][:, 1] > 0.5


def run_seed(seed):
    print("seed", seed, flush=True)
    path = markov_switching_gaussian(252 * YEARS, seed)
    x = path.log_returns
    th = ny.adaptive_estimates(x)
    est, _ = ordered(th[-1])
    alpha = ny.filtered_probs(x, th)
    hi = np.array([ordered(t)[1][1] for t in th[259:]])
    pred = alpha[np.arange(259, x.size), hi] > 0.5
    truth = path.states[259:]
    row = {"seed": seed, **dict(zip(NAMES, est)),
           "adaptive bal.acc": balanced_accuracy(pred, truth),
           "oracle bal.acc": balanced_accuracy(oracle_filter(path)[259:], truth)}
    means = np.full((x.size, 100), np.nan)
    variances = np.full((x.size, 100), np.nan)
    for t in range(259, x.size):
        means[t], variances[t] = ny.forecast_moments(alpha[t], th[t], 100)
    simple = np.expm1(x)
    for gamma, rho in ((0.0, 0.0), (0.0, 0.02)):
        r, _, turn, fails = ny.simulate_mpc(simple, means, variances, 259, gamma, rho, 0.001, 100, 0)
        row[f"MPC(0,{rho:g}) Sharpe"] = sharpe(r)
        row[f"MPC(0,{rho:g}) turnover"] = turn.sum() / (r.size / 252)
        row[f"MPC(0,{rho:g}) failures"] = fails
    row["B&H Sharpe"] = sharpe(simple[260:])
    d = drifting_hmm(252 * DRIFT_YEARS, seed)
    thd = ny.adaptive_estimates(d.log_returns)
    est_path = np.array([ordered(t)[0] for t in thd[259:]])
    true_mu, true_sd = d.mu[259:], d.sigma[259:]
    mle = hmm.fit(d.log_returns, n_init=10, seed=seed)
    o = np.argsort(mle["var"])
    row["drift MAE adaptive mu"] = np.abs(est_path[:, :2] - true_mu).mean()
    row["drift MAE static mu"] = np.abs(mle["mu"][o][None, :] - true_mu).mean()
    row["drift MAE adaptive sd"] = np.abs(est_path[:, 2:4] - true_sd).mean()
    row["drift MAE static sd"] = np.abs(np.sqrt(mle["var"][o])[None, :] - true_sd).mean()
    return row


def results():
    return cached("sim_nystrup2018", lambda: pd.DataFrame([run_seed(s) for s in range(SEEDS)]), DEPS)


TRUTH = {"mu1": 0.0006, "mu2": -0.0008, "sd1": 0.007, "sd2": 0.018, "g11": 0.996, "g22": 0.985}


def criteria(df):
    m = df.mean(numeric_only=True)
    out = {}
    for k, v in TRUTH.items():
        bias, sd = df[k].mean() - v, df[k].std(ddof=1)
        out[f"15a {k}: |mean - truth| <= 2 sd across seeds"] = (abs(bias) <= 2 * sd,
                                                                f"mean {df[k].mean():.5g}, truth {v}, sd {sd:.3g}")
    out["15b drifting path: adaptive MAE(mu) < static MLE"] = (
        m["drift MAE adaptive mu"] < m["drift MAE static mu"],
        f"{m['drift MAE adaptive mu']:.2e} vs {m['drift MAE static mu']:.2e}")
    out["15b drifting path: adaptive MAE(sd) < static MLE"] = (
        m["drift MAE adaptive sd"] < m["drift MAE static sd"],
        f"{m['drift MAE adaptive sd']:.2e} vs {m['drift MAE static sd']:.2e}")
    out["15c filtered-state bal.acc >= 90% of oracle filter"] = (
        m["adaptive bal.acc"] >= 0.9 * m["oracle bal.acc"], f"{m['adaptive bal.acc']:.3f} vs 0.9 x {m['oracle bal.acc']:.3f}")
    out["16a MPC(0,0) Sharpe >= B&H Sharpe (mean)"] = (m["MPC(0,0) Sharpe"] >= m["B&H Sharpe"],
                                                       f"{m['MPC(0,0) Sharpe']:.3f} vs {m['B&H Sharpe']:.3f}")
    out["16b MPC(0,0.02) turnover < MPC(0,0) turnover"] = (
        m["MPC(0,0.02) turnover"] < m["MPC(0,0) turnover"],
        f"{m['MPC(0,0.02) turnover']:.2f} vs {m['MPC(0,0) turnover']:.2f}")
    return out


def main():
    df = results()
    crit = criteria(df)
    intro = (f"{SEEDS} seeds. Parameter and MPC checks on `markov_switching_gaussian` ({YEARS} years, benchmark "
             f"parameters); tracking on `drifting_hmm` ({DRIFT_YEARS} years, sinusoidal means and vols). Estimator "
             "settings as in the paper (N_eff = 260, A = 1/N_eff, first 260 days for initialisation) plus "
             "DEVIATIONS.md#N3-N4. MPC: K = 100, kappa = 0.001, no delay. Parameters at the last day are reported "
             "with state 1 = the lower-variance state.")
    lines = ["# Nystrup, Madsen & Lindström (2018): simulation ground truth", "", intro, "",
             "## Acceptance criteria (SPEC 15-16)", "", "| Criterion | Result | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {verdict(ok)} | {txt} |" for k, (ok, txt) in crit.items()]
    cols = [c for c in df.columns if c != "seed"]
    summ = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(), "min": df[cols].min(), "max": df[cols].max()})
    lines += ["", "## Per-seed summary", "", summ.to_markdown(floatfmt=".4g"), ""]
    (ROOT / "reports" / "sim_nystrup2018.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))
    return crit


if __name__ == "__main__":
    sys.exit(0 if all(ok for ok, _ in main().values()) else 1)
