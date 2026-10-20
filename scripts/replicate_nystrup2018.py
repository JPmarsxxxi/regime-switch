"""Nystrup, Madsen & Lindström (2018) on free data: S&P 500 for Tables 1-2 (MSCI World has no free daily history),
S&P 500 / DAX / FTSE for Table 3. 1984-2015, first two years for initialisation."""

import numpy as np
import pandas as pd

from regime_switch import data
from regime_switch import nystrup2018 as ny
from regime_switch.metrics import nystrup_metrics
from scripts.common import ROOT, cached

END = "2015-12-31"
INDICES = {"S&P500": ("^GSPC", "1984-01-01"), "DAX": ("^GDAXI", "1988-01-01"), "FTSE": ("^FTSE", "1984-01-01")}
ROWS = ["Annualized return", "Standard deviation", "Sharpe ratio", "Maximum drawdown", "Calmar ratio",
        "Annual turnover"]
PAPER_T1 = {"MPC(g=0, rho=0)": [0.076, 0.12, 0.65, 0.26, 0.29, 4.61], "MPC(g=2, rho=0)": [0.062, 0.10, 0.61, 0.21, 0.29, 5.30],
            "MPC(g=0, rho=0.02)": [0.070, 0.11, 0.67, 0.23, 0.31, 1.17]}
PAPER_T2 = {"MPC(g=0, rho=0)": [0.065, 0.12, 0.56, 0.26, 0.25, 4.16], "MPC(g=0, rho=0.02)": [0.069, 0.11, 0.63, 0.25, 0.28, 1.17],
            "Index": [0.056, 0.18, 0.30, 0.57, 0.10, 0.00], "SAA": [0.041, 0.13, 0.32, 0.44, 0.09, 0.09]}
PAPER_T3 = {"S&P500": ([0.110, 0.14, 0.79, 0.36, 0.30, 3.29], [0.108, 0.16, 0.68, 0.55, 0.20, 0.00]),
            "DAX": ([0.066, 0.16, 0.41, 0.39, 0.17, 5.89], [0.073, 0.22, 0.33, 0.73, 0.10, 0.00]),
            "FTSE": ([0.051, 0.14, 0.36, 0.36, 0.14, 4.59], [0.078, 0.15, 0.54, 0.48, 0.16, 0.00])}
DEPS = ["regime_switch/nystrup2018.py", "regime_switch/hmm.py", "regime_switch/metrics.py", "regime_switch/data.py",
        "scripts/replicate_nystrup2018.py"]


def inputs(name):
    ticker, start = INDICES[name]
    px = data.yahoo_close(ticker)[start:END]
    return np.log(px).diff().dropna()


def saa(simple, weight, kappa=0.001):
    """Monthly rebalancing to a fixed stock weight, with proportional costs on the rebalancing trades."""
    month = simple.index.to_period("M")
    w = weight
    rets, turn = [], []
    for i, r in enumerate(simple.to_numpy()):
        trade = 0.0
        if i == 0 or month[i] != month[i - 1]:
            trade = abs(weight - w)
            w = weight
        v = w * (1 + r) + (1 - w) - kappa * trade
        rets.append(v - 1)
        turn.append(trade)
        w = w * (1 + r) / v
    return np.array(rets), np.array(turn)


def run_index(name, configs, delays):
    x = inputs(name)
    test_start = x.index[0] + pd.DateOffset(years=2)
    return ny.run(x, test_start=test_start, configs=configs, delays=delays)


def results():
    sp = cached("nystrup_sp500", lambda: run_index("S&P500", ((0.0, 0.0), (2.0, 0.0), (0.0, 0.02)), (0, 1)), DEPS)
    others = {n: cached(f"nystrup_{n}", lambda n=n: run_index(n, ((0.0, 0.0),), (0,)), DEPS) for n in ("DAX", "FTSE")}
    return sp, others


def fmt(df):
    return df.loc[ROWS].map(lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}")


def main():
    sp, others = results()
    th, out, met, *_ = sp
    intro = ("Data: Yahoo Finance closes (`^GSPC`, `^GDAXI`, `^FTSE`) to 2015-12-31; the first two years initialise "
             "the estimator (the paper's choice for these indices), testing starts after them. The paper's Tables 1-2 "
             "are for MSCI World 1997-2015, which has no free daily history; S&P 500 stands in and the paper's MSCI "
             "numbers are shown for reference only. Costs kappa = 0.001, K = 100, long-only. See DEVIATIONS.md#N1-N5.")
    lines = ["# Nystrup, Madsen & Lindström (2018): replication on free data", "", intro, ""]
    t1 = met[["MPC(g=0, rho=0, delay=0)", "MPC(g=2, rho=0, delay=0)", "MPC(g=0, rho=0.02, delay=0)"]]
    t1.columns = list(PAPER_T1)
    lines += ["## Table 1: MPC with and without risk and trading aversion (no delay)", "", "Ours (S&P 500):", "",
              fmt(t1).to_markdown(), "", "Paper (MSCI World):", "",
              pd.DataFrame(PAPER_T1, index=ROWS).to_markdown(floatfmt=".3f"), ""]
    simple = np.expm1(inputs("S&P500"))
    test = simple.index >= simple.index[0] + pd.DateOffset(years=2)
    w_bar = out[(0.0, 0.0, 0)].weights.mean()
    saa_r, saa_t = saa(simple[test], w_bar)
    t2 = met[["MPC(g=0, rho=0, delay=1)", "MPC(g=0, rho=0.02, delay=1)", "Buy and hold"]].copy()
    t2.columns = ["MPC(g=0, rho=0)", "MPC(g=0, rho=0.02)", "Index"]
    t2["SAA"] = pd.Series(nystrup_metrics(saa_r, saa_t))
    note = (f"Ours (S&P 500; SAA rebalanced monthly to {100 * w_bar:.0f}% stocks, MPC(0,0)'s average weight; "
            "RBAA not reproduced, DEVIATIONS.md#N5):")
    lines += ["## Table 2: one-day delay in allocation changes", "", note, "", fmt(t2).to_markdown(), "",
              "Paper (MSCI World):", "",
              pd.DataFrame(PAPER_T2, index=ROWS).to_markdown(floatfmt=".3f"), ""]
    t3, claims = {}, []
    for name in INDICES:
        m = met if name == "S&P500" else others[name][2]
        mpc, bh = m["MPC(g=0, rho=0, delay=0)"], m["Buy and hold"]
        t3[name] = [f"{a:.3f} ({b:.3f})" for a, b in zip(mpc[ROWS], bh[ROWS])]
        claims += [(f"{name}: MPC SD < B&H SD", mpc["Standard deviation"] < bh["Standard deviation"],
                    f"{mpc['Standard deviation']:.3f} vs {bh['Standard deviation']:.3f}"),
                   (f"{name}: MPC MDD < B&H MDD", mpc["Maximum drawdown"] < bh["Maximum drawdown"],
                    f"{mpc['Maximum drawdown']:.3f} vs {bh['Maximum drawdown']:.3f}")]
    paper3 = {n: [f"{a:.3f} ({b:.3f})" for a, b in zip(*PAPER_T3[n])] for n in INDICES}
    lines += ["## Table 3: MPC(g=0, rho=0), no delay, vs buy-and-hold (in parentheses)", "", "Ours:", "",
              pd.DataFrame(t3, index=ROWS).to_markdown(), "", "Paper:", "",
              pd.DataFrame(paper3, index=ROWS).to_markdown(), ""]
    beats = [n for n in INDICES if (met if n == "S&P500" else others[n][2]).loc["Sharpe ratio"].pipe(
        lambda s: s["MPC(g=0, rho=0, delay=0)"] > s["Buy and hold"])]
    claims += [
        ("Trading penalty lowers turnover (S&P 500, no delay)",
         t1.loc["Annual turnover", "MPC(g=0, rho=0.02)"] < t1.loc["Annual turnover", "MPC(g=0, rho=0)"],
         f"{t1.loc['Annual turnover', 'MPC(g=0, rho=0.02)']:.2f} vs {t1.loc['Annual turnover', 'MPC(g=0, rho=0)']:.2f}"),
        ("With a one-day delay the trading penalty raises the annualised return",
         t2.loc["Annualized return", "MPC(g=0, rho=0.02)"] > t2.loc["Annualized return", "MPC(g=0, rho=0)"],
         f"{t2.loc['Annualized return', 'MPC(g=0, rho=0.02)']:.3f} vs {t2.loc['Annualized return', 'MPC(g=0, rho=0)']:.3f}"),
        ("MPC Sharpe > B&H Sharpe on most indices (>= 2 of 3)", len(beats) >= 2, f"beats B&H on {beats or 'none'}"),
    ]
    fails = {k: v.failures for k, v in out.items()} | {f"{n} {k}": v.failures for n in others for k, v in
                                                        others[n][1].items()}
    lines += ["## Headline claims", "", "| Claim | Verdict | Numbers |", "|---|---|---|"]
    lines += [f"| {c} | {'REPRODUCED' if ok else 'NOT REPRODUCED'} | {n} |" for c, ok, n in claims]
    box = {}
    for n in INDICES:
        t = (sp[0] if n == "S&P500" else others[n][0]).dropna()
        raw = np.column_stack([t.mu1, t.mu2, np.log(t.var1), np.log(t.var2), np.log(t.gamma11 / (1 - t.gamma11)),
                               np.log(t.gamma22 / (1 - t.gamma22))])
        box[n] = int((np.isclose(raw, ny.LO) | np.isclose(raw, ny.HI)).any(axis=1).sum())
    lines += ["", f"MPC solver failures (held the portfolio that day): {fails}.",
              f"Days with a parameter on the safety box (DEVIATIONS.md#N4): {box}.", "",
              "## Estimated HMM parameters (S&P 500, year ends; state 1 = lower variance)", ""]
    th = th.dropna()
    swap = th.var1 > th.var2
    tidy = pd.DataFrame({"mu1": th.mu1.where(~swap, th.mu2), "mu2": th.mu2.where(~swap, th.mu1),
                         "sd1": np.sqrt(th.var1.where(~swap, th.var2)), "sd2": np.sqrt(th.var2.where(~swap, th.var1)),
                         "gamma11": th.gamma11.where(~swap, th.gamma22), "gamma22": th.gamma22.where(~swap, th.gamma11)})
    lines += [tidy.resample("YE").last().to_markdown(floatfmt=".5f"), ""]
    (ROOT / "reports" / "replicate_nystrup2018.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
