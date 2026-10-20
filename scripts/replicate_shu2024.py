"""Shu, Yu & Mulvey (2024) on free data: S&P 500, DAX, Nikkei 225 (Tables 3, 4 and 5)."""

import numpy as np
import pandas as pd

from regime_switch import data, shu2024
from regime_switch.metrics import shifts_per_year
from scripts.common import ROOT, cached

INDICES = {"S&P 500": ("^GSPC", "us"), "DAX": ("^GDAXI", "de"), "Nikkei 225": ("^N225", "jp")}
START, END, TEST = "1970-01-01", "2023-12-31", "1990-01-01"
PAPER_T3 = {"HMM": {0: 8.5, 2: 6.6, 4: 4.9, 8: 3.2, 20: 2.0},
            "JM": {0.0: 9.7, 5.0: 2.7, 15.0: 1.7, 35.0: 0.8, 70.0: 0.5, 150.0: 0.4}}
PAPER_T4 = {
    "S&P 500": {"B&H": [10.2, 18.2, 0.48, -55.2, 0.16, -2.7, 0, 100], "HMM": [8.5, 11.3, 0.54, -28.9, 0.21, -1.8, 141, 72],
                "JM": [11.2, 13.1, 0.68, -26.6, 0.33, -2.0, 44, 80]},
    "DAX": {"B&H": [6.8, 22.1, 0.30, -72.7, 0.09, -3.3, 0, 100], "HMM": [6.4, 14.0, 0.35, -40.5, 0.12, -2.2, 246, 73],
            "JM": [8.6, 16.4, 0.44, -39.4, 0.18, -2.5, 170, 84]},
    "Nikkei 225": {"B&H": [0.8, 23.4, 0.12, -79.1, 0.04, -3.4, 0, 100], "HMM": [2.5, 16.0, 0.19, -48.6, 0.06, -2.5, 290, 68],
                   "JM": [4.7, 17.1, 0.31, -45.3, 0.12, -2.6, 72, 75]},
}
PAPER_T5 = {  # Sharpe by delay 1, 5, 10
    "S&P 500": {"B&H": 0.48, "HMM": (0.54, 0.55, 0.51), "JM": (0.68, 0.71, 0.70)},
    "DAX": {"B&H": 0.30, "HMM": (0.35, 0.25, 0.12), "JM": (0.44, 0.38, 0.29)},
    "Nikkei 225": {"B&H": 0.12, "HMM": (0.19, 0.11, 0.07), "JM": (0.31, 0.27, 0.24)},
}
ROWS = ["Return", "Volatility", "Sharpe", "MDD", "Calmar", "ES0.05", "Turnover", "Leverage"]
DEPS = ["regime_switch/shu2024.py", "regime_switch/jump.py", "regime_switch/hmm.py", "regime_switch/backtest.py",
        "regime_switch/metrics.py", "regime_switch/data.py"]


def risk_free(country):
    """Annualised 3-month rate in percent -> daily return. See DEVIATIONS.md#S1."""
    if country == "us":
        y = data.fred("DTB3")
    elif country == "de":
        y = data.fred("IR3TIB01DEM156N")
    else:
        disc, call, ib = data.fred("INTDSRJPM193N"), data.fred("IRSTCI01JPM156N"), data.fred("IR3TIB01JPM156N")
        y = pd.concat([disc[: call.index[0] - pd.Timedelta(days=1)], call[: ib.index[0] - pd.Timedelta(days=1)], ib])
    return y / 100 / 252


def index_inputs(ticker, country):
    px = data.yahoo_close(ticker)[START:END]
    rf = risk_free(country).reindex(px.index.union(risk_free(country).index)).ffill().reindex(px.index)
    return px, rf


def candidates(name):
    ticker, country = INDICES[name]
    px, rf = index_inputs(ticker, country)

    def raw():
        r = px.pct_change().dropna()
        return shu2024.hmm_online_raw(np.log1p(r), shu2024.Config())

    raw_hmm = cached(f"shu_hmm_raw_{ticker.strip('^')}", raw, DEPS)
    return shu2024.candidates(px, rf, shu2024.Config(), hmm_raw=raw_hmm)


def fmt_metrics(m):
    pct = {"Return", "Volatility", "MDD", "ES0.05", "Turnover", "Leverage"}
    return {k: (f"{100 * v:.1f}%" if k in pct else f"{v:.2f}") for k, v in m.items()}


def main():
    intro = ("Data: Yahoo Finance closes (`^GSPC`, `^GDAXI`, `^N225`), 1970-2023; FRED 3-month rates. See "
             "DEVIATIONS.md#S1-S5 for every difference from the paper (notably: S&P 500 and Nikkei are price indices, "
             "the DAX history starts in 1988 so its test period starts later).")
    lines = ["# Shu, Yu & Mulvey (2024): replication on free data", "", intro, ""]
    claims = []
    for name in INDICES:
        print(name, flush=True)
        r, rf, cands, _ = candidates(name)
        start = max(pd.Timestamp(TEST), shu2024.default_test_start(cands, shu2024.Config()))
        res = {d: shu2024.evaluate(r, rf, cands, shu2024.Config(delay=d), start) for d in (1, 5, 10)}
        base = res[1]
        lines += [f"## {name}", "", f"Test period {base.returns.index[0].date()} to {base.returns.index[-1].date()}.", ""]
        if name == "S&P 500":
            span = slice("1982-01-01", END)
            t3 = {"HMM": {k: shifts_per_year(cands["HMM"][k][span].dropna().to_numpy()) for k in PAPER_T3["HMM"]},
                  "JM": {lam: shifts_per_year(cands["JM"][lam][span].dropna().to_numpy()) for lam in PAPER_T3["JM"]}}
            rows = [{"model": m, "param": p, "ours": f"{t3[m][p]:.1f}", "paper": PAPER_T3[m][p]} for m in t3
                    for p in t3[m]]
            lines += ["### Table 3: shifts per year in the online regimes (fixed hyperparameters, 1982-2023)", "",
                      pd.DataFrame(rows).to_markdown(index=False), ""]
        ours = {k: fmt_metrics(base.metrics[k]) for k in base.metrics.columns}
        paper = {k: dict(zip(ROWS, [f"{v}%" if r in ("Return", "Volatility", "MDD", "ES0.05", "Turnover", "Leverage")
                                    else f"{v:.2f}" for r, v in zip(ROWS, vals)]))
                 for k, vals in PAPER_T4[name].items()}
        t4 = pd.DataFrame({f"{k} ours": ours[k] for k in ours} | {f"{k} paper": paper[k] for k in paper})
        t4 = t4[[f"{k} {w}" for k in ("B&H", "HMM", "JM") for w in ("ours", "paper")]]
        lines += ["### Table 4 (delay 1, 10 bp)", "", t4.to_markdown(), ""]
        t5 = {f"delay {d}": {m: res[d].metrics.loc["Sharpe", m] for m in ("B&H", "HMM", "JM")} for d in (1, 5, 10)}
        p5 = PAPER_T5[name]
        t5_df = pd.DataFrame(t5).map(lambda v: f"{v:.2f}")
        t5_df["paper (1 / 5 / 10)"] = [f"{p5['B&H']:.2f}", " / ".join(f"{v:.2f}" for v in p5["HMM"]),
                                       " / ".join(f"{v:.2f}" for v in p5["JM"])]
        lines += ["### Table 5: Sharpe ratio by trading delay", "", t5_df.to_markdown(), ""]
        bear_pct = 100 * base.bear["JM"].mean()
        n_shifts = int((base.bear["JM"].diff().abs() > 0).sum())
        lam = base.choices["JM"]["choice"].astype(float)
        dist = lam.value_counts(normalize=True).round(2).to_dict()
        note = (f"JM bear share {bear_pct:.1f}%, {n_shifts} regime shifts; selected lambda: median "
                f"{lam.median():g}, distribution {dist}.")
        lines += [note, ""]
        m = base.metrics
        claims += [
            (f"{name}: JM Sharpe > B&H Sharpe", m.loc["Sharpe", "JM"] > m.loc["Sharpe", "B&H"],
             f"{m.loc['Sharpe', 'JM']:.2f} vs {m.loc['Sharpe', 'B&H']:.2f}"),
            (f"{name}: JM Sharpe > HMM Sharpe", m.loc["Sharpe", "JM"] > m.loc["Sharpe", "HMM"],
             f"{m.loc['Sharpe', 'JM']:.2f} vs {m.loc['Sharpe', 'HMM']:.2f}"),
            (f"{name}: JM MDD smaller than B&H MDD", m.loc["MDD", "JM"] > m.loc["MDD", "B&H"],
             f"{100 * m.loc['MDD', 'JM']:.1f}% vs {100 * m.loc['MDD', 'B&H']:.1f}%"),
            (f"{name}: JM turnover < HMM turnover", m.loc["Turnover", "JM"] < m.loc["Turnover", "HMM"],
             f"{100 * m.loc['Turnover', 'JM']:.0f}% vs {100 * m.loc['Turnover', 'HMM']:.0f}%"),
        ]
    lines += ["## Headline claims", "", "| Claim | Verdict | Numbers |", "|---|---|---|"]
    lines += [f"| {c} | {'REPRODUCED' if ok else 'NOT REPRODUCED'} | {n} |" for c, ok, n in claims]
    (ROOT / "reports" / "replicate_shu2024.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
