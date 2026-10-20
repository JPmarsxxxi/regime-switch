"""Replication of Horvath, Issa & Muguruza (2021) Sec. 3.3 (Tables 3 and 4): 50 synthetic gBm and Merton paths each,
WK-means vs MK-means vs Gaussian HMM, plus centroid moments vs the true measures and MMD self-similarity."""

import sys

import numpy as np
import pandas as pd

from regime_switch import horvath2021 as hv
from regime_switch.sim import GBM_BEAR, GBM_BULL, HOURS_PER_YEAR, MJD_BEAR, MJD_BULL, merton_moments
from scripts.common import ROOT, cached, verdict

RUNS = 50
PAPER = {
    "gbm": {"Wasserstein": (90.60, 87.24, 91.72), "Moment": (93.23, 74.83, 99.38), "HMM": (58.16, 41.51, 63.72)},
    "merton": {"Wasserstein": (91.28, 86.87, 92.76), "Moment": (66.64, 27.25, 79.79), "HMM": (75.05, 0.66, 99.87)},
}
DEPS = ["regime_switch/horvath2021.py", "regime_switch/hmm.py", "regime_switch/sim.py", "scripts/sim_horvath2021.py"]


def study(model):
    rows, mmd = [], []
    for seed in range(RUNS):
        print(model, seed, flush=True)
        o = hv.synthetic_run(model, seed)
        for m in ("Wasserstein", "Moment", "HMM", "HMM (hmmlearn defaults)"):
            rows.append({"seed": seed, "method": m, **{k: v for k, v in o[m].items() if k != "labels"}})
        if seed < 5:
            segs = o["segments"]
            for m in ("Wasserstein", "Moment"):
                lab = o[m]["labels"]
                c = [segs[lab == j] for j in range(2) if np.any(lab == j)]
                if len(c) == 2:
                    mmd.append({"seed": seed, "method": m,
                                "within C_off": hv.self_similarity(c[0], c[0], 2000, seed=seed),
                                "within C_on": hv.self_similarity(c[1], c[1], 2000, seed=seed),
                                "between": hv.self_similarity(c[0], c[1], 2000, seed=seed)})
    return pd.DataFrame(rows), pd.DataFrame(mmd)


def results():
    return {m: cached(f"sim_horvath2021_{m}", lambda m=m: study(m), DEPS) for m in ("gbm", "merton")}


def ci(s):
    return f"{100 * s.mean():.2f}% +- {100 * 1.96 * s.std(ddof=1) / np.sqrt(len(s)):.2f}%"


def acc_table(df, model):
    rows = []
    for m in ("Wasserstein", "Moment", "HMM", "HMM (hmmlearn defaults)"):
        d = df[df.method == m]
        p = PAPER[model].get(m)
        rows.append({"Algorithm": m, "Total": ci(d.TA), "Regime-on": ci(d.RONS), "Regime-off": ci(d.ROFS),
                     "Paper total / on / off": "-" if p is None else f"{p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}"})
    return pd.DataFrame(rows).to_markdown(index=False)


def centroid_table(df, model):
    dt = 1 / HOURS_PER_YEAR
    if model == "gbm":
        true = {"off": ((GBM_BULL[0] - GBM_BULL[1] ** 2 / 2) * dt, GBM_BULL[1] ** 2 * dt),
                "on": ((GBM_BEAR[0] - GBM_BEAR[1] ** 2 / 2) * dt, GBM_BEAR[1] ** 2 * dt)}  # eq. 35
    else:
        true = {"off": merton_moments(MJD_BULL), "on": merton_moments(MJD_BEAR)}  # eqs. 39-40
    rows = [{"Method": "True measure", "mean off": true["off"][0], "var off": true["off"][1],
             "mean on": true["on"][0], "var on": true["on"][1]}]
    for m in ("Wasserstein", "Moment"):
        d = df[df.method == m]
        rows.append({"Method": m, "mean off": d.centroid_mean_off.mean(), "var off": d.centroid_var_off.mean(),
                     "mean on": d.centroid_mean_on.mean(), "var on": d.centroid_var_on.mean()})
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3e")


def criteria(res):
    g, m = res["gbm"][0], res["merton"][0]

    def ta(df, meth):
        return 100 * df[df.method == meth].TA.mean()

    return {
        "17a WK total accuracy within 5pp of paper (gBm, 90.60%)": (abs(ta(g, "Wasserstein") - 90.60) <= 5,
                                                                    f"{ta(g, 'Wasserstein'):.2f}%"),
        "17b WK total accuracy within 5pp of paper (Merton, 91.28%)": (abs(ta(m, "Wasserstein") - 91.28) <= 5,
                                                                       f"{ta(m, 'Wasserstein'):.2f}%"),
        "17c Merton: WK > MK": (ta(m, "Wasserstein") > ta(m, "Moment"),
                                f"{ta(m, 'Wasserstein'):.2f}% vs {ta(m, 'Moment'):.2f}%"),
        "17d Merton: WK > HMM (maximum likelihood)": (ta(m, "Wasserstein") > ta(m, "HMM"),
                                                      f"{ta(m, 'Wasserstein'):.2f}% vs {ta(m, 'HMM'):.2f}%"),
        "17e Merton: WK > HMM (hmmlearn defaults)": (
            ta(m, "Wasserstein") > ta(m, "HMM (hmmlearn defaults)"),
            f"{ta(m, 'Wasserstein'):.2f}% vs {ta(m, 'HMM (hmmlearn defaults)'):.2f}%"),
    }


def main():
    res = results()
    crit = criteria(res)
    intro = (f"{RUNS} runs per model, (h1, h2) = (35, 28), k = 2, p = 1, hourly mesh 252 x 7, 20 years, 10 bear "
             "episodes of half a year. 95% CI = mean +- 1.96 sd / sqrt(runs). 'HMM' is a 2-state Gaussian HMM fitted "
             "by maximum likelihood (10 starts, EM to convergence, Viterbi); 'HMM (hmmlearn defaults)' is the same "
             "model through `hmmlearn.GaussianHMM()` with its default settings (see DEVIATIONS.md#H3).")
    lines = ["# Horvath, Issa & Muguruza (2021): synthetic replication (Tables 3 and 4)", "", intro, "",
             "## Acceptance criteria (SPEC 17)", "", "| Criterion | Result | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {verdict(ok)} | {txt} |" for k, (ok, txt) in crit.items()]
    for model, title in (("gbm", "Table 3: geometric Brownian motion"), ("merton", "Table 4: Merton jump diffusion")):
        df, mmd = res[model]
        cen = "Centroid moments vs the true measures (mean over runs; MK 'centroid' moments are those of its members):"
        sim = "MMD^2 self-similarity (Def. 1.9, Gaussian kernel sigma = 0.1, 2000 pairs, first 5 runs):"
        mmd_t = mmd.groupby("method").mean(numeric_only=True).drop(columns="seed").to_markdown(floatfmt=".3e")
        lines += ["", f"## {title}", "", acc_table(df, model), "", cen, "", centroid_table(df, model), "", sim, "",
                  mmd_t]
    (ROOT / "reports" / "sim_horvath2021.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    return crit


if __name__ == "__main__":
    sys.exit(0 if all(ok for ok, _ in main().values()) else 1)
