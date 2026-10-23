"""WK-means with an HMM head (SPEC: Extension, criteria 28-30): both variants on the cross-generator paths and on
point-in-time real data, next to the existing detectors."""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from regime_switch import shu2024, wk_hmm
from regime_switch.backtest import zero_one
from regime_switch.metrics import balanced_accuracy, detection_latency, sharpe, shifts_per_year, shu_metrics
from scripts import cross_generators as cg
from scripts import replicate_shu2024 as rs
from scripts.common import ROOT, cached

DEPS = ["regime_switch/wk_hmm.py", "regime_switch/horvath2021.py", "regime_switch/shu2024.py", "regime_switch/sim.py",
        "regime_switch/backtest.py", "regime_switch/metrics.py", "scripts/wk_hmm_study.py"]
VARIANTS = {"A": "WK-HMM A (Wasserstein emissions)", "B": "WK-HMM B (empirical-density HMM)"}


def test_start(price, cfg):
    """Shu's default test start without the HMM refits: the jump model's first regime and the HMM's (k = 20 filter)."""
    r = price.pct_change().dropna()
    feat = shu2024.features(r)
    valid = feat.notna().all(axis=1).to_numpy().argmax() + cfg.burn_in
    fidx = feat.index[valid:]
    first_jm = shu2024.refit_dates(fidx, fidx[cfg.train - 1], cfg.refit_months)[0]
    first_hmm = r.index[cfg.train - 1 + max(cfg.ks) - 1]
    first = max(first_jm, first_hmm)
    return (first + pd.DateOffset(years=cfg.val_years, months=1)).to_period("M").to_timestamp()


def synth_path(args):
    name, seed = args
    x, truth, _ = cg.generate(name, seed)
    idx = pd.bdate_range("1980-01-01", periods=x.size + 1)
    price = pd.Series(np.exp(np.r_[0.0, np.cumsum(x)]), index=idx)
    r = price.pct_change().dropna()
    test = r.index >= test_start(price, shu2024.Config())
    rows = []
    for v, label in VARIANTS.items():
        s = (wk_hmm.online(pd.Series(np.log1p(r.to_numpy()), index=r.index), v, seed=seed) > 0.5).astype(float)
        s = s.where(s.index.isin(s.dropna().index)).to_numpy()
        strat, _ = zero_one(r.to_numpy(), np.zeros(r.size), s, delay=1, cost=0.001)
        st = s[test]
        ok = ~np.isnan(st)
        rows.append({"generator": name, "seed": seed, "detector": label, "coverage": ok.mean(),
                     "bal.acc": balanced_accuracy(st[ok], truth[test][ok]), "shifts/yr": shifts_per_year(st[ok]),
                     "latency": detection_latency(st[ok], truth[test][ok]), "Sharpe": sharpe(strat[test]),
                     "test years": test.sum() / 252})
    return rows


def synth_results():
    jobs = [(g, s) for g in cg.GENERATORS for s in range(cg.SEEDS)]

    def compute():
        with ProcessPoolExecutor(4) as ex:
            return [row for rows in ex.map(synth_path, jobs) for row in rows]

    return pd.DataFrame(cached("wk_hmm_synth", compute, DEPS))


def real_index(name):
    r, rf, cands, _ = rs.candidates(name)
    start = max(pd.Timestamp(rs.TEST), shu2024.default_test_start(cands, shu2024.Config()))
    base = shu2024.evaluate(r, rf, cands, shu2024.Config(), start)
    test = r.index >= start
    logr = pd.Series(np.log1p(r.to_numpy()), index=r.index)
    met = {k: base.metrics[k] for k in ("B&H", "HMM", "JM")}
    shifts = {"HMM": shifts_per_year(base.bear["HMM"].to_numpy()), "JM": shifts_per_year(base.bear["JM"].to_numpy())}
    for v, label in VARIANTS.items():
        s = (wk_hmm.online(logr, v) > 0.5).astype(float).where(lambda q: q.notna()).to_numpy()
        strat, w = zero_one(r.to_numpy(), rf.to_numpy(), s, delay=1, cost=0.001)
        met[label] = shu_metrics(strat[test], rf.to_numpy()[test], w[test])
        shifts[label] = shifts_per_year(s[test])
    return {"metrics": pd.DataFrame(met), "shifts": shifts, "start": start, "end": r.index[-1]}


def real_results():
    return {n: cached(f"wk_hmm_real_{rs.INDICES[n][0].strip('^')}", lambda n=n: real_index(n), DEPS + rs.DEPS)
            for n in rs.INDICES}


def main():
    new = synth_results()
    old = cg.results()
    years = old[old.detector == "Buy and hold"].set_index(["generator", "seed"])["test years"]
    check = new.groupby(["generator", "seed"])["test years"].first().round(6).equals(years.round(6))
    keep = ["Shu JM", "Shu HMM", "Nystrup adaptive HMM", "Horvath WK-means (a posteriori)", "Oracle filter",
            "Buy and hold"]
    df = pd.concat([old[old.detector.isin(keep)], new])
    order = list(VARIANTS.values()) + keep
    intro = ("Extension, not a paper replication (SPEC: Extension). Variant A scores each day's trailing 35-day return "
             "distribution by its Wasserstein distance to the two WK-means centroids; variant B is a 2-state HMM whose "
             "state densities are kernel density estimates initialised from WK-means and refined by EM. Both refit "
             "every 6 months on the trailing 3000 days and are online (P(regime-on) > 0.5 is the signal). Nothing "
             "was tuned.")
    lines = ["# WK-means with an HMM head", "", intro, "",
             f"Test windows identical to the cross-generator study: {'yes' if check else 'NO - check'}.", "",
             "## Synthetic generators (5 seeds x 30 years each; mean (sd) over seeds)"]
    summ = []
    for g in cg.GENERATORS:
        d = df[df.generator == g]
        t = d.groupby("detector")[["bal.acc", "shifts/yr", "latency", "Sharpe"]].agg(["mean", "std"])
        t = t.reindex([o for o in order if o in t.index])
        cells = t.apply(lambda row: pd.Series({c: f"{row[(c, 'mean')]:.2f} ({row[(c, 'std')]:.2f})"
                                               for c in ("bal.acc", "shifts/yr", "latency", "Sharpe")}), axis=1)
        cells.loc["Buy and hold", ["bal.acc", "latency"]] = "-"
        lines += ["", f"### {g}", "", cells.to_markdown()]
        m = d.groupby("detector")[["bal.acc", "Sharpe"]].mean()
        summ.append({"generator": g, **{f"{k} acc": m.loc[v, "bal.acc"] for k, v in VARIANTS.items()},
                     "Shu HMM acc": m.loc["Shu HMM", "bal.acc"],
                     "WK a posteriori acc": m.loc["Horvath WK-means (a posteriori)", "bal.acc"],
                     **{f"{k} Sharpe": m.loc[v, "Sharpe"] for k, v in VARIANTS.items()},
                     "Shu HMM Sharpe": m.loc["Shu HMM", "Sharpe"], "B&H Sharpe": m.loc["Buy and hold", "Sharpe"]})
    lines += ["", "### Summary", "", pd.DataFrame(summ).to_markdown(index=False, floatfmt=".2f")]
    lines += ["", "## Real data, point-in-time (0/1 strategy, 1-day delay, 10 bp, Shu test periods)"]
    for n, res in real_results().items():
        met = res["metrics"].copy()
        pct = ["Return", "Volatility", "MDD", "ES0.05", "Turnover", "Leverage"]
        met.loc["Shifts/yr"] = pd.Series(res["shifts"])
        fmt = pd.DataFrame({c: [f"{100 * v:.1f}%" if i in pct else f"{v:.2f}" for i, v in met[c].items()]
                            for c in met.columns}, index=met.index)
        lines += ["", f"### {n} ({res['start'].date()} to {res['end'].date()})", "", fmt.to_markdown()]
    (ROOT / "reports" / "wk_hmm.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
