"""Ground truth for Daniel & Moskowitz: the full pipeline on `momentum_panic` paths, where the true conditional mean
follows the paper's interaction model and the true variance follows GJR-GARCH with the Appendix D parameters."""

import sys

import numpy as np
import pandas as pd

from regime_switch import daniel_moskowitz as dm
from regime_switch.sim import DAYS_PER_MONTH, DM_GAMMA_INT, momentum_panic
from scripts.common import ROOT, cached, verdict

SEEDS, YEARS = 20, 80
DEPS = ["regime_switch/daniel_moskowitz.py", "regime_switch/sim.py", "scripts/sim_daniel_moskowitz.py"]


def frames(path):
    daily = pd.DataFrame({"wml": path.wml, "mkt": path.mkt, "month": path.month})
    g = daily.groupby("month")
    monthly = pd.DataFrame({"wml": g.wml.apply(lambda v: np.prod(1 + v) - 1),
                            "mkt": g.mkt.apply(lambda v: np.prod(1 + v) - 1)})
    return daily, monthly


def run_seed(seed):
    path = momentum_panic(seed, YEARS)
    daily, monthly = frames(path)
    res = dm.run(daily, monthly)
    t4 = res.table5["(4)"]
    coef, t = t4.loc["int", "coef"], t4.loc["int", "t"]
    se = coef / t
    months = res.weights.index
    w_orc = pd.Series(path.mu_month[months] / path.var_month[months], index=months)
    keep = np.isin(path.month, months)
    w_orc = dm.scale_to_vol(w_orc, path.wml[keep], path.month[keep])
    orc = w_orc.reindex(path.month[keep]).to_numpy() * path.wml[keep]
    sr = res.sharpe.to_dict()
    sr["oracle"] = orc.mean() / orc.std(ddof=1) * np.sqrt(dm.DAYS)
    return {"seed": seed, "gamma_int": coef, "se": se, "covers": abs(coef - DM_GAMMA_INT) <= 1.96 * se,
            **{f"Sharpe {k}": v for k, v in sr.items()}, "GJR alpha": res.gjr["alpha"], "GJR beta": res.gjr["beta"],
            "GJR gamma": res.gjr["gamma"]}


def results():
    return cached("sim_daniel_moskowitz", lambda: pd.DataFrame([run_seed(s) for s in range(SEEDS)]), DEPS)


def criteria(df):
    m = df.mean(numeric_only=True)
    return {
        "18a interaction CI covers the true -0.397 in >= 80% of seeds": (df.covers.mean() >= 0.8,
                                                                        f"{100 * df.covers.mean():.0f}%"),
        "18b dyn Sharpe > static Sharpe in >= 80% of seeds": (
            (df["Sharpe dyn"] > df["Sharpe static"]).mean() >= 0.8,
            f"{100 * (df['Sharpe dyn'] > df['Sharpe static']).mean():.0f}%"),
        "18c mean dyn Sharpe >= mean cvol Sharpe": (m["Sharpe dyn"] >= m["Sharpe cvol"],
                                                   f"{m['Sharpe dyn']:.3f} vs {m['Sharpe cvol']:.3f}"),
        "18d oracle mean Sharpe >= dyn, cvol, static": (
            m["Sharpe oracle"] >= max(m["Sharpe dyn"], m["Sharpe cvol"], m["Sharpe static"]),
            f"{m['Sharpe oracle']:.3f} vs {m['Sharpe dyn']:.3f} / {m['Sharpe cvol']:.3f} / {m['Sharpe static']:.3f}"),
    }


def main():
    df = results()
    crit = criteria(df)
    intro = (f"{SEEDS} seeds x {YEARS} years of `momentum_panic` ({DAYS_PER_MONTH}-day months): daily market returns "
             "from a calm/turbulent Markov chain; daily WML with the true monthly mean gamma0 + gamma_int * I_B * "
             "sigma^2_m (Table 5 col. 4 values) and GJR-GARCH variance (Appendix D values). The paper's pipeline is "
             "run unchanged; the oracle uses the true monthly mean and variance.")
    lines = ["# Daniel & Moskowitz: simulation ground truth", "", intro, "",
             "## Acceptance criteria (SPEC 18)", "", "| Criterion | Result | Numbers |", "|---|---|---|"]
    lines += [f"| {k} | {verdict(ok)} | {txt} |" for k, (ok, txt) in crit.items()]
    cols = [c for c in df.columns if c not in ("seed", "covers")]
    summ = pd.DataFrame({"mean": df[cols].mean(), "sd": df[cols].std(), "min": df[cols].min(), "max": df[cols].max()})
    lines += ["", "## Per-seed summary", "", summ.to_markdown(floatfmt=".4f"), ""]
    (ROOT / "reports" / "sim_daniel_moskowitz.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))
    return crit


if __name__ == "__main__":
    sys.exit(0 if all(ok for ok, _ in main().values()) else 1)
