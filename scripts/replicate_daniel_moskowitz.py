"""Daniel & Moskowitz on Ken French data, 1927-2013 (the paper's sample) and extended to the latest month."""

import numpy as np
import pandas as pd

from regime_switch import daniel_moskowitz as dm
from regime_switch import data
from scripts.common import ROOT

PAPER_T5 = {  # coefficient (t); gamma0, gammaB in percent per month
    "(1)": {"const": (1.955, 6.6), "IB": (-2.626, -3.8)},
    "(2)": {"const": (2.428, 7.5), "var": (-0.330, -5.1)},
    "(3)": {"const": (2.500, 7.7), "IB": (-1.281, -1.6), "var": (-0.275, -3.8)},
    "(4)": {"const": (1.973, 7.1), "int": (-0.397, -5.7)},
    "(5)": {"const": (2.129, 5.8), "IB": (0.023, 0.0), "var": (-0.088, -0.8), "int": (-0.323, -2.2)},
}
PAPER_GJR = {"mu": 0.86e-3, "omega": 1.17e-6, "alpha": 0.111, "gamma": -0.016, "beta": 0.896}
PAPER_VOLREG = {"const": 0.0010, "sigma_GARCH": 0.6114, "sigma_126": 0.2640}
PAPER_T1 = {"r - rf": 17.9, "sigma": 30.0, "alpha": 22.2, "t(alpha)": 7.3, "beta": -0.58, "SR": 0.60, "sk(m)": -4.70}
PAPER_T7 = {"dyn on Mkt+WML": (23.74, 11.99), "dyn on Mkt+cvol": (7.27, 6.86), "cvol on Mkt+WML": (14.27, 11.44)}


def load(end):  # Ken French data stands in for CRSP, see DEVIATIONS.md#D1
    d = data.french("10_Portfolios_Prior_12_2_Daily")
    m = data.french("10_Portfolios_Prior_12_2")
    fd = data.french("F-F_Research_Data_Factors_daily")
    fm = data.french("F-F_Research_Data_Factors")
    daily = pd.DataFrame({"wml": d["Hi PRIOR"] - d["Lo PRIOR"], "mkt": fd["Mkt-RF"] + fd["RF"],
                          "mktrf": fd["Mkt-RF"], "rf": fd["RF"]}).dropna()[:end]
    daily["month"] = daily.index.to_period("M")
    monthly = pd.DataFrame({"wml": m["Hi PRIOR"] - m["Lo PRIOR"], "mkt": fm["Mkt-RF"] + fm["RF"],
                            "mktrf": fm["Mkt-RF"], "rf": fm["RF"]}).dropna()
    monthly.index = monthly.index.to_period("M")
    return daily, monthly[: pd.Period(end, "M")]


def table1(monthly, start="1927-01"):
    m = monthly[start:]
    res, _ = dm.ols(m.wml, [m.mktrf], ["Mkt"])
    return {"r - rf": 1200 * m.wml.mean(), "sigma": 100 * np.sqrt(12) * m.wml.std(ddof=1),
            "alpha": 1200 * res.loc["const", "coef"], "t(alpha)": res.loc["const", "t"], "beta": res.loc["Mkt", "coef"],
            "SR": np.sqrt(12) * m.wml.mean() / m.wml.std(ddof=1),
            "sk(m)": float(pd.Series(np.log1p(m.wml + m.rf)).skew())}


def t5_table(t5):
    rows = []
    for col, res in t5.items():
        for var in res.index:
            c, t = res.loc[var, "coef"], res.loc[var, "t"]
            if var in ("const", "IB"):
                c *= 100
            p = PAPER_T5[col].get(var)
            rows.append({"column": col, "variable": var, "ours": f"{c:.3f} ({t:.1f})",
                         "paper (1927:07-2013:03)": f"{p[0]:.3f} ({p[1]:.1f})" if p else "-"})
    return pd.DataFrame(rows).to_markdown(index=False)


def spanning(res, daily):
    r = res.daily
    sc = r.static.std()
    dyn, cvol = r.dyn / r.dyn.std() * sc, r.cvol / r.cvol.std() * sc
    mk = daily.mktrf.loc[r.index]
    return {"dyn on Mkt+WML": dm.spanning(dyn, [mk, r.static], ["Mkt", "WML"]).loc["const"],
            "dyn on Mkt+cvol": dm.spanning(dyn, [mk, cvol], ["Mkt", "cvol"]).loc["const"],
            "cvol on Mkt+WML": dm.spanning(cvol, [mk, r.static], ["Mkt", "WML"]).loc["const"]}


def section(end, title, paper):
    daily, monthly = load(end)
    sl = slice(pd.Period("1927-07", "M"), pd.Period(end, "M"))  # see DEVIATIONS.md#D4
    t5, _ = dm.table5(monthly.wml[sl], dm.bear_indicator(monthly.mkt)[sl], dm.market_variance(daily)[sl])
    res = dm.run(daily, monthly)
    res5 = dm.run(daily, monthly, mean_col="(5)")
    span = spanning(res, daily)
    t1 = table1(monthly[: pd.Period(end, "M")])
    span_txt = (f"Daily data {daily.index[0].date()} to {daily.index[-1].date()}; strategies from "
                f"{res.daily.index[0].date()}.")
    out = [f"## {title}", "", span_txt, "", "### Table 1, WML row (monthly, annualised %)", "",
           pd.DataFrame({"ours": t1, "paper": PAPER_T1 if paper else {}}).to_markdown(floatfmt=".2f"), "",
           "### Table 5 (coefficient (t); const and IB in % per month)", "", t5_table(t5), "",
           "### Appendix D: GJR-GARCH on daily WML", "",
           pd.DataFrame({"ours": res.gjr, "paper": PAPER_GJR if paper else {}}).drop("loglik").to_markdown(
               floatfmt=".4g"), "", "### Appendix D: vol forecast regression (monthly)", "",
           pd.DataFrame({"ours coef": res.vol_reg.coef, "ours t": res.vol_reg.t,
                         "paper coef": pd.Series(PAPER_VOLREG) if paper else np.nan}).to_markdown(floatfmt=".4f"),
           "", "### Strategy Sharpe ratios (daily, annualised; all scaled to 19% vol)", "",
           pd.DataFrame({"mean = Table 5 col. (4)": res.sharpe, "mean = col. (5)": res5.sharpe}).to_markdown(
               floatfmt=".3f"), "", "### Table 7 spanning alphas (% per year (t); dyn and cvol scaled to WML vol)", "",
           pd.DataFrame({k: {"ours": f"{v['coef']:.2f} ({v['t']:.2f})",
                             "paper": f"{PAPER_T7[k][0]:.2f} ({PAPER_T7[k][1]:.2f})" if paper else "-"}
                         for k, v in span.items()}).T.to_markdown(), ""]
    claims = [
        ("Interaction coefficient (col. 4) negative and significant (t < -2)",
         t5["(4)"].loc["int", "t"] < -2, f"{t5['(4)'].loc['int', 'coef']:.3f} (t = {t5['(4)'].loc['int', 't']:.1f})"),
        ("Sharpe: dyn > cvol > static", res.sharpe.dyn > res.sharpe.cvol > res.sharpe.static,
         f"{res.sharpe.dyn:.3f} > {res.sharpe.cvol:.3f} > {res.sharpe.static:.3f}"),
        ("Table 7 col. 1 alpha positive and significant", span["dyn on Mkt+WML"]["t"] > 2,
         f"{span['dyn on Mkt+WML']['coef']:.2f}% (t = {span['dyn on Mkt+WML']['t']:.2f})"),
        ("Table 7 col. 4: dyn not spanned by cvol (alpha t > 2)", span["dyn on Mkt+cvol"]["t"] > 2,
         f"{span['dyn on Mkt+cvol']['coef']:.2f}% (t = {span['dyn on Mkt+cvol']['t']:.2f})"),
    ]
    out += ["### Headline claims", "", "| Claim | Verdict | Numbers |", "|---|---|---|"]
    out += [f"| {c} | {'REPRODUCED' if ok else 'NOT REPRODUCED'} | {n} |" for c, ok, n in claims]
    return out


def main():
    intro = ("Data: Ken French's `10_Portfolios_Prior_12_2` (daily and monthly, value-weighted) and Fama-French "
             "factors. WML = decile 10 - decile 1. Differences from the paper's CRSP construction are listed in "
             "DEVIATIONS.md#D1. Every estimate is full-sample, as in the paper.")
    lines = ["# Daniel & Moskowitz: replication on Ken French data", "", intro, ""]
    lines += section("2013-03-31", "Paper sample (to 2013-03)", True)
    lines += [""] + section("2099-12-31", "Extended to the latest available month", False)
    (ROOT / "reports" / "replicate_daniel_moskowitz.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
