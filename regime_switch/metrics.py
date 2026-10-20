"""Performance metrics exactly as each paper defines them, plus regime-accuracy scores against known truth."""

import numpy as np
import pandas as pd

DAYS = 252


def max_drawdown(wealth):
    wealth = np.asarray(wealth, dtype=float)
    peak = np.maximum.accumulate(wealth)
    return float(np.max(1 - wealth / peak))


def cagr(daily_returns):
    r = np.asarray(daily_returns, dtype=float)
    return float(np.prod(1 + r) ** (DAYS / r.size) - 1)


def shu_metrics(total_returns, rf, weights):
    """Table 4 of Shu, Yu & Mulvey (2024). `total_returns` include the risk-free leg and costs."""
    r = np.asarray(total_returns, dtype=float)
    ex = r - np.asarray(rf, dtype=float)
    w = np.asarray(weights, dtype=float)
    vol = ex.std(ddof=1) * np.sqrt(DAYS)
    mean_ex = ex.mean() * DAYS
    mdd = max_drawdown(np.cumprod(1 + r))
    tail = np.sort(r)[: max(1, int(np.floor(0.05 * r.size)))]
    years = r.size / DAYS
    return {
        "Return": cagr(r),
        "Volatility": vol,
        "Sharpe": mean_ex / vol,
        "MDD": -mdd,
        "Calmar": mean_ex / mdd,
        "ES0.05": float(tail.mean()),
        # "buys and sells 44% of total allocation (a combined 88% trading) each year"
        "Turnover": float(np.abs(np.diff(w)).sum() / 2 / years),
        "Leverage": float(w.mean()),
    }


def kinlaw_annual_sd(daily_returns, horizon=DAYS):
    """Annual SD adjusted for autocorrelation (Kinlaw, Kritzman & Turkington 2015):
    var_T = var_1 * (T + 2 * sum_{k=1}^{T-1} (T - k) rho_k)."""
    x = np.asarray(daily_returns, dtype=float)
    x = x - x.mean()
    var1 = x.var()
    n = x.size
    k = np.arange(1, horizon)
    rho = np.array([np.dot(x[:-j], x[j:]) / (n * var1) for j in k])
    return float(np.sqrt(var1 * (horizon + 2 * np.sum((horizon - k) * rho))))


def nystrup_metrics(daily_returns, turnover_per_day):
    """Table 1 of Nystrup, Madsen & Lindström (2018): AR, SD (autocorrelation-adjusted), SR = AR/SD, MDD, CR = AR/MDD, AT."""
    r = np.asarray(daily_returns, dtype=float)
    ar = cagr(r)
    sd = kinlaw_annual_sd(r)
    mdd = max_drawdown(np.cumprod(1 + r))
    return {
        "Annualized return": ar,
        "Standard deviation": sd,
        "Sharpe ratio": ar / sd,
        "Maximum drawdown": mdd,
        "Calmar ratio": ar / mdd,
        "Annual turnover": float(np.sum(turnover_per_day) / (r.size / DAYS)),
        "SD (unadjusted)": float(r.std(ddof=1) * np.sqrt(DAYS)),
    }


def sharpe(daily_excess):
    x = np.asarray(daily_excess, dtype=float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(DAYS))


def shifts_per_year(states):
    s = np.asarray(states)
    return float(np.sum(s[1:] != s[:-1]) / (s.size / DAYS))


def balanced_accuracy(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    recalls = [np.mean(pred[truth == k] == k) for k in (0, 1) if np.any(truth == k)]
    return float(np.mean(recalls))


def detection_latency(pred, truth):
    """Median number of days from each true regime change until the prediction first agrees with the new regime."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    changes = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    lags = []
    for i, c in enumerate(changes):
        end = changes[i + 1] if i + 1 < len(changes) else truth.size
        hit = np.flatnonzero(pred[c:end] == truth[c])
        if hit.size:
            lags.append(hit[0])
    return float(np.median(lags)) if lags else float("nan")


def table(rows: dict, fmt="{:.3f}"):
    df = pd.DataFrame(rows)
    return df.map(lambda v: fmt.format(v) if isinstance(v, float) else v).to_markdown()
