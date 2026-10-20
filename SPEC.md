# regime-switch SPEC

## Goal

Re-implement four regime-switching papers **as written**, one module per paper, and prove each implementation is correct in two ways:

1. **Ground truth on simulated data.** Every method runs on synthetic paths whose true regimes (or true conditional moments) are known, and is scored against that truth and against an oracle that knows the generating parameters.
2. **Replication on real data.** Every method is re-run on the closest freely available data and its headline table is printed next to the paper's numbers, with each headline claim marked reproduced or not.

| Division | Paper | Module |
|---|---|---|
| Jump model | Shu, Yu & Mulvey (2024), *Downside Risk Reduction Using Regime-Switching Signals: A Statistical Jump Model Approach*, arXiv 2402.05272 | `regime_switch/shu2024.py` (+ `jump.py`) |
| Classical HMM | Nystrup, Madsen & Lindström (2018), *Dynamic Portfolio Optimization Across Hidden Market Regimes*, Quant. Finance 18(1) | `regime_switch/nystrup2018.py` |
| Distribution clustering | Horvath, Issa & Muguruza (2021), *Clustering Market Regimes using the Wasserstein Distance*, arXiv 2110.11848 | `regime_switch/horvath2021.py` |
| Strategy switching | Daniel & Moskowitz, *Momentum Crashes* (2013 working paper, JFE 2016) | `regime_switch/daniel_moskowitz.py` |

"As written" means: every equation, hyperparameter, window, delay, cost and metric definition is taken from the paper. Where the paper is silent or the data is unavailable, the choice made is logged in `DEVIATIONS.md` with the reason. Nothing is "improved".

## Non-goals / out of scope

- Improving on the papers, adding features, or tuning beyond what a paper specifies.
- A live trading system, broker integration, or intraday execution.
- Paid data (Bloomberg, CRSP, MSCI, GFD). Only free sources: Yahoo Finance, Ken French's data library, FRED.
- Multi-asset portfolios. Every paper here is one risky asset vs cash (or one long-short strategy), and so is this repo.
- The continuous jump model (CJM), sparse jump models, and the other follow-ups the papers cite.
- Horvath et al.'s real-data section (hourly SPY 2005-2020 is not freely available). Only their synthetic experiments are replicated.
- Integration with `backtest_engine`. The backtests here are deliberately small and paper-specific; wiring into the engine is a later spec.

## Files & interfaces

```
regime_switch/
  sim.py               ground-truth simulators (see Simulations)
  metrics.py           performance metrics as each paper defines them; regime-accuracy scores
  backtest.py          0/1 and weight backtests with one-way costs and trading delay
  data.py              cached loaders: Yahoo index closes, FRED rates, Ken French files
  jump.py              statistical jump model: DP, coordinate-descent fit, lookback online inference
  shu2024.py           features, rolling JM + rolling HMM, smoothing, time-series CV, 0/1 strategy
  nystrup2018.py       2-state Gaussian HMM with adaptive (forgetting-factor) score estimator, forecasts, MPC
  horvath2021.py       stream lift, WK-means, MK-means, HMM benchmark, accuracy scores, MMD
  daniel_moskowitz.py  WML, bear indicator, market variance, mean regression, GJR-GARCH, vol regression, dyn/cvol weights
scripts/
  sim_<paper>.py       ground-truth validation runs -> reports/sim_<paper>.md
  replicate_<paper>.py real-data replication runs -> reports/replicate_<paper>.md
tests/                 pytest; fast unit tests by default, `-m slow` for simulation studies
reports/               generated markdown reports (committed)
DEVIATIONS.md          every deviation from a paper, with reason
```

Public entry points (one per paper), each returning plain pandas objects:

- `shu2024.run(returns, rf, cfg) -> Result` (online regimes, chosen lambda/k per month, strategy returns, metrics)
- `nystrup2018.run(log_returns, cfg) -> Result` (parameter paths, forecasts, MPC trades/holdings, metrics)
- `horvath2021.wk_means(segments, k, p) / mk_means(segments, k, p)` and `horvath2021.synthetic_experiment(model, cfg)`
- `daniel_moskowitz.run(daily, monthly, cfg) -> Result` (forecasts, weights, dyn/cvol/static returns, regressions)

## Key decisions & tradeoffs

1. **Python package, numba for the hot loops.** The JM dynamic program and the HMM forward/score recursions run per day over thousands of observations; numba keeps the exact algorithms (no approximations) at acceptable speed. Pure numpy elsewhere.
2. **Reference cross-checks over trust.** Where a published implementation exists it is used as an oracle in tests, never as the implementation: `jumpmodels` (the Shu et al. authors' package) for JM fits and features, `hmmlearn` for HMM filtering/likelihood, `arch` for GJR-GARCH, `scipy.stats.wasserstein_distance` for W1.
3. **Two-tier validation.** Simulations prove the algorithms do what the papers claim when the truth is known. Real-data runs show how close we get to the paper's tables. A failed real-data replication is reported as such, never tuned away.
4. **Oracle-relative thresholds.** Simulation pass marks are set relative to what is achievable: the forward filter run with the *true* generating parameters is the best online state classifier possible, so detectors are scored as a fraction of it, not against an invented absolute.
5. **Deterministic.** Every simulation and every multi-start fit takes a seed; reports are reproducible bit-for-bit on the same machine.

## Paper-by-paper specification

### 1. Shu, Yu & Mulvey (2024): JM-guided 0/1 strategy

- Universe: one index + its 3-month risk-free rate. Strategy holds 100% index (`s=0`, bull) or 100% risk-free (`s=1`, bear).
- Features from daily excess returns `R` (Table 2): EWM downside deviation `sqrt(EWM(min(R,0)^2))`, halflife 10; EWM Sortino `EWM(R)/DD` at halflives 20 and 60 (pandas `ewm(halflife=h)`, as in the authors' example code). Standardised with mean/std of the training window.
- JM (eq. 1): K=2, loss `0.5*||x-theta||^2`, jump penalty lambda; coordinate descent (M: within-state means, E: DP), 10 k-means++ inits, keep lowest objective. Bull = state with higher cumulative excess return in the training window.
- Refit every 6 months on a 3000-day training window. Online inference for day t: DP over the 3000-day lookback ending at t with the latest parameters; take the last state.
- HMM baseline: 2-state Gaussian HMM (hmmlearn) on daily log total returns, 3000-day window refit **daily**, 10 inits with k-means++ means, keep max log-likelihood; Viterbi, last state; high-vol state = 1; median filter = rolling mean of the online states over k days, bear if > 0.5.
- Hyperparameter CV: at each month start, for each candidate lambda (JM) or k (HMM), run the 0/1 strategy on the online regimes over the trailing 8-year window with 1-day delay and 10 bp one-way cost; pick max Sharpe; use it for the next month; the choice itself also obeys the delay.
- Trading delay d: regime inferred at end of day t is applied from day t+1+d (d=1 baseline; 5 and 10 for Table 5).
- Metrics (Table 4): CAGR incl. rf, annualised vol, Sharpe (mean excess / vol), MDD, Calmar (mean excess / MDD), daily ES 5%, annual turnover, average leverage.
- Paper targets: Table 3 (shifts/yr by lambda and k), Table 4, Table 5.

### 2. Nystrup, Madsen & Lindström (2018): adaptive HMM + MPC

- 2-state Gaussian HMM on daily log-returns, parameters `(mu1, mu2, sigma1^2, sigma2^2, gamma11, gamma22)`, estimated online with eq. (7): `theta_t = theta_{t-1} + A * I_t^{-1} * grad l~_t(theta_{t-1})`, exponential weights `f^(t-n)`, `N_eff = 1/(1-f) = 260`, `A = 1/N_eff`, gradient by an exact forward-mode (Lystig-Hughes style) recursion of the conditional log-likelihood terms; first 260 observations used for initialisation.
- Forecasts: filtered state probabilities (eq. 8), k-step probabilities (eq. 9), per-state log-normal mean/variance of simple returns (eqs. 12-13), mixture moments (eqs. 10-11), horizons 1..K, K=100.
- MPC (eq. 25) with stock + zero-interest cash, holdings normalised to V_t=1, self-financing with costs (eq. 16, kappa=0.001 on the stock leg), long-only, risk penalty `gamma * h'Sigma h / V+` (eq. 22), trading penalty `rho*|u|` (eq. 23), terminal value `1'h_{t+K}`; execute the first trade only (Algorithm 1). Solved with CVXPY.
- Configurations: MPC(gamma=0, rho=0), MPC(gamma=2, rho=0), MPC(gamma=0, rho=0.02); with and without a one-day delay (delayed trades executed as a fraction of the holding).
- Metrics (Table 1 notes): annualised return, SD adjusted for autocorrelation (Kinlaw et al. 2015), Sharpe = AR/SD, MDD, Calmar = AR/MDD, annual turnover.
- Paper targets: Table 1 and Table 3 (S&P 500, DAX, FTSE columns; MSCI World/TOPIX/MSCI EM are not freely available).

### 3. Horvath, Issa & Muguruza (2021): Wasserstein k-means

- Lift (eq. 3): segments of length h1 with offset h2; empirical measures; W_p between equal-size empirical measures = `(mean |sorted a - sorted b|^p)^(1/p)` (eq. 21); barycenter for p=1 = coordinate-wise median of sorted atoms (Prop. 2.6); Algorithm 1 with centroids initialised by sampling k segments, stop when the summed centroid move (eq. 23) < tolerance.
- Benchmarks: MK-means on standardised first p raw moments (eqs. 24-25); 2-state Gaussian HMM per return.
- Accuracy (eqs. 31-33): ROFS, RONS, TA with per-return label vectors aggregated over every segment containing that return. Regime-on = cluster with the higher centroid variance.
- Synthetic experiments (Sec. 3.3): hourly mesh `n=252*7`, T=20 years, r=10 bear episodes of length 0.5*252*7 at random non-overlapping starts; gBm `theta_bull=(0.02,0.2)`, `theta_bear=(-0.02,0.3)`; Merton `theta_bull=(0.05,0.2,5,0.02,0.0125)`, `theta_bear=(-0.05,0.4,10,-0.04,0.1)`; 50 runs; (h1,h2)=(35,28); k=2, p=1 for WK-means.
- MMD validation: biased MMD^2 with Gaussian kernel sigma=0.1 and the median self-similarity score (Def. 1.9).
- Paper targets: Table 3 (gBm) and Table 4 (Merton).

### 4. Daniel & Moskowitz: dynamic momentum

- WML = winner decile - loser decile (value-weighted, prior 12-2 ranking), daily and monthly, from Ken French's `10_Portfolios_Prior_12_2` files; market = Mkt-RF + RF; RF = 1-month T-bill.
- Bear indicator `I_B,t-1` = 1 if the cumulative market return over the past 24 months is negative. Market variance `sigma^2_m,t-1` = variance of daily market returns over the 126 days before the month.
- Table 5 regressions (monthly, 1927:07-2013:03). Mean forecast = fitted values of the interaction regression (see DEVIATIONS on column 4 vs 5).
- Vol forecast: GJR-GARCH(1,1) (eqs. 6-7) fit by ML on full-sample daily WML; monthly OLS of next-month realised 22-day daily vol on the GARCH vol and lagged 126-day realised vol; fitted value is the forecast.
- Weights (eq. 5): `w = mu / (2*lambda*sigma^2)` set monthly, lambda chosen so the in-sample annualised vol is 19%. Constant-volatility comparison (Barroso & Santa-Clara): `w ∝ 1/sigma_126`, scaled to 19%.
- Paper targets: Table 1 (WML row), Table 5, Appendix D estimates, Table 7 Panel A column 1 (alpha 23.74%/yr, dyn scaled to WML vol) and column 4.

## Simulations (ground truth)

All in `sim.py`, all seeded, all returning the series **and** the true latent path / true conditional moments.

- `markov_switching_gaussian`: daily 2-state Gaussian Markov-switching returns. Default "benchmark" parameters: bull mu=0.0006, sigma=0.007; bear mu=-0.0008, sigma=0.018; P(stay bull)=0.996, P(stay bear)=0.985 (mean bear spell ~67 days).
- `markov_switching_t`: same, Student-t (df=4) innovations scaled to the same variance (non-Gaussian robustness).
- `drifting_hmm`: Gaussian 2-state HMM whose parameters move slowly over time along a known path (tests the adaptive estimator's tracking).
- `horvath_gbm`, `horvath_merton`: the paper's exact synthetic generators (above).
- `momentum_panic`: daily market returns from a 2-state calm/turbulent Markov chain; daily WML returns whose conditional mean follows the paper's interaction model (true gammas from Table 5 col. 4) and whose variance follows GJR-GARCH with the Appendix D parameters. Returns the true conditional mean and variance for every day.

## Edge cases

- Warm-up: no regime or forecast is produced before a method's window is full (3000 days for Shu, 260 for Nystrup, 126 days / 24 months for D&M); strategies hold the benchmark position (index for 0/1, zero for WML overlays) during warm-up and this is excluded from scored periods.
- A JM fit that collapses to one state (all labels equal) is valid output, not an error; bull/bear labelling then marks that single state by its cumulative return.
- Label switching: every K=2 model maps states to (favourable, unfavourable) by the paper's rule before any scoring.
- Missing data / holidays: returns are computed on each index's own trading calendar; rates are forward-filled onto it; no cross-index alignment (each paper tests indices one at a time).
- MPC infeasibility or solver failure: fall back to holding the current portfolio for that day and count the event; the report lists the count (expected 0).
- Ties in the DP: lowest state index wins (deterministic).
- GJR-GARCH non-convergence: fail loudly; never silently keep starting values.

## Acceptance criteria

Fast suite: `pytest -q` passes in under 5 minutes. Slow suite: `pytest -q -m slow` passes. Each item names the test or artifact that checks it.

### Correctness (unit, fast)
1. JM DP returns the exact optimum: equals brute-force enumeration on 200 random problems with T<=10, K in {2,3} (`test_jump.py`).
2. JM fit reaches an objective <= `jumpmodels.JumpModel` objective (+1e-8 relative) on the same standardised features, lambda and inits, for 5 random datasets (`test_jump.py`).
3. Lookback online inference equals the last state of the brute-force optimal path over the same window (`test_jump.py`).
4. Shu features equal the authors' example definitions (pandas EWM, same halflives) to 1e-12 (`test_shu2024.py`).
5. The 0/1 backtest applies the delay exactly: a signal change at end of day t first affects the day t+1+d return; costs are 10 bp per one-way switch (`test_backtest.py`).
6. Nystrup per-observation score equals central finite differences of the conditional log-likelihood to rtol 1e-5; filtered probabilities equal hmmlearn's forward pass to 1e-10 for fixed parameters (`test_nystrup2018.py`).
7. Mixture forecast mean/variance (eqs. 10-13) match a 10^6-draw Monte Carlo of the same mixture to 3 standard errors (`test_nystrup2018.py`).
8. MPC sanity: with gamma=rho=kappa=0 it is all-in stocks iff the horizon-1 forecast mean > 0 and all cash otherwise; with kappa=0.02 there exists a no-trade band (a state where neither buying nor selling is optimal) (`test_nystrup2018.py`).
9. W_p for equal-size empirical measures equals `scipy.stats.wasserstein_distance` for p=1 to 1e-12; the median barycenter attains a sum-of-W1 no larger than 1000 random perturbations of it (`test_horvath2021.py`).
10. ROFS/RONS/TA reproduce hand-computed values on a small worked example (`test_horvath2021.py`).
11. GJR-GARCH ML estimates agree with the `arch` package on the same series (each parameter within 2 of arch's standard errors; log-likelihood within 0.1) (`test_daniel_moskowitz.py`).
12. Bear indicator, 126-day market variance and monthly WML aggregation match direct hand computations on a toy series (`test_daniel_moskowitz.py`).

### Ground truth (slow, simulations; numbers written to `reports/sim_<paper>.md`)
13. **JM (Shu):** on `markov_switching_gaussian`, 20 seeds x 30 years, the online JM regimes at the CV-selected lambda reach >= 90% of the true-parameter forward filter's balanced accuracy, and have fewer shifts/yr than lambda=0 (k-means). The JM 0/1 strategy's Sharpe is >= the buy-and-hold Sharpe on average across seeds.
14. **JM vs HMM (Shu's claim):** on the same paths, JM's online shifts/yr < HMM's (k=0) and JM's 0/1 turnover < HMM's. Reported, together with whether JM's Sharpe > HMM's.
15. **Adaptive HMM (Nystrup):** on `markov_switching_gaussian` the adaptively estimated parameters converge to within 2 standard deviations of their sampling spread around the truth (measured over 20 seeds at t = 10 years); on `drifting_hmm` the estimated `mu`/`sigma` track the true path with lower mean absolute error than a static full-sample MLE. The filtered state classifier reaches >= 90% of the true-parameter filter's balanced accuracy.
16. **MPC (Nystrup):** on `markov_switching_gaussian`, MPC(0,0) with kappa=0.001 achieves a Sharpe >= buy-and-hold on average over 20 seeds, and MPC(0,0.02) has lower turnover than MPC(0,0).
17. **WK-means (Horvath):** on the paper's gBm and Merton generators (50 runs each) the mean total accuracy of WK-means is within 5 percentage points of the paper's Table 3/4 value (90.60%, 91.28%), and the paper's ordering holds on Merton (WK > MK and WK > HMM in total accuracy). MK and HMM numbers are reported beside the paper's.
18. **Dynamic momentum (D&M):** on `momentum_panic` (20 seeds x 80 years) the interaction-regression coefficient covers the true value in its 95% CI in >= 80% of seeds; the dynamic strategy's Sharpe exceeds the static WML Sharpe in >= 80% of seeds and its mean Sharpe is >= the constant-vol strategy's; the oracle strategy (true mu, sigma) bounds all three from above on average.

### Replication (real data; `reports/replicate_<paper>.md`)
19. Each report contains: data sources and date ranges actually used, the paper's table, our table in the same layout, and one line per headline claim marked REPRODUCED / NOT REPRODUCED with the numbers that decide it. A NOT REPRODUCED line is acceptable; a missing line is not.
20. Shu: Table 3 (shifts/yr), Table 4 and Table 5 for S&P 500, DAX, Nikkei 225. Headline claims checked: JM-0/1 Sharpe > B&H and > HMM-0/1 on each index; JM MDD < B&H MDD on each index; JM turnover < HMM turnover on each index.
21. Nystrup: Table 1 for S&P 500 (closest free substitute for MSCI World) and Table 3 for S&P 500, DAX, FTSE. Headline claims checked: MPC SD and MDD < B&H on every index; trading penalty lowers turnover; MPC Sharpe > B&H on most indices.
22. Horvath: synthetic Tables 3 and 4 (this is their replication; see criterion 17) plus the MMD self-similarity scores of the WK and MK clusters.
23. D&M: Table 1 WML row, Table 5 (all five columns), Appendix D GJR and vol-regression estimates, Table 7 Panel A columns 1 and 4. Headline claims checked: interaction coefficient negative and significant; dyn Sharpe > cvol Sharpe > static WML Sharpe; Table 7 col. 1 alpha positive and significant.

### Hygiene
24. `DEVIATIONS.md` lists every place the implementation departs from a paper or fills a gap the paper leaves, each with a reason; every DEVIATIONS entry is referenced by an `# see DEVIATIONS.md#<id>` comment at the code site.
25. `ruff check .` is clean.
