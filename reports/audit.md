# Implementation audit against the papers

Every methodological element of each paper, checked against the code. Status: **Match** (as written), **Deviation** (logged in `DEVIATIONS.md`, with the reason), **Open** (a paper result we do not reproduce and cannot explain from the text). No implementation errors were found in this pass.

## Shu, Yu & Mulvey (2024): `regime_switch/shu2024.py`, `jump.py`

| Element | Paper | Code | Status |
|---|---|---|---|
| Features | EWM downside deviation (hl 10), EWM Sortino (hl 20, 60) on daily excess returns (Table 2) | `features`, checked against the authors' example code to 1e-12 | Match |
| Feature warm-up | Not stated | First 252 rows dropped | Deviation S3 |
| Feature clipping | Not stated (authors' example code clips at +-3 sd) | Not clipped; tested, no material effect | Deviation S6 |
| Standardisation | "standardized features" | Training-window mean/std, applied to online data | Match |
| JM objective | eq. 1: 0.5 squared L2 loss + lambda per switch, K = 2 | `jump.fit`, `dp_path`; DP equals brute force; objective <= authors' package | Match |
| Fitting | Coordinate descent, DP for states, 10 runs, lowest objective | `jump.fit` with 10 k-means++ starts | Match |
| Bull / bear | Higher cumulative excess return over the training window = bull | `jm_online` | Match (collapsed fits labelled by sign) |
| Refits | Every 6 months, 3000-day window | June / December month ends | Match (calendar S5) |
| Online inference | DP over 3000-day lookback with latest parameters, last state | `jump.online_states`; equals brute force | Match |
| HMM | 2-state Gaussian on daily log total returns, 3000 days, refit daily, 10 k-means++ starts, Viterbi, last state; high vol = 1 | `hmm_online_raw` | Match (implementation S4) |
| Median filter | Rolling mean of online states over k days, bear if > 0.5 | `median_filter` | Match |
| Candidate grids | Not stated | lambda {0..150}, k {0..20} | Deviation S2 |
| Cross-validation | Monthly, 8-year window, 0/1 strategy with 1-day delay and 10 bp, max Sharpe; choice also delayed | `cross_validate` | Match |
| 0/1 strategy | Signal at end of day t applied from day t+2; 10 bp one-way | `backtest.zero_one`, tested | Match |
| Test period | 1990-2023 | 1990-2023 (DAX 2009 for lack of data) | Deviation S1 |
| Metrics | Table 4 definitions | `metrics.shu_metrics` | Match |
| Data | Bloomberg total returns, GFD 3-month bills | Yahoo price indices (DAX total return), FRED rates | Deviation S1 |
| Table 3 JM shift counts | lambda = 0: 9.7/yr ... 150: 0.4/yr | 15.2 ... 0.4/yr; the authors' own package gives the same (15.7 ... 0.5) | **Open**: outside the paper and published code (S6) |

## Nystrup, Madsen & Lindström (2018): `regime_switch/nystrup2018.py`

| Element | Paper | Code | Status |
|---|---|---|---|
| Model | 2-state Gaussian HMM on daily log-returns | `score_pass` | Match |
| Parameter transform | "a transformation" for constrained parameters | log variances, logit staying probabilities | Deviation N2 |
| Weighted likelihood | weights f^(t-n), N_eff = 1/(1-f) = 260 | `score_pass` | Match |
| Score | Lystig-Hughes exact score of log Pr(y_n given y_1..n-1) | Forward-mode recursion; equals finite differences | Match |
| Update (eq. 7) | theta_t = theta_{t-1} + A I_t^-1 grad, A = 1/N_eff | `_adapt` | Match, with the sum reading of I_t (Deviation N4) |
| Initialisation | "first 260 observations used for initialization" | Neutral persistent start refined over them | Deviation N3 |
| Safety box | Not in paper | Parameter bounds, never hit in reported runs | Deviation N4 |
| Filtered probabilities (eq. 8) | Time-varying parameters | `filtered_probs`; equals hmmlearn for fixed parameters | Match |
| Forecasts (eqs. 9-13) | k-step state probabilities, log-normal per-state moments, mixture moments | `forecast_moments`; equals Monte Carlo | Match |
| MPC (eq. 25) | Maximise terminal value minus risk (eq. 22) and trading (eq. 23) penalties; self-financing with costs (eq. 16); long-only; K = 100 | `MPC` | Match (costs on the stock leg) |
| Algorithm 1 | Execute first trade, re-solve daily | `simulate_mpc` | Match |
| One-day delay | Executed next day as fraction of holding | `simulate_mpc(delay=1)` | Match (decision basis N6) |
| Metrics | AR, SD adjusted for autocorrelation (Kinlaw et al.), SR = AR/SD, MDD, CR, AT | `metrics.nystrup_metrics` | Match |
| RBAA benchmark | Rule from Nystrup et al. (2017) | Not reproduced | Deviation N5 |
| Data | MSCI World and others, net total return | S&P 500, DAX, FTSE from Yahoo | Deviation N1 |

## Horvath, Issa & Muguruza (2021): `regime_switch/horvath2021.py`

| Element | Paper | Code | Status |
|---|---|---|---|
| Stream lift (eq. 3) | Segments of h1 with overlap h2 | `lift`, step h1 - h2 | Deviation H1 (reading of eq. 3) |
| W_p (eq. 21) | Mean of abs differences of sorted atoms, power 1/p | `wasserstein`; equals scipy | Match |
| Barycenter (Prop. 2.6) | Coordinate-wise median for p = 1 | `barycenter`; minimises summed W1 | Match |
| WK-means (Algorithm 1) | Sample k initial centroids, assign by W_p, barycenter update, stop when summed centroid move < tol | `wk_means` | Match |
| MK-means (eqs. 24-25, Algorithm 2) | Standardised raw moments, standard k-means | `mk_means`, p = 4 | Match (p unstated: H2) |
| Accuracy (eqs. 31-33) | ROFS, RONS, TA over per-return label counts | `label_counts`, `accuracy`; hand example | Match |
| MMD | Biased MMD^2, Gaussian kernel sigma = 0.1, median self-similarity | `mmd2_biased`, `self_similarity` | Match |
| gBm generator (eqs. 34-35) | Hourly mesh, 20 years, 10 half-year episodes, given parameters | `sim.horvath_gbm` | Match |
| Merton generator (eqs. 36-40) | Poisson jumps, log jump N(gamma, delta^2), given parameters | `sim.horvath_merton`; moments match eqs. 39-40 | Match |
| HMM benchmark | Gaussian HMM, settings unstated | MLE fit and hmmlearn-defaults fit both reported | Deviation H3 |
| Merton accuracy | WK 91.28% | 98.92% | **Open** (we exceed the paper; generator matches its equations) |

## Daniel & Moskowitz: `regime_switch/daniel_moskowitz.py`

| Element | Paper | Code | Status |
|---|---|---|---|
| WML | Top minus bottom decile, value-weighted, 12-2 ranking | French deciles | Deviation D1 (NYSE breakpoints) |
| Bear indicator | Cumulative market return over the past 24 months < 0 | `bear_indicator`, hand-tested | Match |
| Market variance | Daily market return variance, annualised, preceding 126 days | `market_variance`, hand-tested | Match (D3 now confirmed by the paper's Table 8 notes) |
| Table 5 regressions | Monthly WML on I_B, variance, interaction | `table5` | Match |
| Mean forecast | Fitted interaction regression ("last column") | Column (4), column (5) also reported | Deviation D2 |
| GJR-GARCH (eqs. 6-7) | Full-sample Gaussian ML on daily WML | `gjr_fit`; equals the `arch` package | Match |
| Vol forecast | Monthly OLS of next-month 22-day realised vol on sigma_GARCH,t and lagged 126-day vol | `vol_forecast` | Match (GARCH timing D5) |
| Weights (eq. 5) | mu / (2 lambda sigma^2), lambda set for 19% in-sample vol | `run`, `scale_to_vol` | Match |
| Constant-vol benchmark | Scaled by trailing realised vol, 19% target | `run` | Match |
| Spanning tests (Table 7) | Daily OLS, alpha x 252 x 100, dyn/cvol scaled to WML vol | `spanning` + replication script | Match |
| Sample | 1927:07-2013:03 | Regressions from 1928:07 (24-month warm-up on French data) | Deviation D4 |
