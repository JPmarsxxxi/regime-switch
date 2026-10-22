# Deviations from the papers

Every place the code departs from a paper, or fills a gap the paper leaves open, is listed here with the reason. Code sites carry `# see DEVIATIONS.md#<id>`.

## Shu, Yu & Mulvey (2024)

<a id="S1"></a>**S1. Data.** Paper: Bloomberg daily total-return indices (S&P 500, DAX, Nikkei 225) 1970-2023, risk-free = each country's 3-month T-bill yield from GFD. Here: Yahoo Finance closes. `^GSPC` and `^N225` are price indices (dividends excluded, so B&H and strategy returns are lower by roughly the dividend yield while invested); `^GDAXI` is a total-return index but only starts 1987-12-30, so the DAX test period starts in 2008 instead of 1990 (12 years training + 8 years validation). Risk-free: FRED `DTB3` (US, daily), `IR3TIB01DEM156N` (Germany) and `IR3TIB01JPM156N` (Japan) 3-month interbank rates (monthly, forward-filled), converted as yield/252 per trading day.

<a id="S2"></a>**S2. Candidate grids.** The paper does not list the candidate jump penalties or median-filter lengths searched by its cross-validation. Used: lambda in {0, 5, 15, 35, 50, 70, 100, 150} (the Table 3 values plus 50 and 100, which the text calls "typical") and k in {0, 2, 4, 6, 8, 20} (the Table 3 values plus Bulla et al.'s original k = 6).

<a id="S3"></a>**S3. Feature burn-in.** The paper does not say how the start of the EWM features is handled. The first rows have a near-zero downside deviation and Sortino ratios in the hundreds; left in, they capture one of the two centroids. The first 252 days after the first finite feature row are dropped before any fit.

<a id="S4"></a>**S4. HMM fitting.** The paper uses hmmlearn with unspecified settings; hmmlearn's defaults (`n_iter=10`, `tol=1e-2`, `covars_prior=1e-2`, `min_covar=1e-3`) are sized for unit-scale data and would stop EM early and bias variances of daily returns upward. Here: a numba Baum-Welch (`regime_switch/hmm.py`, cross-checked against hmmlearn with the priors switched off in `tests/test_hmm.py`) run to convergence (`tol=1e-8`, at most 500 iterations), 10 starts with k-means++ means, hmmlearn's default starting values for the other parameters. Same model, same likelihood, same Viterbi decoding.

<a id="S5"></a>**S5. Refit calendar.** "Updated every six months": refits happen at the end of the last trading day of June and December, on the 3000 days ending that day; the fit is used from that day's inference onward.

<a id="S6"></a>**S6. No feature clipping.** The authors' public example code (`jumpmodels`, examples/nasdaq) clips each feature to its training mean +- 3 standard deviations before standardising; the paper does not mention clipping. The paper's text is followed (no clipping). Checked on the S&P 500: clipping changes the online shift counts by at most 0.5/yr (lambda = 0: 15.2 -> 15.7), so it does not explain why our jump-model shift counts are 1.5-1.8x the paper's Table 3 (the HMM counts match to within 0.1/yr). That gap is unexplained.

## Nystrup, Madsen & Lindström (2018)

<a id="N1"></a>**N1. Data.** Paper: net total-return indices (MSCI World 1997-2015; S&P 500, TOPIX, DAX, FTSE 1984-2015; MSCI EM 1988-2015). MSCI World, TOPIX and MSCI EM have no free daily history. Here: Yahoo `^GSPC` (price index) stands in for the Table 1 analysis, and Table 3 is reproduced for S&P 500 (price), FTSE (price) and DAX (total return, from 1988 instead of 1984).

<a id="N2"></a>**N2. Parametrisation.** The paper says constrained parameters must be transformed but not how. Used: means untransformed, log variances, logit of the two staying probabilities. Initial distribution of the score recursion: the stationary distribution of the current transition matrix.

<a id="N3"></a>**N3. Initialisation.** "The first 260 observations were used for initialization" is not specified further. A maximum-likelihood fit on those 260 days is not usable: on the S&P 500 in 1984 every start converges to a non-persistent "spike" state (P(stay) about 0.15 capturing a handful of large up days), from which the recursion degenerates. Used instead: a neutral persistent start (both means at the sample mean, variances at 0.5x and 2x the sample variance, P(stay) = 0.98), with the information matrix started from the scores of all 260 days at that start, run through the recursion over days 20-260.

<a id="N4"></a>**N4. Form of the information recursion, and a safety box.** Eq. 7 can be read with `I_t` as the exponentially weighted *mean* of score outer products (so A = 1/N_eff gives a full Newton step) or as the exponentially weighted *sum* `R_t = f R_{t-1} + s_t s_t'` (the recursive-least-squares form, which is what "updated recursively using (6) and the matrix inversion lemma" describes; A = 1/N_eff then damps the step). The mean form was implemented first: on simulated paths 8-11 of 20 held-out seeds degenerated (a regime unvisited for a year loses its information, the Newton steps for its parameters blow up and the state collapses), and on the S&P 500 it collapsed in 1993-95; a damping term only reduced this to 2-3 of 10. The sum form had 0 of 10 degenerate paths and the lowest tracking error on drifting parameters, so it is used. The choice was made on simulation seeds 100-119, which are not used in any reported result, and not on real data. As a pure safety net the parameters are also projected onto a box after each step (daily means within +-1%, daily variances within [1e-8, 1e-2], staying probabilities within [0.5, 0.9999]); with the sum form the box is not hit in the reported runs.

<a id="N5"></a>**N5. Rule-based RBAA benchmark.** Table 2's RBAA comparison depends on the rule of Nystrup et al. (2017) ("probability that a regime change has occurred exceeds 0.9998"), which is not specified in this paper. It is not reproduced. The SAA comparison (monthly rebalanced to MPC(0,0)'s average stock weight) is.

<a id="N6"></a>**N6. Delayed trades.** With a one-day delay the paper says only that decisions are executed the next day "as fractions of the asset holding". Each day's MPC decision is made on that day's holdings before the previous day's decision is executed.

## Horvath, Issa & Muguruza (2021)

<a id="H1"></a>**H1. Stream lift.** Eq. 3 as printed indexes segments by the overlap h2, but the text says (h1, h2) = (35, 28) "roughly partitions the time-series into weeks, with adjacent partitions within one day of each other" and that "a single return can potentially belong to 5 different empirical measures". Both statements hold only if consecutive segments start h1 - h2 = 7 hours apart, so h2 is read as the overlap and the step is h1 - h2.

<a id="H2"></a>**H2. Unstated experiment settings.** The synthetic section does not restate (h1, h2), the MK-means moment order, or the number of k-means restarts. Used: (35, 28) from the real-data section, the first p = 4 raw moments for MK-means, and a single run of Algorithm 1 / Algorithm 2 per path with centroids initialised by sampling k segments, as the algorithms are written. Regime-on is the cluster whose members have the higher mean variance. Episode starts are uniform over all placements satisfying s_i + l_i + 2 < s_{i+1}.

<a id="H3"></a>**H3. HMM benchmark.** The paper does not say how its Gaussian HMM was fitted. Two versions are reported: a maximum-likelihood fit (10 starts, EM to convergence, Viterbi path) and `hmmlearn.GaussianHMM()` with its default settings (`n_iter=10`, `tol=1e-2`, `min_covar=1e-3`, `covars_prior=1e-2`). The defaults put a variance floor of 1e-3 on returns whose hourly variance is about 2e-5, which makes the two states indistinguishable; the paper's low HMM accuracies are consistent with that configuration.

## Daniel & Moskowitz

<a id="D1"></a>**D1. Data.** Paper: CRSP common shares on NYSE/AMEX/NASDAQ ranked into deciles on all-firm breakpoints. Here: Ken French's `10_Portfolios_Prior_12_2` (also CRSP-based, value-weighted, prior 12-2 ranking, but NYSE breakpoints), Fama-French `Mkt-RF` + `RF` as the market and `RF` as the risk-free rate. NYSE breakpoints put fewer small stocks in the extreme deciles, so WML is less volatile (27% vs 30% a year) and has a lower mean.

<a id="D2"></a>**D2. Mean forecast column.** Sec. 4 says the mean forecast is "the fitted regression of the WML returns on the interaction between the bear-market indicator and the market variance ... i.e., the regression estimated in the last column of Table 5". The interaction-only regression is column (4); the last column is (5), which also includes the two main effects. Column (4) is used (it matches the verbal description); column (5) is reported alongside.

<a id="D3"></a>**D3. Units of the market variance.** Table 5 does not state them; the international section (Table 8 notes) describes the same variable as "the realized daily market return variance, annualized, over the preceding 126 trading days", which is what is used. With it our estimates (-0.290, -0.336) land next to the paper's (-0.330, -0.397). Table 5 calls it the variance of the market return and Sec. 3.6 of the market excess return; total returns are used (the difference is negligible over 126 days).

<a id="D4"></a>**D4. Sample start.** The paper's regressions start in 1927:07. The bear indicator needs 24 months of market returns, and French's monthly market series starts in 1926:07, so our regressions effectively start in 1928:07 and the strategies in 1929:01.


<a id="D5"></a>**D5. Timing of the GARCH regressor.** Appendix D regresses next-month realised vol on "the GJR-GARCH estimate (sigma_GARCH,t)" without saying which horizon. Used: the one-step-ahead conditional volatility for the first day of the month, which uses only information to the end of the previous month.
