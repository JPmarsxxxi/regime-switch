# Deviations from the papers

Every place the code departs from a paper, or fills a gap the paper leaves open, is listed here with the reason. Code sites carry `# see DEVIATIONS.md#<id>`.

## Shu, Yu & Mulvey (2024)

<a id="S1"></a>**S1. Data.** Paper: Bloomberg daily total-return indices (S&P 500, DAX, Nikkei 225) 1970-2023, risk-free = each country's 3-month T-bill yield from GFD. Here: Yahoo Finance closes. `^GSPC` and `^N225` are price indices (dividends excluded, so B&H and strategy returns are lower by roughly the dividend yield while invested); `^GDAXI` is a total-return index but only starts 1987-12-30, so the DAX test period starts in 2008 instead of 1990 (12 years training + 8 years validation). Risk-free: FRED `DTB3` (US, daily), `IR3TIB01DEM156N` (Germany) and `IR3TIB01JPM156N` (Japan) 3-month interbank rates (monthly, forward-filled), converted as yield/252 per trading day.

<a id="S2"></a>**S2. Candidate grids.** The paper does not list the candidate jump penalties or median-filter lengths searched by its cross-validation. Used: lambda in {0, 5, 15, 35, 50, 70, 100, 150} (the Table 3 values plus 50 and 100, which the text calls "typical") and k in {0, 2, 4, 6, 8, 20} (the Table 3 values plus Bulla et al.'s original k = 6).

<a id="S3"></a>**S3. Feature burn-in.** The paper does not say how the start of the EWM features is handled. The first rows have a near-zero downside deviation and Sortino ratios in the hundreds; left in, they capture one of the two centroids. The first 252 days after the first finite feature row are dropped before any fit.

<a id="S4"></a>**S4. HMM fitting.** The paper uses hmmlearn with unspecified settings; hmmlearn's defaults (`n_iter=10`, `tol=1e-2`, `covars_prior=1e-2`, `min_covar=1e-3`) are sized for unit-scale data and would stop EM early and bias variances of daily returns upward. Here: a numba Baum-Welch (`regime_switch/hmm.py`, cross-checked against hmmlearn with the priors switched off in `tests/test_hmm.py`) run to convergence (`tol=1e-8`, at most 500 iterations), 10 starts with k-means++ means, hmmlearn's default starting values for the other parameters. Same model, same likelihood, same Viterbi decoding.

<a id="S5"></a>**S5. Refit calendar.** "Updated every six months": refits happen at the end of the last trading day of June and December, on the 3000 days ending that day; the fit is used from that day's inference onward.

## Nystrup, Madsen & Lindström (2018)

<a id="N1"></a>**N1. Data.** Paper: net total-return indices (MSCI World 1997-2015; S&P 500, TOPIX, DAX, FTSE 1984-2015; MSCI EM 1988-2015). MSCI World, TOPIX and MSCI EM have no free daily history. Here: Yahoo `^GSPC` (price index) stands in for the Table 1 analysis, and Table 3 is reproduced for S&P 500 (price), FTSE (price) and DAX (total return, from 1988 instead of 1984).

<a id="N2"></a>**N2. Parametrisation.** The paper says constrained parameters must be transformed but not how. Used: means untransformed, log variances, logit of the two staying probabilities. Initial distribution of the score recursion: the stationary distribution of the current transition matrix.

<a id="N3"></a>**N3. Initialisation.** "The first 260 observations were used for initialization" is not specified further. A maximum-likelihood fit on those 260 days is not usable: on the S&P 500 in 1984 every start converges to a non-persistent "spike" state (P(stay) about 0.15 capturing a handful of large up days), from which the recursion degenerates. Used instead: a neutral persistent start (both means at the sample mean, variances at 0.5x and 2x the sample variance, P(stay) = 0.98), run through the recursion over those first 260 days, with the Fisher information started from the mean outer product of the scores.

<a id="N4"></a>**N4. Stabilisation of the recursive estimator.** With the Fisher information updated as the exponentially weighted mean of outer products of scores, a regime that goes unvisited for a year or more loses its information, the Newton steps for its parameters blow up and the state collapses (variance to zero, both staying probabilities to one). This happens on the S&P 500 in 1993-95 and on simulated paths. Two minimal safeguards, neither in the paper: (i) after each step the parameters are projected onto a box (daily means within +-1%, daily variances within [1e-8, 1e-2], staying probabilities within [0.5, 0.9999]); (ii) the information matrix used in the step is damped by `0.1 * diag(I_init)` (Levenberg-Marquardt style). The damping factor was chosen on simulation seeds 100-104, which are not used in any reported result, and was not tuned on real data.

<a id="N5"></a>**N5. Rule-based RBAA benchmark.** Table 2's RBAA comparison depends on the rule of Nystrup et al. (2017) ("probability that a regime change has occurred exceeds 0.9998"), which is not specified in this paper. It is not reproduced. The SAA comparison (monthly rebalanced to MPC(0,0)'s average stock weight) is.
