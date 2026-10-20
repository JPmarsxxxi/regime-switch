# Horvath, Issa & Muguruza (2021): synthetic replication (Tables 3 and 4)

50 runs per model, (h1, h2) = (35, 28), k = 2, p = 1, hourly mesh 252 x 7, 20 years, 10 bear episodes of half a year. 95% CI = mean +- 1.96 sd / sqrt(runs). 'HMM' is a 2-state Gaussian HMM fitted by maximum likelihood (10 starts, EM to convergence, Viterbi); 'HMM (hmmlearn defaults)' is the same model through `hmmlearn.GaussianHMM()` with its default settings (see DEVIATIONS.md#H3).

## Acceptance criteria (SPEC 17)

| Criterion | Result | Numbers |
|---|---|---|
| 17a WK total accuracy within 5pp of paper (gBm, 90.60%) | PASS | 90.22% |
| 17b WK total accuracy within 5pp of paper (Merton, 91.28%) | FAIL | 98.92% |
| 17c Merton: WK > MK | PASS | 98.92% vs 73.66% |
| 17d Merton: WK > HMM (maximum likelihood) | PASS | 98.92% vs 98.84% |
| 17e Merton: WK > HMM (hmmlearn defaults) | PASS | 98.92% vs 74.39% |

## Table 3: geometric Brownian motion

| Algorithm               | Total           | Regime-on       | Regime-off      | Paper total / on / off   |
|:------------------------|:----------------|:----------------|:----------------|:-------------------------|
| Wasserstein             | 90.22% +- 3.15% | 86.71% +- 2.45% | 91.39% +- 3.41% | 90.60 / 87.24 / 91.72    |
| Moment                  | 93.16% +- 0.26% | 74.44% +- 1.05% | 99.41% +- 0.05% | 93.23 / 74.83 / 99.38    |
| HMM                     | 99.44% +- 0.05% | 98.68% +- 0.15% | 99.69% +- 0.05% | 58.16 / 41.51 / 63.72    |
| HMM (hmmlearn defaults) | 70.41% +- 4.54% | 34.06% +- 8.76% | 82.55% +- 6.36% | -                        |

Centroid moments vs the true measures (mean over runs; MK 'centroid' moments are those of its members):

| Method       |   mean off |   var off |    mean on |    var on |
|:-------------|-----------:|----------:|-----------:|----------:|
| True measure | -1.967e-21 | 2.268e-05 | -3.685e-05 | 5.102e-05 |
| Wasserstein  | -2.009e-05 | 2.059e-05 | -1.346e-05 | 4.341e-05 |
| Moment       | -1.060e-06 | 2.371e-05 | -4.790e-05 | 5.519e-05 |

MMD^2 self-similarity (Def. 1.9, Gaussian kernel sigma = 0.1, 2000 pairs, first 5 runs):

| method      |   within C_off |   within C_on |   between |
|:------------|---------------:|--------------:|----------:|
| Moment      |      9.062e-05 |     1.006e-04 | 1.084e-04 |
| Wasserstein |      8.097e-05 |     9.399e-05 | 9.504e-05 |

## Table 4: Merton jump diffusion

| Algorithm               | Total           | Regime-on       | Regime-off      | Paper total / on / off   |
|:------------------------|:----------------|:----------------|:----------------|:-------------------------|
| Wasserstein             | 98.92% +- 0.04% | 96.87% +- 0.18% | 99.60% +- 0.02% | 91.28 / 86.87 / 92.76    |
| Moment                  | 73.66% +- 1.51% | 7.66% +- 4.16%  | 95.72% +- 3.36% | 66.64 / 27.25 / 79.79    |
| HMM                     | 98.84% +- 0.44% | 96.18% +- 1.77% | 99.72% +- 0.03% | 75.05 / 0.66 / 99.87     |
| HMM (hmmlearn defaults) | 74.39% +- 4.36% | 40.59% +- 7.77% | 85.67% +- 6.07% | -                        |

Centroid moments vs the true measures (mean over runs; MK 'centroid' moments are those of its members):

| Method       |   mean off |   var off |    mean on |    var on |
|:-------------|-----------:|----------:|-----------:|----------:|
| True measure |  7.370e-05 | 2.425e-05 | -3.005e-04 | 1.565e-04 |
| Wasserstein  |  4.499e-05 | 2.101e-05 | -1.101e-04 | 8.489e-05 |
| Moment       |  6.056e-05 | 4.932e-05 | -5.698e-03 | 1.484e-03 |

MMD^2 self-similarity (Def. 1.9, Gaussian kernel sigma = 0.1, 2000 pairs, first 5 runs):

| method      |   within C_off |   within C_on |   between |
|:------------|---------------:|--------------:|----------:|
| Moment      |      4.983e-04 |     1.984e-04 | 1.965e-03 |
| Wasserstein |      2.004e-04 |     2.312e-04 | 2.470e-04 |
