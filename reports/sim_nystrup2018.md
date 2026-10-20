# Nystrup, Madsen & Lindström (2018): simulation ground truth

20 seeds. Parameter and MPC checks on `markov_switching_gaussian` (10 years, benchmark parameters); tracking on `drifting_hmm` (16 years, sinusoidal means and vols). Estimator settings as in the paper (N_eff = 260, A = 1/N_eff, first 260 days for initialisation) plus DEVIATIONS.md#N3-N4. MPC: K = 100, kappa = 0.001, no delay. Parameters at the last day are reported with state 1 = the lower-variance state.

## Acceptance criteria (SPEC 15-16)

| Criterion | Result | Numbers |
|---|---|---|
| 15a mu1: |mean - truth| <= 2 sd across seeds | PASS | mean 0.0006585, truth 0.0006, sd 0.000247 |
| 15a mu2: |mean - truth| <= 2 sd across seeds | PASS | mean -0.00031419, truth -0.0008, sd 0.00196 |
| 15a sd1: |mean - truth| <= 2 sd across seeds | PASS | mean 0.0069866, truth 0.007, sd 0.00017 |
| 15a sd2: |mean - truth| <= 2 sd across seeds | PASS | mean 0.016207, truth 0.018, sd 0.00257 |
| 15a g11: |mean - truth| <= 2 sd across seeds | PASS | mean 0.99764, truth 0.996, sd 0.00187 |
| 15a g22: |mean - truth| <= 2 sd across seeds | PASS | mean 0.96832, truth 0.985, sd 0.0194 |
| 15b drifting path: adaptive MAE(mu) < static MLE | FAIL | 7.88e-04 vs 4.66e-04 |
| 15b drifting path: adaptive MAE(sd) < static MLE | FAIL | 3.29e-03 vs 2.44e-03 |
| 15c filtered-state bal.acc >= 90% of oracle filter | PASS | 0.916 vs 0.9 x 0.949 |
| 16a MPC(0,0) Sharpe >= B&H Sharpe (mean) | PASS | 0.918 vs 0.813 |
| 16b MPC(0,0.02) turnover < MPC(0,0) turnover | PASS | 0.23 vs 0.92 |

## Per-seed summary

|                       |       mean |        sd |        min |       max |
|:----------------------|-----------:|----------:|-----------:|----------:|
| mu1                   |  0.0006585 | 0.0002475 |  0.0002919 | 0.001146  |
| mu2                   | -0.0003142 | 0.001956  | -0.00313   | 0.005234  |
| sd1                   |  0.006987  | 0.0001701 |  0.006698  | 0.007316  |
| sd2                   |  0.01621   | 0.002569  |  0.009852  | 0.01956   |
| g11                   |  0.9976    | 0.001873  |  0.9929    | 0.9998    |
| g22                   |  0.9683    | 0.0194    |  0.9131    | 0.9924    |
| days on safety box    |  0         | 0         |  0         | 0         |
| adaptive bal.acc      |  0.9163    | 0.06314   |  0.7405    | 0.9733    |
| oracle bal.acc        |  0.9493    | 0.02142   |  0.9064    | 0.9774    |
| MPC(0,0) Sharpe       |  0.918     | 0.4701    | -0.1445    | 1.763     |
| MPC(0,0) turnover     |  0.9195    | 0.8409    |  0.1114    | 3.232     |
| MPC(0,0) failures     |  0         | 0         |  0         | 0         |
| MPC(0,0.02) Sharpe    |  0.8226    | 0.51      | -0.1445    | 1.694     |
| MPC(0,0.02) turnover  |  0.2285    | 0.1963    |  0.1114    | 0.7801    |
| MPC(0,0.02) failures  |  0         | 0         |  0         | 0         |
| B&H Sharpe            |  0.8134    | 0.4871    | -0.1439    | 1.681     |
| drift MAE adaptive mu |  0.0007884 | 0.0001849 |  0.0004283 | 0.001244  |
| drift MAE static mu   |  0.0004662 | 0.0001655 |  0.0003179 | 0.0008401 |
| drift MAE adaptive sd |  0.003294  | 0.0005812 |  0.002427  | 0.004304  |
| drift MAE static sd   |  0.002437  | 8.733e-05 |  0.002346  | 0.002631  |
