# Shu, Yu & Mulvey (2024): simulation ground truth

Paths: `markov_switching_gaussian` (benchmark parameters in `sim.py`), 20 seeds x 30 years; robustness: `markov_switching_t` (df=4), 10 seeds. Full paper pipeline: 3000-day training, 6-monthly JM refits, daily HMM refits, 8-year CV, 1-day delay, 10 bp costs, rf = 0. Scores are over the out-of-sample test period only. 'Oracle filter' is the forward filter run with the true parameters.

## Acceptance criteria (SPEC 13-14)

| Criterion | Result | Numbers |
|---|---|---|
| 13a JM bal.acc >= 90% of oracle filter | FAIL | 0.668 vs 0.9 x 0.941 |
| 13b JM shifts/yr < lambda=0 shifts/yr | PASS | 2.39 vs 14.36 |
| 13c JM 0/1 Sharpe >= B&H Sharpe | PASS | 0.826 vs 0.725 |
| 14a JM shifts/yr < HMM (k=0) shifts/yr | FAIL | 2.39 vs 2.31 |
| 14b JM turnover < HMM turnover | FAIL | 1.205 vs 0.892 |

Reported (not a criterion): JM Sharpe 0.826 vs HMM Sharpe 1.124 -> HMM higher.

## Gaussian paths

|                         |   mean |     sd |    min |     max |
|:------------------------|-------:|-------:|-------:|--------:|
| truth shifts/yr         |  1.522 |  0.535 |  0.684 |   2.507 |
| JM bal.acc              |  0.668 |  0.107 |  0.461 |   0.839 |
| JM shifts/yr            |  2.388 |  2.100 |  0.456 |   9.346 |
| JM latency              |  9.150 | 11.094 |  0.000 |  35.000 |
| HMM bal.acc             |  0.923 |  0.025 |  0.864 |   0.963 |
| HMM shifts/yr           |  1.767 |  0.701 |  0.456 |   3.419 |
| HMM latency             |  5.775 |  2.042 |  2.000 |   9.000 |
| JM lam=0 bal.acc        |  0.695 |  0.075 |  0.561 |   0.868 |
| JM lam=0 shifts/yr      | 14.361 |  3.316 |  7.864 |  22.567 |
| JM lam=0 latency        |  4.400 |  4.119 |  0.000 |  13.500 |
| HMM k=0 bal.acc         |  0.944 |  0.017 |  0.911 |   0.974 |
| HMM k=0 shifts/yr       |  2.314 |  0.735 |  1.368 |   3.647 |
| HMM k=0 latency         |  4.050 |  1.366 |  1.000 |   6.000 |
| Oracle filter bal.acc   |  0.941 |  0.018 |  0.909 |   0.977 |
| Oracle filter shifts/yr |  3.260 |  1.268 |  1.140 |   6.155 |
| Oracle filter latency   |  3.250 |  1.164 |  1.000 |   5.000 |
| B&H Sharpe              |  0.725 |  0.388 | -0.039 |   1.477 |
| B&H turnover            |  0.000 |  0.000 |  0.000 |   0.000 |
| JM Sharpe               |  0.826 |  0.469 |  0.006 |   1.947 |
| JM turnover             |  1.205 |  1.051 |  0.228 |   4.673 |
| HMM Sharpe              |  1.124 |  0.351 |  0.547 |   2.199 |
| HMM turnover            |  0.892 |  0.361 |  0.228 |   1.710 |
| Oracle Sharpe           |  1.120 |  0.355 |  0.596 |   2.156 |
| JM lambda (median)      | 54.500 | 55.935 |  0.000 | 150.000 |
| HMM k (median)          |  3.200 |  3.002 |  0.000 |   8.000 |

## Student-t paths

|                         |   mean |     sd |    min |    max |
|:------------------------|-------:|-------:|-------:|-------:|
| truth shifts/yr         |  1.607 |  0.536 |  0.798 |  2.507 |
| JM bal.acc              |  0.639 |  0.104 |  0.445 |  0.841 |
| JM shifts/yr            |  4.103 |  2.317 |  0.912 |  8.092 |
| JM latency              | 11.400 | 13.117 |  0.000 | 35.000 |
| HMM bal.acc             |  0.788 |  0.117 |  0.572 |  0.906 |
| HMM shifts/yr           |  7.716 |  3.623 |  1.824 | 13.221 |
| HMM latency             |  3.650 |  2.583 |  0.000 |  9.500 |
| JM lam=0 bal.acc        |  0.688 |  0.084 |  0.493 |  0.809 |
| JM lam=0 shifts/yr      | 13.084 |  2.425 |  9.346 | 17.552 |
| JM lam=0 latency        |  4.900 |  5.990 |  0.000 | 17.000 |
| HMM k=0 bal.acc         |  0.814 |  0.080 |  0.673 |  0.894 |
| HMM k=0 shifts/yr       | 19.490 |  4.206 | 11.967 | 25.189 |
| HMM k=0 latency         |  1.350 |  0.944 |  0.000 |  3.000 |
| Oracle filter bal.acc   |  0.886 |  0.036 |  0.797 |  0.921 |
| Oracle filter shifts/yr |  8.092 |  1.740 |  4.901 | 10.030 |
| Oracle filter latency   |  2.700 |  0.856 |  1.000 |  4.000 |
| B&H Sharpe              |  0.612 |  0.423 | -0.106 |  1.165 |
| B&H turnover            |  0.000 |  0.000 |  0.000 |  0.000 |
| JM Sharpe               |  0.538 |  0.350 | -0.215 |  1.086 |
| JM turnover             |  2.063 |  1.158 |  0.456 |  4.046 |
| HMM Sharpe              |  0.798 |  0.354 |  0.030 |  1.272 |
| HMM turnover            |  3.858 |  1.808 |  0.912 |  6.611 |
| Oracle Sharpe           |  0.900 |  0.350 |  0.204 |  1.237 |
| JM lambda (median)      | 32.500 | 28.112 |  5.000 | 70.000 |
| HMM k (median)          |  6.600 |  5.582 |  0.000 | 20.000 |
