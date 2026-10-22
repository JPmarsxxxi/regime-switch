# Cross-generator study: every detector on every generator

5 seeds x 30 years of daily returns per generator; every detector run as its module implements the paper (SPEC: cross-generator study). Scores are over the Shu out-of-sample test window of each path (about 9 years). 0/1 strategy: bear -> cash, 1-day delay, 10 bp one-way, rf = 0. WK-means labels are a posteriori (the clustering sees the whole path), every other detector is online.

## Acceptance criteria (SPEC 26-27)

| Criterion | Result | Numbers |
|---|---|---|
| 26 every detector produced a signal on >= 99% of every test window | PASS | 0 short paths |
| 27 Markov-switching Gaussian: no online detector beats the oracle by > 0.02 | PASS | best online 0.931 vs oracle 0.946 |
| 27 Markov-switching Student-t: no online detector beats the oracle by > 0.02 | PASS | best online 0.798 vs oracle 0.898 |
| 27 Drifting-parameter HMM: no online detector beats the oracle by > 0.02 | PASS | best online 0.891 vs oracle 0.920 |
| 27 Momentum-panic market: no online detector beats the oracle by > 0.02 | PASS | best online 0.943 vs oracle 0.965 |

## Markov-switching Gaussian

True shifts/yr 1.37; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr   | latency       | Sharpe      |
|:--------------------------------|:------------|:------------|:--------------|:------------|
| Shu JM                          | 0.68 (0.14) | 3.17 (2.10) | 12.60 (13.76) | 0.70 (0.87) |
| Shu HMM                         | 0.93 (0.01) | 1.57 (0.64) | 6.20 (1.75)   | 1.30 (0.62) |
| Nystrup adaptive HMM            | 0.91 (0.03) | 3.92 (0.90) | 3.00 (1.58)   | 1.23 (0.57) |
| Horvath WK-means (a posteriori) | 0.94 (0.02) | 0.89 (0.58) | 0.20 (0.45)   | 1.32 (0.62) |
| D&M bear indicator              | 0.48 (0.07) | 0.57 (0.46) | 0.00 (0.00)   | 0.61 (0.65) |
| Oracle filter                   | 0.95 (0.02) | 3.33 (1.23) | 3.60 (0.89)   | 1.29 (0.57) |
| Perfect hindsight               | 1.00 (0.00) | 1.37 (0.59) | 0.00 (0.00)   | 1.33 (0.59) |
| Buy and hold                    | -           | 0.00 (0.00) | -             | 0.75 (0.59) |

## Markov-switching Student-t

True shifts/yr 1.37; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr    | latency       | Sharpe      |
|:--------------------------------|:------------|:-------------|:--------------|:------------|
| Shu JM                          | 0.69 (0.10) | 3.76 (2.55)  | 10.40 (11.33) | 0.52 (0.62) |
| Shu HMM                         | 0.80 (0.13) | 5.99 (3.34)  | 3.20 (1.79)   | 0.69 (0.49) |
| Nystrup adaptive HMM            | 0.77 (0.06) | 30.76 (3.47) | 0.80 (0.45)   | 0.36 (0.37) |
| Horvath WK-means (a posteriori) | 0.95 (0.03) | 0.98 (0.41)  | 0.60 (1.34)   | 1.13 (0.26) |
| D&M bear indicator              | 0.57 (0.12) | 0.93 (0.68)  | 1.50 (3.35)   | 0.50 (0.31) |
| Oracle filter                   | 0.90 (0.01) | 7.20 (1.96)  | 2.70 (0.45)   | 0.79 (0.44) |
| Perfect hindsight               | 1.00 (0.00) | 1.37 (0.59)  | 0.00 (0.00)   | 1.13 (0.25) |
| Buy and hold                    | -           | 0.00 (0.00)  | -             | 0.50 (0.45) |

## Drifting-parameter HMM

True shifts/yr 1.37; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr    | latency       | Sharpe      |
|:--------------------------------|:------------|:-------------|:--------------|:------------|
| Shu JM                          | 0.67 (0.13) | 3.81 (3.71)  | 10.70 (16.28) | 0.76 (0.59) |
| Shu HMM                         | 0.89 (0.03) | 2.67 (1.16)  | 7.20 (3.03)   | 1.04 (0.58) |
| Nystrup adaptive HMM            | 0.87 (0.05) | 10.16 (1.41) | 2.90 (0.89)   | 0.89 (0.65) |
| Horvath WK-means (a posteriori) | 0.94 (0.01) | 1.03 (0.34)  | 0.20 (0.45)   | 1.21 (0.56) |
| D&M bear indicator              | 0.48 (0.08) | 0.75 (0.47)  | 4.50 (10.06)  | 0.59 (0.71) |
| Oracle filter                   | 0.92 (0.02) | 3.46 (1.12)  | 3.50 (0.50)   | 1.13 (0.55) |
| Perfect hindsight               | 1.00 (0.00) | 1.37 (0.59)  | 0.00 (0.00)   | 1.26 (0.56) |
| Buy and hold                    | -           | 0.00 (0.00)  | -             | 0.69 (0.61) |

## Horvath gBm (daily)

True shifts/yr 1.12; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr   | latency      | Sharpe       |
|:--------------------------------|:------------|:------------|:-------------|:-------------|
| Shu JM                          | 0.60 (0.11) | 4.56 (3.99) | 7.00 (8.37)  | -0.26 (0.36) |
| Shu HMM                         | 0.84 (0.05) | 2.51 (1.19) | 20.90 (3.81) | -0.07 (0.27) |
| Nystrup adaptive HMM            | 0.84 (0.07) | 5.76 (2.31) | 10.60 (1.98) | -0.03 (0.29) |
| Horvath WK-means (a posteriori) | 0.91 (0.03) | 1.69 (0.35) | 2.60 (2.97)  | -0.03 (0.38) |
| D&M bear indicator              | 0.51 (0.05) | 0.93 (0.28) | 5.80 (9.36)  | -0.10 (0.36) |
| Perfect hindsight               | 1.00 (0.00) | 1.12 (0.31) | 0.00 (0.00)  | 0.04 (0.25)  |
| Buy and hold                    | -           | 0.00 (0.00) | -            | -0.13 (0.24) |

## Horvath Merton (daily)

True shifts/yr 1.12; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr   | latency       | Sharpe       |
|:--------------------------------|:------------|:------------|:--------------|:-------------|
| Shu JM                          | 0.62 (0.06) | 5.15 (2.93) | 15.10 (24.43) | -0.10 (0.36) |
| Shu HMM                         | 0.92 (0.03) | 2.92 (0.67) | 8.90 (3.73)   | 0.45 (0.40)  |
| Nystrup adaptive HMM            | 0.92 (0.05) | 5.72 (1.08) | 4.10 (1.47)   | 0.48 (0.33)  |
| Horvath WK-means (a posteriori) | 0.97 (0.01) | 0.98 (0.19) | 1.40 (1.95)   | 0.63 (0.31)  |
| D&M bear indicator              | 0.43 (0.09) | 0.59 (0.35) | 17.80 (39.80) | -0.03 (0.30) |
| Perfect hindsight               | 1.00 (0.00) | 1.12 (0.31) | 0.00 (0.00)   | 0.58 (0.37)  |
| Buy and hold                    | -           | 0.00 (0.00) | -             | 0.05 (0.28)  |

## Momentum-panic market

True shifts/yr 0.75; test window 8.8 years. Mean (sd) over seeds.

| detector                        | bal.acc     | shifts/yr   | latency       | Sharpe      |
|:--------------------------------|:------------|:------------|:--------------|:------------|
| Shu JM                          | 0.61 (0.14) | 1.34 (0.57) | 22.60 (22.82) | 0.61 (0.64) |
| Shu HMM                         | 0.94 (0.03) | 0.93 (0.50) | 7.20 (1.64)   | 0.95 (0.52) |
| Nystrup adaptive HMM            | 0.86 (0.20) | 1.98 (2.20) | 3.30 (2.08)   | 1.03 (0.48) |
| Horvath WK-means (a posteriori) | 0.97 (0.01) | 0.64 (0.42) | 1.00 (1.73)   | 0.94 (0.47) |
| D&M bear indicator              | 0.54 (0.07) | 0.84 (0.97) | 17.40 (38.91) | 0.60 (0.71) |
| Oracle filter                   | 0.97 (0.01) | 1.57 (0.91) | 4.60 (1.39)   | 0.94 (0.51) |
| Perfect hindsight               | 1.00 (0.00) | 0.75 (0.49) | 0.00 (0.00)   | 0.99 (0.48) |
| Buy and hold                    | -           | 0.00 (0.00) | -             | 0.71 (0.63) |

## Summary

| generator                  | best online accuracy   | best online Sharpe   |   online Sharpe |   B&H Sharpe |   hindsight Sharpe |
|:---------------------------|:-----------------------|:---------------------|----------------:|-------------:|-------------------:|
| Markov-switching Gaussian  | Shu HMM                | Shu HMM              |            1.30 |         0.75 |               1.33 |
| Markov-switching Student-t | Shu HMM                | Shu HMM              |            0.69 |         0.50 |               1.13 |
| Drifting-parameter HMM     | Shu HMM                | Shu HMM              |            1.04 |         0.69 |               1.26 |
| Horvath gBm (daily)        | Nystrup adaptive HMM   | Nystrup adaptive HMM |           -0.03 |        -0.13 |               0.04 |
| Horvath Merton (daily)     | Nystrup adaptive HMM   | Nystrup adaptive HMM |            0.48 |         0.05 |               0.58 |
| Momentum-panic market      | Shu HMM                | Nystrup adaptive HMM |            1.03 |         0.71 |               0.99 |

