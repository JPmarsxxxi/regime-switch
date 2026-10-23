# WK-means with an HMM head

Extension, not a paper replication (SPEC: Extension). Variant A scores each day's trailing 35-day return distribution by its Wasserstein distance to the two WK-means centroids; variant B is a 2-state HMM whose state densities are kernel density estimates initialised from WK-means and refined by EM. Both refit every 6 months on the trailing 3000 days and are online (P(regime-on) > 0.5 is the signal). Nothing was tuned.

Test windows identical to the cross-generator study: yes.

## Synthetic generators (5 seeds x 30 years each; mean (sd) over seeds)

### Markov-switching Gaussian

| detector                         | bal.acc     | shifts/yr   | latency       | Sharpe      |
|:---------------------------------|:------------|:------------|:--------------|:------------|
| WK-HMM A (Wasserstein emissions) | 0.83 (0.05) | 1.82 (1.10) | 14.30 (8.48)  | 1.13 (0.59) |
| WK-HMM B (empirical-density HMM) | 0.93 (0.02) | 3.65 (1.53) | 3.30 (1.20)   | 1.24 (0.59) |
| Shu JM                           | 0.68 (0.14) | 3.17 (2.10) | 12.60 (13.76) | 0.70 (0.87) |
| Shu HMM                          | 0.93 (0.01) | 1.57 (0.64) | 6.20 (1.75)   | 1.30 (0.62) |
| Nystrup adaptive HMM             | 0.91 (0.03) | 3.92 (0.90) | 3.00 (1.58)   | 1.23 (0.57) |
| Horvath WK-means (a posteriori)  | 0.94 (0.02) | 0.89 (0.58) | 0.20 (0.45)   | 1.32 (0.62) |
| Oracle filter                    | 0.95 (0.02) | 3.33 (1.23) | 3.60 (0.89)   | 1.29 (0.57) |
| Buy and hold                     | -           | 0.00 (0.00) | -             | 0.75 (0.59) |

### Markov-switching Student-t

| detector                         | bal.acc     | shifts/yr    | latency       | Sharpe      |
|:---------------------------------|:------------|:-------------|:--------------|:------------|
| WK-HMM A (Wasserstein emissions) | 0.84 (0.03) | 1.53 (0.93)  | 21.20 (6.29)  | 0.78 (0.42) |
| WK-HMM B (empirical-density HMM) | 0.91 (0.03) | 6.04 (2.42)  | 2.80 (0.84)   | 0.82 (0.43) |
| Shu JM                           | 0.69 (0.10) | 3.76 (2.55)  | 10.40 (11.33) | 0.52 (0.62) |
| Shu HMM                          | 0.80 (0.13) | 5.99 (3.34)  | 3.20 (1.79)   | 0.69 (0.49) |
| Nystrup adaptive HMM             | 0.77 (0.06) | 30.76 (3.47) | 0.80 (0.45)   | 0.36 (0.37) |
| Horvath WK-means (a posteriori)  | 0.95 (0.03) | 0.98 (0.41)  | 0.60 (1.34)   | 1.13 (0.26) |
| Oracle filter                    | 0.90 (0.01) | 7.20 (1.96)  | 2.70 (0.45)   | 0.79 (0.44) |
| Buy and hold                     | -           | 0.00 (0.00)  | -             | 0.50 (0.45) |

### Drifting-parameter HMM

| detector                         | bal.acc     | shifts/yr    | latency       | Sharpe      |
|:---------------------------------|:------------|:-------------|:--------------|:------------|
| WK-HMM A (Wasserstein emissions) | 0.79 (0.06) | 1.66 (0.75)  | 17.50 (1.00)  | 0.96 (0.64) |
| WK-HMM B (empirical-density HMM) | 0.89 (0.02) | 5.19 (1.23)  | 3.40 (1.67)   | 1.08 (0.57) |
| Shu JM                           | 0.67 (0.13) | 3.81 (3.71)  | 10.70 (16.28) | 0.76 (0.59) |
| Shu HMM                          | 0.89 (0.03) | 2.67 (1.16)  | 7.20 (3.03)   | 1.04 (0.58) |
| Nystrup adaptive HMM             | 0.87 (0.05) | 10.16 (1.41) | 2.90 (0.89)   | 0.89 (0.65) |
| Horvath WK-means (a posteriori)  | 0.94 (0.01) | 1.03 (0.34)  | 0.20 (0.45)   | 1.21 (0.56) |
| Oracle filter                    | 0.92 (0.02) | 3.46 (1.12)  | 3.50 (0.50)   | 1.13 (0.55) |
| Buy and hold                     | -           | 0.00 (0.00)  | -             | 0.69 (0.61) |

### Horvath gBm (daily)

| detector                         | bal.acc     | shifts/yr   | latency       | Sharpe       |
|:---------------------------------|:------------|:------------|:--------------|:-------------|
| WK-HMM A (Wasserstein emissions) | 0.76 (0.04) | 3.39 (0.75) | 15.20 (11.64) | -0.04 (0.30) |
| WK-HMM B (empirical-density HMM) | 0.83 (0.06) | 5.70 (2.32) | 9.10 (1.75)   | -0.08 (0.33) |
| Shu JM                           | 0.60 (0.11) | 4.56 (3.99) | 7.00 (8.37)   | -0.26 (0.36) |
| Shu HMM                          | 0.84 (0.05) | 2.51 (1.19) | 20.90 (3.81)  | -0.07 (0.27) |
| Nystrup adaptive HMM             | 0.84 (0.07) | 5.76 (2.31) | 10.60 (1.98)  | -0.03 (0.29) |
| Horvath WK-means (a posteriori)  | 0.91 (0.03) | 1.69 (0.35) | 2.60 (2.97)   | -0.03 (0.38) |
| Buy and hold                     | -           | 0.00 (0.00) | -             | -0.13 (0.24) |

### Horvath Merton (daily)

| detector                         | bal.acc     | shifts/yr   | latency       | Sharpe       |
|:---------------------------------|:------------|:------------|:--------------|:-------------|
| WK-HMM A (Wasserstein emissions) | 0.89 (0.02) | 1.03 (0.25) | 22.10 (1.52)  | 0.35 (0.28)  |
| WK-HMM B (empirical-density HMM) | 0.95 (0.01) | 3.94 (0.47) | 4.60 (1.67)   | 0.57 (0.31)  |
| Shu JM                           | 0.62 (0.06) | 5.15 (2.93) | 15.10 (24.43) | -0.10 (0.36) |
| Shu HMM                          | 0.92 (0.03) | 2.92 (0.67) | 8.90 (3.73)   | 0.45 (0.40)  |
| Nystrup adaptive HMM             | 0.92 (0.05) | 5.72 (1.08) | 4.10 (1.47)   | 0.48 (0.33)  |
| Horvath WK-means (a posteriori)  | 0.97 (0.01) | 0.98 (0.19) | 1.40 (1.95)   | 0.63 (0.31)  |
| Buy and hold                     | -           | 0.00 (0.00) | -             | 0.05 (0.28)  |

### Momentum-panic market

| detector                         | bal.acc     | shifts/yr   | latency       | Sharpe      |
|:---------------------------------|:------------|:------------|:--------------|:------------|
| WK-HMM A (Wasserstein emissions) | 0.85 (0.09) | 1.62 (1.36) | 22.50 (2.92)  | 0.79 (0.52) |
| WK-HMM B (empirical-density HMM) | 0.95 (0.02) | 2.26 (1.45) | 4.00 (0.94)   | 1.00 (0.54) |
| Shu JM                           | 0.61 (0.14) | 1.34 (0.57) | 22.60 (22.82) | 0.61 (0.64) |
| Shu HMM                          | 0.94 (0.03) | 0.93 (0.50) | 7.20 (1.64)   | 0.95 (0.52) |
| Nystrup adaptive HMM             | 0.86 (0.20) | 1.98 (2.20) | 3.30 (2.08)   | 1.03 (0.48) |
| Horvath WK-means (a posteriori)  | 0.97 (0.01) | 0.64 (0.42) | 1.00 (1.73)   | 0.94 (0.47) |
| Oracle filter                    | 0.97 (0.01) | 1.57 (0.91) | 4.60 (1.39)   | 0.94 (0.51) |
| Buy and hold                     | -           | 0.00 (0.00) | -             | 0.71 (0.63) |

### Summary

| generator                  |   A acc |   B acc |   Shu HMM acc |   WK a posteriori acc |   A Sharpe |   B Sharpe |   Shu HMM Sharpe |   B&H Sharpe |
|:---------------------------|--------:|--------:|--------------:|----------------------:|-----------:|-----------:|-----------------:|-------------:|
| Markov-switching Gaussian  |    0.83 |    0.93 |          0.93 |                  0.94 |       1.13 |       1.24 |             1.30 |         0.75 |
| Markov-switching Student-t |    0.84 |    0.91 |          0.80 |                  0.95 |       0.78 |       0.82 |             0.69 |         0.50 |
| Drifting-parameter HMM     |    0.79 |    0.89 |          0.89 |                  0.94 |       0.96 |       1.08 |             1.04 |         0.69 |
| Horvath gBm (daily)        |    0.76 |    0.83 |          0.84 |                  0.91 |      -0.04 |      -0.08 |            -0.07 |        -0.13 |
| Horvath Merton (daily)     |    0.89 |    0.95 |          0.92 |                  0.97 |       0.35 |       0.57 |             0.45 |         0.05 |
| Momentum-panic market      |    0.85 |    0.95 |          0.94 |                  0.97 |       0.79 |       1.00 |             0.95 |         0.71 |

## Real data, point-in-time (0/1 strategy, 1-day delay, 10 bp, Shu test periods)

### S&P 500 (1991-01-01 to 2023-12-29)

|            | B&H    | HMM    | JM     | WK-HMM A (Wasserstein emissions)   | WK-HMM B (empirical-density HMM)   |
|:-----------|:-------|:-------|:-------|:-----------------------------------|:-----------------------------------|
| Return     | 8.4%   | 6.9%   | 8.5%   | 7.4%                               | 7.3%                               |
| Volatility | 18.2%  | 11.1%  | 12.3%  | 10.0%                              | 10.6%                              |
| Sharpe     | 0.40   | 0.44   | 0.53   | 0.51                               | 0.49                               |
| MDD        | -56.8% | -26.4% | -34.9% | -22.2%                             | -25.3%                             |
| Calmar     | 0.13   | 0.18   | 0.19   | 0.23                               | 0.20                               |
| ES0.05     | -2.7%  | -1.8%  | -1.9%  | -1.6%                              | -1.7%                              |
| Turnover   | 0.0%   | 160.7% | 36.4%  | 71.2%                              | 433.5%                             |
| Leverage   | 100.0% | 71.3%  | 73.3%  | 63.0%                              | 71.2%                              |
| Shifts/yr  | nan    | 3.21   | 0.73   | 1.39                               | 8.67                               |

### DAX (2009-01-01 to 2023-12-29)

|            | B&H    | HMM    | JM     | WK-HMM A (Wasserstein emissions)   | WK-HMM B (empirical-density HMM)   |
|:-----------|:-------|:-------|:-------|:-----------------------------------|:-----------------------------------|
| Return     | 8.6%   | 6.7%   | 2.2%   | 5.0%                               | 6.4%                               |
| Volatility | 20.6%  | 14.7%  | 14.8%  | 14.3%                              | 15.9%                              |
| Sharpe     | 0.49   | 0.49   | 0.19   | 0.38                               | 0.45                               |
| MDD        | -38.8% | -26.5% | -34.8% | -31.6%                             | -26.6%                             |
| Calmar     | 0.26   | 0.27   | 0.08   | 0.17                               | 0.27                               |
| ES0.05     | -3.1%  | -2.3%  | -2.4%  | -2.3%                              | -2.5%                              |
| Turnover   | 0.0%   | 218.5% | 119.2% | 76.1%                              | 278.1%                             |
| Leverage   | 100.0% | 80.4%  | 73.1%  | 73.5%                              | 85.3%                              |
| Shifts/yr  | nan    | 4.30   | 2.32   | 1.52                               | 5.50                               |

### Nikkei 225 (1991-07-01 to 2023-12-29)

|            | B&H    | HMM    | JM     | WK-HMM A (Wasserstein emissions)   | WK-HMM B (empirical-density HMM)   |
|:-----------|:-------|:-------|:-------|:-----------------------------------|:-----------------------------------|
| Return     | 1.2%   | 1.8%   | 1.4%   | 2.1%                               | -1.5%                              |
| Volatility | 23.1%  | 16.2%  | 12.2%  | 14.1%                              | 16.4%                              |
| Sharpe     | 0.14   | 0.15   | 0.13   | 0.17                               | -0.05                              |
| MDD        | -72.0% | -54.7% | -40.8% | -52.9%                             | -77.8%                             |
| Calmar     | 0.04   | 0.04   | 0.04   | 0.05                               | -0.01                              |
| ES0.05     | -3.3%  | -2.5%  | -2.1%  | -2.3%                              | -2.5%                              |
| Turnover   | 0.0%   | 315.7% | 238.4% | 135.8%                             | 712.0%                             |
| Leverage   | 100.0% | 69.1%  | 40.9%  | 54.8%                              | 72.9%                              |
| Shifts/yr  | nan    | 6.28   | 5.43   | 2.72                               | 14.24                              |
