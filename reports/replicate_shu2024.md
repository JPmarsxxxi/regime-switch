# Shu, Yu & Mulvey (2024): replication on free data

Data: Yahoo Finance closes (`^GSPC`, `^GDAXI`, `^N225`), 1970-2023; FRED 3-month rates. See DEVIATIONS.md#S1-S5 for every difference from the paper (notably: S&P 500 and Nikkei are price indices, the DAX history starts in 1988 so its test period starts later).

## S&P 500

Test period 1991-01-02 to 2023-12-29.

### Table 3: shifts per year in the online regimes (fixed hyperparameters, 1982-2023)

| model   |   param |   ours |   paper |
|:--------|--------:|-------:|--------:|
| HMM     |       0 |    8.5 |     8.5 |
| HMM     |       2 |    6.6 |     6.6 |
| HMM     |       4 |    4.8 |     4.9 |
| HMM     |       8 |    3.1 |     3.2 |
| HMM     |      20 |    1.9 |     2   |
| JM      |       0 |   15.2 |     9.7 |
| JM      |       5 |    4.2 |     2.7 |
| JM      |      15 |    2.5 |     1.7 |
| JM      |      35 |    1.4 |     0.8 |
| JM      |      70 |    0.9 |     0.5 |
| JM      |     150 |    0.4 |     0.4 |

### Table 4 (delay 1, 10 bp)

|            | B&H ours   | B&H paper   | HMM ours   | HMM paper   | JM ours   | JM paper   |
|:-----------|:-----------|:------------|:-----------|:------------|:----------|:-----------|
| Return     | 8.4%       | 10.2%       | 6.9%       | 8.5%        | 8.5%      | 11.2%      |
| Volatility | 18.2%      | 18.2%       | 11.1%      | 11.3%       | 12.3%     | 13.1%      |
| Sharpe     | 0.40       | 0.48        | 0.44       | 0.54        | 0.53      | 0.68       |
| MDD        | -56.8%     | -55.2%      | -26.4%     | -28.9%      | -34.9%    | -26.6%     |
| Calmar     | 0.13       | 0.16        | 0.18       | 0.21        | 0.19      | 0.33       |
| ES0.05     | -2.7%      | -2.7%       | -1.8%      | -1.8%       | -1.9%     | -2.0%      |
| Turnover   | 0.0%       | 0%          | 160.7%     | 141%        | 36.4%     | 44%        |
| Leverage   | 100.0%     | 100%        | 71.3%      | 72%         | 73.3%     | 80%        |

### Table 5: Sharpe ratio by trading delay

|     |   delay 1 |   delay 5 |   delay 10 | paper (1 / 5 / 10)   |
|:----|----------:|----------:|-----------:|:---------------------|
| B&H |      0.4  |      0.4  |       0.4  | 0.48                 |
| HMM |      0.44 |      0.49 |       0.45 | 0.54 / 0.55 / 0.51   |
| JM  |      0.53 |      0.54 |       0.51 | 0.68 / 0.71 / 0.70   |

JM bear share 26.7%, 24 regime shifts; selected lambda: median 100, distribution {100.0: 0.62, 35.0: 0.14, 150.0: 0.14, 15.0: 0.11}.

## DAX

Test period 2009-01-02 to 2023-12-29.

### Table 4 (delay 1, 10 bp)

|            | B&H ours   | B&H paper   | HMM ours   | HMM paper   | JM ours   | JM paper   |
|:-----------|:-----------|:------------|:-----------|:------------|:----------|:-----------|
| Return     | 8.6%       | 6.8%        | 6.7%       | 6.4%        | 2.2%      | 8.6%       |
| Volatility | 20.6%      | 22.1%       | 14.7%      | 14.0%       | 14.8%     | 16.4%      |
| Sharpe     | 0.49       | 0.30        | 0.49       | 0.35        | 0.19      | 0.44       |
| MDD        | -38.8%     | -72.7%      | -26.5%     | -40.5%      | -34.8%    | -39.4%     |
| Calmar     | 0.26       | 0.09        | 0.27       | 0.12        | 0.08      | 0.18       |
| ES0.05     | -3.1%      | -3.3%       | -2.3%      | -2.2%       | -2.4%     | -2.5%      |
| Turnover   | 0.0%       | 0%          | 218.5%     | 246%        | 119.2%    | 170%       |
| Leverage   | 100.0%     | 100%        | 80.4%      | 73%         | 73.1%     | 84%        |

### Table 5: Sharpe ratio by trading delay

|     |   delay 1 |   delay 5 |   delay 10 | paper (1 / 5 / 10)   |
|:----|----------:|----------:|-----------:|:---------------------|
| B&H |      0.49 |      0.49 |       0.49 | 0.30                 |
| HMM |      0.49 |      0.29 |       0.11 | 0.35 / 0.25 / 0.12   |
| JM  |      0.19 |     -0.12 |       0.23 | 0.44 / 0.38 / 0.29   |

JM bear share 26.9%, 35 regime shifts; selected lambda: median 50, distribution {50.0: 0.39, 35.0: 0.21, 5.0: 0.19, 150.0: 0.19, 70.0: 0.01, 0.0: 0.01}.

## Nikkei 225

Test period 1991-07-01 to 2023-12-29.

### Table 4 (delay 1, 10 bp)

|            | B&H ours   | B&H paper   | HMM ours   | HMM paper   | JM ours   | JM paper   |
|:-----------|:-----------|:------------|:-----------|:------------|:----------|:-----------|
| Return     | 1.2%       | 0.8%        | 1.8%       | 2.5%        | 1.4%      | 4.7%       |
| Volatility | 23.1%      | 23.4%       | 16.2%      | 16.0%       | 12.2%     | 17.1%      |
| Sharpe     | 0.14       | 0.12        | 0.15       | 0.19        | 0.13      | 0.31       |
| MDD        | -72.0%     | -79.1%      | -54.7%     | -48.6%      | -40.8%    | -45.3%     |
| Calmar     | 0.04       | 0.04        | 0.04       | 0.06        | 0.04      | 0.12       |
| ES0.05     | -3.3%      | -3.4%       | -2.5%      | -2.5%       | -2.1%     | -2.6%      |
| Turnover   | 0.0%       | 0%          | 315.7%     | 290%        | 238.4%    | 72%        |
| Leverage   | 100.0%     | 100%        | 69.1%      | 68%         | 40.9%     | 75%        |

### Table 5: Sharpe ratio by trading delay

|     |   delay 1 |   delay 5 |   delay 10 | paper (1 / 5 / 10)   |
|:----|----------:|----------:|-----------:|:---------------------|
| B&H |      0.14 |      0.14 |       0.14 | 0.12                 |
| HMM |      0.15 |      0.1  |       0.11 | 0.19 / 0.11 / 0.07   |
| JM  |      0.13 |      0.06 |       0.18 | 0.31 / 0.27 / 0.24   |

JM bear share 59.3%, 150 regime shifts; selected lambda: median 15, distribution {5.0: 0.28, 35.0: 0.2, 0.0: 0.19, 70.0: 0.14, 50.0: 0.1, 15.0: 0.09, 150.0: 0.0}.

## Headline claims

| Claim | Verdict | Numbers |
|---|---|---|
| S&P 500: JM Sharpe > B&H Sharpe | REPRODUCED | 0.53 vs 0.40 |
| S&P 500: JM Sharpe > HMM Sharpe | REPRODUCED | 0.53 vs 0.44 |
| S&P 500: JM MDD smaller than B&H MDD | REPRODUCED | -34.9% vs -56.8% |
| S&P 500: JM turnover < HMM turnover | REPRODUCED | 36% vs 161% |
| DAX: JM Sharpe > B&H Sharpe | NOT REPRODUCED | 0.19 vs 0.49 |
| DAX: JM Sharpe > HMM Sharpe | NOT REPRODUCED | 0.19 vs 0.49 |
| DAX: JM MDD smaller than B&H MDD | REPRODUCED | -34.8% vs -38.8% |
| DAX: JM turnover < HMM turnover | REPRODUCED | 119% vs 218% |
| Nikkei 225: JM Sharpe > B&H Sharpe | NOT REPRODUCED | 0.13 vs 0.14 |
| Nikkei 225: JM Sharpe > HMM Sharpe | NOT REPRODUCED | 0.13 vs 0.15 |
| Nikkei 225: JM MDD smaller than B&H MDD | REPRODUCED | -40.8% vs -72.0% |
| Nikkei 225: JM turnover < HMM turnover | REPRODUCED | 238% vs 316% |
