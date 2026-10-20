# Daniel & Moskowitz: replication on Ken French data

Data: Ken French's `10_Portfolios_Prior_12_2` (daily and monthly, value-weighted) and Fama-French factors. WML = decile 10 - decile 1. Differences from the paper's CRSP construction are listed in DEVIATIONS.md#D1. Every estimate is full-sample, as in the paper.

## Paper sample (to 2013-03)

Daily data 1926-11-03 to 2013-03-28; strategies from 1929-01-02.

### Table 1, WML row (monthly, annualised %)

|          |   ours |   paper |
|:---------|-------:|--------:|
| r - rf   |  14.75 |   17.90 |
| sigma    |  27.28 |   30.00 |
| alpha    |  18.81 |   22.20 |
| t(alpha) |   6.84 |    7.30 |
| beta     |  -0.53 |   -0.58 |
| SR       |   0.54 |    0.60 |
| sk(m)    |  -6.16 |   -4.70 |

### Table 5 (coefficient (t); const and IB in % per month)

| column   | variable   | ours          | paper (1927:07-2013:03)   |
|:---------|:-----------|:--------------|:--------------------------|
| (1)      | const      | 1.617 (5.9)   | 1.955 (6.6)               |
| (1)      | IB         | -2.468 (-3.8) | -2.626 (-3.8)             |
| (2)      | const      | 2.009 (6.7)   | 2.428 (7.5)               |
| (2)      | var        | -0.290 (-4.9) | -0.330 (-5.1)             |
| (3)      | const      | 2.082 (6.9)   | 2.500 (7.7)               |
| (3)      | IB         | -1.307 (-1.8) | -1.281 (-1.6)             |
| (3)      | var        | -0.234 (-3.5) | -0.275 (-3.8)             |
| (4)      | const      | 1.593 (6.2)   | 1.973 (7.1)               |
| (4)      | int        | -0.336 (-5.3) | -0.397 (-5.7)             |
| (5)      | const      | 1.814 (5.2)   | 2.129 (5.8)               |
| (5)      | IB         | -0.450 (-0.5) | 0.023 (0.0)               |
| (5)      | var        | -0.099 (-0.9) | -0.088 (-0.8)             |
| (5)      | int        | -0.219 (-1.6) | -0.323 (-2.2)             |

### Appendix D: GJR-GARCH on daily WML

|       |       ours |     paper |
|:------|-----------:|----------:|
| mu    |  0.0006461 |  0.00086  |
| omega |  6.342e-07 |  1.17e-06 |
| alpha |  0.09637   |  0.111    |
| gamma | -0.01496   | -0.016    |
| beta  |  0.9111    |  0.896    |

### Appendix D: vol forecast regression (monthly)

|             |   ours coef |   ours t |   paper coef |
|:------------|------------:|---------:|-------------:|
| const       |      0.0005 |   1.9058 |       0.0010 |
| sigma_GARCH |      0.6890 |  19.1889 |       0.6114 |
| sigma_126   |      0.2211 |   6.1473 |       0.2640 |

### Strategy Sharpe ratios (daily, annualised; all scaled to 19% vol)

|        |   mean = Table 5 col. (4) |   mean = col. (5) |
|:-------|--------------------------:|------------------:|
| dyn    |                     1.337 |             1.359 |
| cvol   |                     1.090 |             1.090 |
| static |                     0.528 |             0.528 |

### Table 7 spanning alphas (% per year (t); dyn and cvol scaled to WML vol)

|                 | ours          | paper         |
|:----------------|:--------------|:--------------|
| dyn on Mkt+WML  | 19.43 (10.88) | 23.74 (11.99) |
| dyn on Mkt+cvol | 7.66 (6.68)   | 7.27 (6.86)   |
| cvol on Mkt+WML | 11.41 (10.13) | 14.27 (11.44) |

### Headline claims

| Claim | Verdict | Numbers |
|---|---|---|
| Interaction coefficient (col. 4) negative and significant (t < -2) | REPRODUCED | -0.336 (t = -5.3) |
| Sharpe: dyn > cvol > static | REPRODUCED | 1.337 > 1.090 > 0.528 |
| Table 7 col. 1 alpha positive and significant | REPRODUCED | 19.43% (t = 10.88) |
| Table 7 col. 4: dyn not spanned by cvol (alpha t > 2) | REPRODUCED | 7.66% (t = 6.68) |

## Extended to the latest available month

Daily data 1926-11-03 to 2026-08-31; strategies from 1929-01-02.

### Table 1, WML row (monthly, annualised %)

|          |   ours |   paper |
|:---------|-------:|--------:|
| r - rf   |  13.62 |     nan |
| sigma    |  27.61 |     nan |
| alpha    |  18.04 |     nan |
| t(alpha) |   6.91 |     nan |
| beta     |  -0.53 |     nan |
| SR       |   0.49 |     nan |
| sk(m)    |  -5.58 |     nan |

### Table 5 (coefficient (t); const and IB in % per month)

| column   | variable   | ours          | paper (1927:07-2013:03)   |
|:---------|:-----------|:--------------|:--------------------------|
| (1)      | const      | 1.509 (5.9)   | 1.955 (6.6)               |
| (1)      | IB         | -2.619 (-4.2) | -2.626 (-3.8)             |
| (2)      | const      | 1.899 (6.7)   | 2.428 (7.5)               |
| (2)      | var        | -0.281 (-4.9) | -0.330 (-5.1)             |
| (3)      | const      | 1.978 (6.9)   | 2.500 (7.7)               |
| (3)      | IB         | -1.587 (-2.3) | -1.281 (-1.6)             |
| (3)      | var        | -0.219 (-3.5) | -0.275 (-3.8)             |
| (4)      | const      | 1.485 (6.2)   | 1.973 (7.1)               |
| (4)      | int        | -0.361 (-5.8) | -0.397 (-5.7)             |
| (5)      | const      | 1.670 (5.2)   | 2.129 (5.8)               |
| (5)      | IB         | -0.437 (-0.5) | 0.023 (0.0)               |
| (5)      | var        | -0.075 (-0.8) | -0.088 (-0.8)             |
| (5)      | int        | -0.267 (-2.1) | -0.323 (-2.2)             |

### Appendix D: GJR-GARCH on daily WML

|       |       ours |   paper |
|:------|-----------:|--------:|
| mu    |  0.0006367 |     nan |
| omega |  6.301e-07 |     nan |
| alpha |  0.09399   |     nan |
| gamma | -0.01486   |     nan |
| beta  |  0.9134    |     nan |

### Appendix D: vol forecast regression (monthly)

|             |   ours coef |   ours t |   paper coef |
|:------------|------------:|---------:|-------------:|
| const       |      0.0006 |   2.1490 |          nan |
| sigma_GARCH |      0.6740 |  19.4809 |          nan |
| sigma_126   |      0.2424 |   6.9116 |          nan |

### Strategy Sharpe ratios (daily, annualised; all scaled to 19% vol)

|        |   mean = Table 5 col. (4) |   mean = col. (5) |
|:-------|--------------------------:|------------------:|
| dyn    |                     1.268 |             1.288 |
| cvol   |                     1.004 |             1.004 |
| static |                     0.462 |             0.462 |

### Table 7 spanning alphas (% per year (t); dyn and cvol scaled to WML vol)

|                 | ours          | paper   |
|:----------------|:--------------|:--------|
| dyn on Mkt+WML  | 18.67 (11.14) | -       |
| dyn on Mkt+cvol | 7.88 (7.05)   | -       |
| cvol on Mkt+WML | 10.79 (10.45) | -       |

### Headline claims

| Claim | Verdict | Numbers |
|---|---|---|
| Interaction coefficient (col. 4) negative and significant (t < -2) | REPRODUCED | -0.361 (t = -5.8) |
| Sharpe: dyn > cvol > static | REPRODUCED | 1.268 > 1.004 > 0.462 |
| Table 7 col. 1 alpha positive and significant | REPRODUCED | 18.67% (t = 11.14) |
| Table 7 col. 4: dyn not spanned by cvol (alpha t > 2) | REPRODUCED | 7.88% (t = 7.05) |
