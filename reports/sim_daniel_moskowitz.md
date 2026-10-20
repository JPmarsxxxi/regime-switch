# Daniel & Moskowitz: simulation ground truth

20 seeds x 80 years of `momentum_panic` (21-day months): daily market returns from a calm/turbulent Markov chain; daily WML with the true monthly mean gamma0 + gamma_int * I_B * sigma^2_m (Table 5 col. 4 values) and GJR-GARCH variance (Appendix D values). The paper's pipeline is run unchanged; the oracle uses the true monthly mean and variance.

## Acceptance criteria (SPEC 18)

| Criterion | Result | Numbers |
|---|---|---|
| 18a interaction CI covers the true -0.397 in >= 80% of seeds | PASS | 100% |
| 18b dyn Sharpe > static Sharpe in >= 80% of seeds | PASS | 100% |
| 18c mean dyn Sharpe >= mean cvol Sharpe | PASS | 1.184 vs 0.818 |
| 18d oracle mean Sharpe >= dyn, cvol, static | PASS | 1.202 vs 1.184 / 0.818 / 0.557 |

## Per-seed summary

|               |    mean |     sd |     min |     max |
|:--------------|--------:|-------:|--------:|--------:|
| gamma_int     | -0.3923 | 0.0934 | -0.6077 | -0.2449 |
| se            |  0.1079 | 0.0347 |  0.0633 |  0.2002 |
| Sharpe dyn    |  1.1842 | 0.1586 |  0.6437 |  1.3793 |
| Sharpe cvol   |  0.8178 | 0.1203 |  0.6680 |  1.0489 |
| Sharpe static |  0.5568 | 0.1982 |  0.0121 |  0.8980 |
| Sharpe oracle |  1.2015 | 0.1007 |  1.0201 |  1.3776 |
| GJR alpha     |  0.1080 | 0.0069 |  0.0919 |  0.1246 |
| GJR beta      |  0.8971 | 0.0048 |  0.8845 |  0.9069 |
| GJR gamma     | -0.0137 | 0.0064 | -0.0240 | -0.0030 |
