# Cookie Cats A/B Test — Statistics Implemented From Scratch

An analysis of the famous Cookie Cats mobile game A/B test (does moving
the first progression gate from level 30 to level 40 affect player
retention?), where the statistical tests themselves — Welch's t-test,
chi-square test of independence, two-proportion z-test — are implemented
directly from their formulas in plain Python, not called from
`scipy.stats`. Every manual implementation is then validated against
scipy/numpy/statsmodels to prove correctness.

## Why this project exists

Calling `scipy.stats.ttest_ind(a, b)` takes one line and proves nothing
about whether you understand what a t-statistic is, why Welch's version
exists, or what degrees of freedom actually represent. This project
exists to demonstrate the difference between *running* a statistical
test and *understanding* one.

## About the dataset

This project uses a **synthetically generated dataset**, calibrated to
match the real, publicly documented statistics of the Cookie Cats A/B
test dataset (Tactile Entertainment, originally distributed via
[Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)).
I didn't have Kaggle API credentials in the environment this was built
in, so rather than download the file, I built a generator
(`scripts/generate_data.py`) calibrated against real published figures
from independent analyses of the original dataset:

| Statistic | Real dataset (published) | This synthetic dataset |
|---|---|---|
| Total players | 90,189 | 90,189 (exact) |
| Group split | ~44,700 / ~45,489 | Same |
| 7-day retention, gate_30 vs gate_40 | gate_30 leads by ~0.8 pts | gate_30 leads by ~1.0 pts |
| 1-day retention, gate_30 vs gate_40 | gate_30 leads by ~0.6 pts | gate_30 leads by ~0.8 pts |
| t-test on sum_gamerounds | t≈1.81, p≈0.07 (not significant) | t≈1.40, p≈0.16 (not significant — same conclusion) |
| sum_gamerounds distribution | heavily right-skewed | log-normal, right-skewed |

The qualitative finding that matters — **sum_gamerounds shows no
significant difference, but retention_7 does, and gate_30 wins** — is
reproduced correctly. The exact p-values differ from the real dataset
because this is simulated data with real-world sampling noise, not the
original file. If you have Kaggle access, swap `data/cookie_cats.csv`
for the real download (identical column names) and every script here
runs unchanged.

## What's actually implemented from scratch

In `scripts/manual_stats.py`:

- **Welch's two-sample t-test** — sample mean, sample variance, standard
  error of the difference, the t-statistic itself, and the
  Welch-Satterthwaite degrees-of-freedom formula, all computed directly.
- **Chi-square test of independence** — expected cell counts under the
  null, the chi-square statistic, computed directly from a 2×2
  contingency table.
- **Two-proportion z-test** — pooled proportion, standard error, z
  statistic, computed directly.
- **Minimum sample size calculation** (power analysis) — the
  pre-experiment design step almost every portfolio project skips
  entirely, computed from the standard normal-approximation formula.

**The one thing NOT implemented from scratch**, stated explicitly rather
than glossed over: converting a test statistic into a p-value requires
the CDF of the t-distribution / chi-square distribution / normal
distribution, and I use `scipy.stats`'s CDF functions for that lookup
specifically — not for the test logic. Implementing a numerically
stable t-distribution CDF (e.g. via the incomplete beta function) is a
legitimate numerical-methods exercise, but it's a different skill from
understanding what a t-test computes and why, which is this project's
actual point. The boundary is documented in the code, not hidden.

## Proof of correctness

`scripts/test_manual_stats.py` — 15 tests, all passing — validates every
manual function:

```
test_welch_ttest_matches_scipy[...]              PASSED (×4 random group pairs)
test_sample_mean_matches_numpy                    PASSED
test_sample_variance_matches_numpy_ddof1          PASSED
test_ttest_on_known_textbook_example              PASSED
test_chi_square_matches_scipy[...]                PASSED (×4 contingency tables)
test_ztest_matches_statsmodels_or_manual_chi2_equivalence[...]  PASSED (×3)
test_sample_size_matches_statsmodels              PASSED

============================== 15 passed in 2.04s ==============================
```

One test worth highlighting: for a 2×2 table, the two-proportion z-test
and the chi-square test are mathematically equivalent (`z² = χ²`). The
test suite verifies this identity holds between the two *independent*
manual implementations — if either had a bug, this cross-check would
fail even though both were written by hand.

## Results on this dataset

Running `scripts/run_analysis.py`:

| Metric | gate_30 | gate_40 | Test | Statistic | p-value | Significant? |
|---|---|---|---|---|---|---|
| sum_gamerounds (mean) | 46.86 | 45.27 | Welch's t-test | t=1.396 | 0.163 | No |
| retention_1 | 45.06% | 44.25% | Chi-square | χ²=6.02 | 0.014 | Yes |
| retention_7 | 19.09% | 18.08% | Chi-square | χ²=15.21 | <0.001 | Yes |
| retention_7 (z-test cross-check) | — | — | Two-proportion z | z=3.90 | <0.001 | Yes (z²=χ² confirmed) |

**Power analysis:** the minimum sample size per group needed to detect
the observed retention_7 effect (1.0 percentage point) at 80% power was
~23,300. The actual experiment had ~44,700–45,500 per group — meaning
this test was comfortably over-powered for the effect it found, which
is exactly why the result is trustworthy and not a false positive from
an underpowered test getting lucky.

## Interpretation

`sum_gamerounds` and `retention_1` and `retention_7` don't all point the
same way, and that's the actual analytical challenge in this dataset —
not running the tests, but deciding which metric should drive the
decision. `sum_gamerounds` measures engagement among players who are
still around; it says nothing about whether players leave. `retention_7`
directly measures whether the gate placement is driving people away,
which is the metric tied to long-term business value. **Recommendation:
keep the gate at level 30.** Moving it to level 40 produces no
measurable lift in engagement and is associated with a statistically
significant drop in 7-day retention.

## Repo structure

```
.
├── data/
│   └── cookie_cats.csv          # synthetic dataset (see disclosure above)
├── scripts/
│   ├── generate_data.py         # builds the synthetic dataset
│   ├── manual_stats.py          # t-test / chi-square / z-test from scratch
│   ├── test_manual_stats.py     # validates manual_stats.py against scipy/numpy/statsmodels
│   └── run_analysis.py          # runs the full analysis, produces the results table above
├── sql/
│   └── cookie_cats_summary.sql  # SQL descriptive summary (verified against the same data)
└── requirements.txt
```

## How to run it

```bash
pip install -r requirements.txt
python scripts/generate_data.py        # builds data/cookie_cats.csv
python -m pytest scripts/test_manual_stats.py -v   # proves the manual implementations are correct
python scripts/run_analysis.py         # runs the actual analysis, prints the report above
```

## What I'd add with more time

- Bayesian A/B testing comparison (Beta-Binomial model) alongside the
  frequentist tests, to show both schools and where they agree/disagree
- Sequential testing correction, since this dataset's real-world
  context (continuous monitoring during a live experiment) is exactly
  the situation where naive repeated significance testing inflates
  false-positive rates
- Bootstrap confidence intervals as a non-parametric cross-check on the
  t-test result

## Tech stack

Python 3 standard library (the actual test logic) · scipy (CDF lookups
only, see disclosure above) · numpy / statsmodels (used only in the
validation test file as independent cross-checks) · pytest
