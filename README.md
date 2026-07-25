# Cookie Cats A/B Test — Statistics From Scratch

Most people call scipy.stats.ttest_ind() and move on. I wanted to
actually understand what's inside it — so I implemented Welch's
t-test, chi-square, and z-test from the formulas, then validated
everything against scipy to 1e-6 to prove they're correct.

---

## What I implemented

- **Welch's t-test** — sample mean, variance, standard error,
  t-statistic, and Welch-Satterthwaite degrees of freedom, all by hand
- **Chi-square test** — expected cell counts under the null, statistic
  computed directly from a 2×2 table
- **Two-proportion z-test** — pooled proportion, SE, z-statistic
- **Power analysis** — minimum sample size before running any test,
  the step most A/B projects skip entirely

---

## Validation

15 tests, all passing. The one worth mentioning: for a 2×2 table,
z² must equal the chi-square statistic exactly — mathematically
guaranteed. The test suite verifies this identity holds between two
independent implementations. If either had a bug, this would break.

---

## The finding

| Metric | Result |
|---|---|
| sum_gamerounds | No significant difference between gates |
| 7-day retention | Significant — gate_30 wins |

These point in different directions. Engagement among players who
stayed looks identical. But whether players come back at all is lower
when the gate moves to level 40. Retention is tied to long-term
revenue. Engagement among an already-retained group is not.
Gate stays at 30.

---

## Dataset

Synthetic, calibrated to the real published Cookie Cats statistics
from the original Kaggle dataset. Same N, same retention rates, same
qualitative pattern. Disclosed explicitly — not presented as a real
download.

---

## How to run it

```bash
pip install -r requirements.txt
python scripts/generate_data.py
python -m pytest scripts/test_manual_stats.py -v
python scripts/run_analysis.py
```

---

## Stack

Python · scipy (CDF lookups only) · statsmodels · numpy · pytest
