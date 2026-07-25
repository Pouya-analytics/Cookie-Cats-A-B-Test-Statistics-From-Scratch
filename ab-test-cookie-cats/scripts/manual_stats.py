"""
manual_stats.py
-----------------
Manual implementations of the two-sample independent t-test (Welch's,
unequal variance) and the chi-square test of independence, built
directly from the underlying formulas rather than calling scipy.stats.

WHY THIS EXISTS: calling scipy.stats.ttest_ind() takes one line and
tells you nothing about whether you understand what a t-statistic
actually is. This module computes every intermediate quantity by hand
(group means, pooled/unpooled variance, standard error, degrees of
freedom via Welch-Satterthwaite, the t-statistic itself, and the
two-tailed p-value via the t-distribution's CDF) and is cross-validated
against scipy in test_manual_stats.py, where every test asserts
agreement to within 1e-6.

The only library dependency outside the standard library is
`scipy.stats` -- but ONLY for its `t` and `chi2` distribution objects'
CDF functions (i.e. "look up a number from a known distribution"), not
for the test logic itself. Implementing a numerically stable CDF for
the t-distribution from scratch (Lentz's continued fraction, or a
series expansion of the incomplete beta function) is a real numerical
methods exercise but a separate skill from understanding *what a
t-test computes and why*, which is the actual point of this project.
This boundary is intentional and stated explicitly so it's clear what
was and wasn't reimplemented.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

from scipy import stats as _scipy_stats_for_cdf_lookup_only


# ---------------------------------------------------------------------
# WELCH'S TWO-SAMPLE T-TEST (unequal variances assumed -- the safer
# default; Student's pooled-variance t-test is the special case where
# you assume equal variances, which is rarely actually true)
# ---------------------------------------------------------------------
@dataclass
class TTestResult:
    mean_a: float
    mean_b: float
    var_a: float
    var_b: float
    n_a: int
    n_b: int
    se: float                 # standard error of the difference in means
    t_statistic: float
    degrees_of_freedom: float
    p_value: float
    mean_diff: float


def sample_mean(data: List[float]) -> float:
    return sum(data) / len(data)


def sample_variance(data: List[float]) -> float:
    """Unbiased sample variance: sum((x - mean)^2) / (n - 1)."""
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 observations to compute variance")
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (n - 1)


def welch_ttest(group_a: List[float], group_b: List[float]) -> TTestResult:
    """
    Welch's t-test for two independent samples with possibly unequal
    variances. Every formula below is the textbook formula, computed
    explicitly rather than via a library call:

      t = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)

      Welch-Satterthwaite degrees of freedom:
        df = (var_a/n_a + var_b/n_b)^2
             / [ (var_a/n_a)^2/(n_a-1) + (var_b/n_b)^2/(n_b-1) ]

      Two-tailed p-value: p = 2 * (1 - CDF_t(|t|, df))
        -- this CDF lookup is the one piece delegated to scipy (see
        module docstring for why).
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_a, mean_b = sample_mean(group_a), sample_mean(group_b)
    var_a, var_b = sample_variance(group_a), sample_variance(group_b)

    se_sq_a = var_a / n_a
    se_sq_b = var_b / n_b
    se = math.sqrt(se_sq_a + se_sq_b)

    mean_diff = mean_a - mean_b
    t_statistic = mean_diff / se

    # Welch-Satterthwaite equation for degrees of freedom
    numerator = (se_sq_a + se_sq_b) ** 2
    denominator = (se_sq_a ** 2) / (n_a - 1) + (se_sq_b ** 2) / (n_b - 1)
    df = numerator / denominator

    # Two-tailed p-value from the t-distribution's CDF.
    # (CDF lookup delegated to scipy -- see module docstring.)
    p_value = 2 * (1 - _scipy_stats_for_cdf_lookup_only.t.cdf(abs(t_statistic), df))

    return TTestResult(
        mean_a=mean_a, mean_b=mean_b, var_a=var_a, var_b=var_b,
        n_a=n_a, n_b=n_b, se=se, t_statistic=t_statistic,
        degrees_of_freedom=df, p_value=p_value, mean_diff=mean_diff,
    )


# ---------------------------------------------------------------------
# CHI-SQUARE TEST OF INDEPENDENCE (2x2 contingency table)
# ---------------------------------------------------------------------
@dataclass
class ChiSquareResult:
    observed: Tuple[Tuple[int, int], Tuple[int, int]]
    expected: Tuple[Tuple[float, float], Tuple[float, float]]
    chi2_statistic: float
    degrees_of_freedom: int
    p_value: float


def chi_square_test(table: Tuple[Tuple[int, int], Tuple[int, int]]) -> ChiSquareResult:
    """
    Chi-square test of independence on a 2x2 contingency table:

        table = ((a, b),
                  (c, d))

    where rows = groups (e.g. gate_30, gate_40) and columns = outcome
    (e.g. retained, not retained).

    Expected count under the null (independence) for cell (i,j):
        E_ij = (row_i_total * col_j_total) / grand_total

    Chi-square statistic:
        chi2 = sum( (observed_ij - expected_ij)^2 / expected_ij )

    Degrees of freedom for an r x c table: (r-1)(c-1) -- for 2x2, df=1.

    p-value: upper tail of the chi-square distribution with df degrees
    of freedom, P(X >= chi2_statistic). (CDF lookup delegated to scipy,
    same boundary as the t-test above.)
    """
    (a, b), (c, d) = table
    row1_total, row2_total = a + b, c + d
    col1_total, col2_total = a + c, b + d
    grand_total = row1_total + row2_total

    e_a = row1_total * col1_total / grand_total
    e_b = row1_total * col2_total / grand_total
    e_c = row2_total * col1_total / grand_total
    e_d = row2_total * col2_total / grand_total

    chi2_statistic = (
        (a - e_a) ** 2 / e_a +
        (b - e_b) ** 2 / e_b +
        (c - e_c) ** 2 / e_c +
        (d - e_d) ** 2 / e_d
    )

    df = 1  # (2-1)(2-1) for a 2x2 table

    p_value = 1 - _scipy_stats_for_cdf_lookup_only.chi2.cdf(chi2_statistic, df)

    return ChiSquareResult(
        observed=table,
        expected=((e_a, e_b), (e_c, e_d)),
        chi2_statistic=chi2_statistic,
        degrees_of_freedom=df,
        p_value=p_value,
    )


# ---------------------------------------------------------------------
# Two-proportion z-test (the test most A/B testing tools actually use
# for binary conversion-rate metrics; included alongside chi-square
# since for a 2x2 table they are mathematically equivalent --
# z^2 == chi2_statistic for a 2-group binary comparison -- which the
# tests also verify directly, as a nice internal consistency check)
# ---------------------------------------------------------------------
@dataclass
class ZTestResult:
    p_a: float
    p_b: float
    pooled_p: float
    se: float
    z_statistic: float
    p_value: float


def two_proportion_ztest(successes_a: int, n_a: int, successes_b: int, n_b: int) -> ZTestResult:
    """
    Two-proportion z-test:
        p_a = successes_a / n_a,  p_b = successes_b / n_b
        pooled_p = (successes_a + successes_b) / (n_a + n_b)
        SE = sqrt( pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b) )
        z = (p_a - p_b) / SE
        two-tailed p-value from the standard normal CDF
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    pooled_p = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n_a + 1 / n_b))
    z_statistic = (p_a - p_b) / se
    p_value = 2 * (1 - _scipy_stats_for_cdf_lookup_only.norm.cdf(abs(z_statistic)))
    return ZTestResult(p_a=p_a, p_b=p_b, pooled_p=pooled_p, se=se,
                        z_statistic=z_statistic, p_value=p_value)


# ---------------------------------------------------------------------
# Minimum sample size for a given MDE (minimum detectable effect) --
# the calculation that should happen BEFORE running a test, included
# here because most portfolio A/B projects skip pre-experiment design
# entirely and jump straight to analyzing results.
# ---------------------------------------------------------------------
def required_sample_size_per_group(baseline_rate: float, mde: float,
                                    alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Sample size per group for a two-proportion z-test, using the
    standard normal approximation formula:

        n = ( (z_alpha/2 + z_beta)^2 * (p1(1-p1) + p2(1-p2)) ) / (p1 - p2)^2

    where p1 = baseline_rate, p2 = baseline_rate + mde,
    z_alpha/2 = critical value for two-tailed significance level alpha,
    z_beta = critical value corresponding to desired power.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    z_alpha = _scipy_stats_for_cdf_lookup_only.norm.ppf(1 - alpha / 2)
    z_beta = _scipy_stats_for_cdf_lookup_only.norm.ppf(power)

    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    n = numerator / denominator
    return math.ceil(n)
