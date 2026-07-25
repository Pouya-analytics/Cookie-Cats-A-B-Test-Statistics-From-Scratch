"""
run_analysis.py
------------------
Runs the actual A/B test analysis on the Cookie Cats dataset using the
manual statistical implementations in manual_stats.py, and prints a
full report. This is the script that produces every number quoted in
the README.
"""
import csv
import os
from manual_stats import welch_ttest, chi_square_test, two_proportion_ztest, required_sample_size_per_group

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cookie_cats.csv")


def load_data(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "userid": int(row["userid"]),
                "version": row["version"],
                "sum_gamerounds": int(row["sum_gamerounds"]),
                "retention_1": row["retention_1"] == "True",
                "retention_7": row["retention_7"] == "True",
            })
    return rows


def main():
    rows = load_data(DATA_PATH)
    gate_30 = [r for r in rows if r["version"] == "gate_30"]
    gate_40 = [r for r in rows if r["version"] == "gate_40"]

    print("=" * 72)
    print("COOKIE CATS A/B TEST -- Manual Statistical Analysis")
    print("=" * 72)
    print(f"\nTotal players: {len(rows):,}  |  gate_30: {len(gate_30):,}  |  gate_40: {len(gate_40):,}")

    # -------------------------------------------------------------
    # TEST 1: sum_gamerounds -- Welch's t-test (continuous metric)
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 1: sum_gamerounds (Welch's two-sample t-test)")
    print("-" * 72)
    rounds_30 = [r["sum_gamerounds"] for r in gate_30]
    rounds_40 = [r["sum_gamerounds"] for r in gate_40]

    t_result = welch_ttest(rounds_30, rounds_40)
    print(f"  gate_30: mean={t_result.mean_a:.3f}  var={t_result.var_a:.2f}  n={t_result.n_a:,}")
    print(f"  gate_40: mean={t_result.mean_b:.3f}  var={t_result.var_b:.2f}  n={t_result.n_b:,}")
    print(f"  mean difference (30 - 40): {t_result.mean_diff:+.4f}")
    print(f"  standard error: {t_result.se:.5f}")
    print(f"  t-statistic: {t_result.t_statistic:.4f}")
    print(f"  degrees of freedom (Welch-Satterthwaite): {t_result.degrees_of_freedom:.1f}")
    print(f"  p-value (two-tailed): {t_result.p_value:.5f}")
    print(f"  --> {'SIGNIFICANT' if t_result.p_value < 0.05 else 'NOT significant'} at alpha=0.05")

    # -------------------------------------------------------------
    # TEST 2: retention_1 -- chi-square test (binary outcome)
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 2: retention_1 (Chi-square test of independence)")
    print("-" * 72)
    ret1_30_yes = sum(1 for r in gate_30 if r["retention_1"])
    ret1_30_no = len(gate_30) - ret1_30_yes
    ret1_40_yes = sum(1 for r in gate_40 if r["retention_1"])
    ret1_40_no = len(gate_40) - ret1_40_yes

    table_ret1 = ((ret1_30_yes, ret1_30_no), (ret1_40_yes, ret1_40_no))
    chi_result_1 = chi_square_test(table_ret1)
    print(f"  gate_30: retained={ret1_30_yes:,}/{len(gate_30):,} ({ret1_30_yes/len(gate_30):.4f})")
    print(f"  gate_40: retained={ret1_40_yes:,}/{len(gate_40):,} ({ret1_40_yes/len(gate_40):.4f})")
    print(f"  chi-square statistic: {chi_result_1.chi2_statistic:.4f}")
    print(f"  degrees of freedom: {chi_result_1.degrees_of_freedom}")
    print(f"  p-value: {chi_result_1.p_value:.5f}")
    print(f"  --> {'SIGNIFICANT' if chi_result_1.p_value < 0.05 else 'NOT significant'} at alpha=0.05")

    # -------------------------------------------------------------
    # TEST 3: retention_7 -- chi-square test (the metric that matters most)
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 3: retention_7 (Chi-square test of independence)")
    print("-" * 72)
    ret7_30_yes = sum(1 for r in gate_30 if r["retention_7"])
    ret7_30_no = len(gate_30) - ret7_30_yes
    ret7_40_yes = sum(1 for r in gate_40 if r["retention_7"])
    ret7_40_no = len(gate_40) - ret7_40_yes

    table_ret7 = ((ret7_30_yes, ret7_30_no), (ret7_40_yes, ret7_40_no))
    chi_result_7 = chi_square_test(table_ret7)
    print(f"  gate_30: retained={ret7_30_yes:,}/{len(gate_30):,} ({ret7_30_yes/len(gate_30):.4f})")
    print(f"  gate_40: retained={ret7_40_yes:,}/{len(gate_40):,} ({ret7_40_yes/len(gate_40):.4f})")
    print(f"  chi-square statistic: {chi_result_7.chi2_statistic:.4f}")
    print(f"  degrees of freedom: {chi_result_7.degrees_of_freedom}")
    print(f"  p-value: {chi_result_7.p_value:.5f}")
    print(f"  --> {'SIGNIFICANT' if chi_result_7.p_value < 0.05 else 'NOT significant'} at alpha=0.05")

    # -------------------------------------------------------------
    # TEST 4: two-proportion z-test on retention_7, cross-checked against chi-square
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("TEST 4: retention_7 (Two-proportion z-test -- cross-check)")
    print("-" * 72)
    z_result = two_proportion_ztest(ret7_30_yes, len(gate_30), ret7_40_yes, len(gate_40))
    print(f"  z-statistic: {z_result.z_statistic:.4f}  (z^2 = {z_result.z_statistic**2:.4f}, "
          f"matches chi-square statistic above: {chi_result_7.chi2_statistic:.4f})")
    print(f"  p-value: {z_result.p_value:.5f}")

    # -------------------------------------------------------------
    # PRE-EXPERIMENT DESIGN: what sample size WOULD have been needed
    # -------------------------------------------------------------
    print("\n" + "-" * 72)
    print("BONUS: Minimum sample size for the retention_7 effect actually observed")
    print("-" * 72)
    baseline = ret7_40_yes / len(gate_40)
    observed_mde = (ret7_30_yes / len(gate_30)) - baseline
    n_required = required_sample_size_per_group(baseline, observed_mde)
    print(f"  Baseline (gate_40) retention_7 rate: {baseline:.4f}")
    print(f"  Observed effect size (MDE): {observed_mde:+.4f}")
    print(f"  Required sample size PER GROUP to detect this effect at 80% power: {n_required:,}")
    print(f"  Actual sample size per group in this experiment: ~{len(gate_30):,} / ~{len(gate_40):,}")
    print(f"  --> The experiment was {'adequately' if min(len(gate_30), len(gate_40)) >= n_required else 'under-'} powered "
          f"to detect this effect size.")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("""
sum_gamerounds shows no significant difference between gates (Test 1) --
moving the gate doesn't change how much players who stick around play.

retention_7 DOES show a statistically significant difference (Test 3),
with gate_30 retaining players at a higher rate than gate_40. This is
the metric that should drive the product decision, not sum_gamerounds:
a non-significant result on one metric does not cancel out a
significant result on a more important metric -- they're answering
different questions, and retention is the one tied to long-term
business value, not engagement-while-still-playing.

Business recommendation: keep the first gate at level 30. Moving it to
level 40 does not improve engagement among retained players and is
associated with a small but statistically significant DECREASE in
7-day retention.
""")


if __name__ == "__main__":
    main()
