"""
generate_data.py
------------------
Generates a SYNTHETIC version of the Cookie Cats mobile game A/B test
dataset, calibrated to match the REAL PUBLISHED STATISTICS of the
original dataset (Tactile Entertainment, distributed via Kaggle:
mursideyarkin/mobile-games-ab-testing-cookie-cats).

This is NOT a copy of the real data -- it's a parametrized simulation,
built because Kaggle API credentials were not available in this dev
environment. The real, documented statistics this generator is
calibrated against (verified via independent published analyses of the
real dataset, not invented):

  - Total players: 90,189 (real number, used exactly)
  - Roughly even split between gate_30 (control) and gate_40 (treatment)
  - 1-day retention: gate_30 ~ +0.6 percentage points higher than gate_40
  - 7-day retention: gate_30 ~ +0.8 percentage points higher than gate_40
  - sum_gamerounds is heavily right-skewed (most players play few rounds,
    a long tail plays hundreds); a Welch t-test on rounds played gives
    t~1.81, p~0.07 (not significant) in the real data -- this generator
    targets that same "no significant difference in rounds played, but
    a real difference in 7-day retention" pattern, which is what makes
    this dataset pedagogically useful in the first place: it's a case
    where two different metrics from the same experiment point in
    different directions, forcing a real interpretive decision.

If you have Kaggle API access, replace data/cookie_cats.csv with the
real download (same column names) and every script in this project
runs unchanged.
"""
import csv
import os
import random

random.seed(42)

N_TOTAL = 90189
N_GATE_30 = 44700   # real dataset split is very close to even, slightly
N_GATE_40 = N_TOTAL - N_GATE_30   # uneven due to random assignment, not a stratified design

# Real published 7-day retention rates are approximately:
#   gate_30: ~19.0%   gate_40: ~18.2%   (gate_30 leads by ~0.8 pts)
# Real published 1-day retention rates are approximately:
#   gate_30: ~44.8%   gate_40: ~44.2%   (gate_30 leads by ~0.6 pts)
RETENTION_1_GATE_30 = 0.448
RETENTION_1_GATE_40 = 0.442
RETENTION_7_GATE_30 = 0.190
RETENTION_7_GATE_40 = 0.182


def sample_gamerounds(retained_7: bool) -> int:
    """
    sum_gamerounds is real-world heavily right-skewed: most players play
    very few rounds (many play 0), a minority play a lot. Players who
    are retained at day 7 systematically play far more rounds (they're
    engaged), so the two are correlated -- modeled here with a
    log-normal distribution whose parameters depend on retention status,
    which reproduces the right shape without claiming false precision
    on the exact original values.
    """
    if retained_7:
        # engaged players: higher mean, still long-tailed
        rounds = int(random.lognormvariate(4.2, 1.3))
    else:
        # most non-retained players played very few rounds; many near 0
        rounds = int(random.lognormvariate(1.8, 1.6))
    return max(rounds, 0)


def generate_group(n: int, version: str, ret1_rate: float, ret7_rate: float):
    rows = []
    for i in range(n):
        # retention_7 implies a player was active enough to be measured;
        # retention_1 is generated with a realistic positive correlation
        # to retention_7 (players who come back at day 7 almost always
        # came back at day 1 too) rather than independently, since
        # independent sampling would create an unrealistic dataset where
        # some players "skip" day 1 but return by day 7 at the same rate
        # as everyone else -- not how real retention curves behave.
        retained_7 = random.random() < ret7_rate
        if retained_7:
            retained_1 = True  # if back at day 7, almost certainly was back at day 1
        else:
            # back-calculate a conditional rate so the MARGINAL retention_1
            # rate still matches the target ret1_rate
            p_ret1_given_not_ret7 = max(
                0.0, (ret1_rate - ret7_rate) / (1 - ret7_rate)
            )
            retained_1 = random.random() < p_ret1_given_not_ret7

        rounds = sample_gamerounds(retained_7)
        rows.append({
            "userid": None,  # assigned globally after both groups generated
            "version": version,
            "sum_gamerounds": rounds,
            "retention_1": retained_1,
            "retention_7": retained_7,
        })
    return rows


def build():
    g30 = generate_group(N_GATE_30, "gate_30", RETENTION_1_GATE_30, RETENTION_7_GATE_30)
    g40 = generate_group(N_GATE_40, "gate_40", RETENTION_1_GATE_40, RETENTION_7_GATE_40)
    all_rows = g30 + g40
    random.shuffle(all_rows)

    # real dataset's userid values are arbitrary large integers, not
    # sequential -- mimic that with a random but unique ID space
    used_ids = set()
    for row in all_rows:
        while True:
            uid = random.randint(116, 9999999)
            if uid not in used_ids:
                used_ids.add(uid)
                row["userid"] = uid
                break
    return all_rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "userid", "version", "sum_gamerounds", "retention_1", "retention_7"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "userid": row["userid"],
                "version": row["version"],
                "sum_gamerounds": row["sum_gamerounds"],
                "retention_1": str(row["retention_1"]),
                "retention_7": str(row["retention_7"]),
            })


if __name__ == "__main__":
    print("Generating synthetic Cookie Cats A/B test dataset...")
    rows = build()
    print(f"  Total players: {len(rows):,}")

    n_30 = sum(1 for r in rows if r["version"] == "gate_30")
    n_40 = sum(1 for r in rows if r["version"] == "gate_40")
    print(f"  gate_30: {n_30:,}  |  gate_40: {n_40:,}")

    ret1_30 = sum(1 for r in rows if r["version"] == "gate_30" and r["retention_1"]) / n_30
    ret1_40 = sum(1 for r in rows if r["version"] == "gate_40" and r["retention_1"]) / n_40
    ret7_30 = sum(1 for r in rows if r["version"] == "gate_30" and r["retention_7"]) / n_30
    ret7_40 = sum(1 for r in rows if r["version"] == "gate_40" and r["retention_7"]) / n_40

    print(f"  retention_1: gate_30={ret1_30:.4f}  gate_40={ret1_40:.4f}  (diff={ret1_30-ret1_40:+.4f})")
    print(f"  retention_7: gate_30={ret7_30:.4f}  gate_40={ret7_40:.4f}  (diff={ret7_30-ret7_40:+.4f})")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cookie_cats.csv")
    write_csv(rows, out_path)
    print(f"\nWritten to: {os.path.abspath(out_path)}")
