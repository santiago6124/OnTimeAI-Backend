"""Test if the fl_date local-timezone grouping is the root cause of NO_PRIOR cases.

For each NO_PRIOR target (no prior leg in DB matching tail+fl_date), check:
  - Does the tail have ANY flight in the DB with scheduled_off in the last 24h
    before the target's scheduled_off? (regardless of fl_date)
  - If yes → fl_date timezone bug (the data is there, code can't find it)
  - If no → real coverage gap

Usage:
    python3 scripts/diagnose_fl_date_bug.py --since 2026-05-04 --until 2026-05-08
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--sample-size", type=int, default=80)
    args = p.parse_args()

    conn = sqlite3.connect("live_data.db")
    targets = pd.read_sql_query(
        f"""SELECT p.fa_flight_id, f.tail_num, f.fl_date,
                   f.op_carrier, f.origin, f.dest, f.scheduled_off_utc
            FROM predictions p
            JOIN flights f ON f.fa_flight_id = p.fa_flight_id
            WHERE p.predicted_at_utc >= ? AND p.predicted_at_utc <= ?
              AND f.tail_num IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?""",
        conn, params=(f"{args.since}T00:00:00", f"{args.until}T23:59:59", args.sample_size),
    )

    # Categorize each target
    n_total = len(targets)
    n_strict_match = 0     # prior found via (tail, fl_date)
    n_relaxed_match = 0    # NO strict match BUT prior found in 24h via UTC
    n_no_prior = 0         # truly no prior

    for _, t in targets.iterrows():
        tail = t["tail_num"]
        fl_date = t["fl_date"]
        sched_off = t["scheduled_off_utc"]

        # Strict (current code logic)
        strict = conn.execute(
            "SELECT 1 FROM flights WHERE tail_num=? AND fl_date=? AND scheduled_off_utc<? LIMIT 1",
            (tail, fl_date, sched_off),
        ).fetchone()
        if strict:
            n_strict_match += 1
            continue

        # Relaxed: same tail, scheduled_off in last 24h, ignore fl_date
        sched_off_ts = pd.to_datetime(sched_off, utc=True, format="ISO8601")
        earliest = (sched_off_ts - pd.Timedelta(hours=24)).isoformat()
        relaxed = conn.execute(
            """SELECT f.fa_flight_id, f.fl_date, f.scheduled_off_utc, a.actual_in_utc
               FROM flights f
               LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
               WHERE f.tail_num=?
                 AND f.scheduled_off_utc < ?
                 AND f.scheduled_off_utc > ?
                 AND a.actual_in_utc IS NOT NULL
               ORDER BY f.scheduled_off_utc DESC LIMIT 1""",
            (tail, sched_off, earliest),
        ).fetchone()
        if relaxed:
            n_relaxed_match += 1
        else:
            n_no_prior += 1

    print(f"Sample of {n_total} predictions\n")
    print("=" * 70)
    print("BREAKDOWN")
    print("=" * 70)
    print(f"  Strict (current code, tail+fl_date):     {n_strict_match:>4} "
          f"({n_strict_match/n_total*100:>5.1f}%)")
    print(f"  Relaxed (tail, last 24h UTC):            {n_relaxed_match:>4} "
          f"({n_relaxed_match/n_total*100:>5.1f}%)")
    print(f"  Truly no prior (real coverage gap):       {n_no_prior:>4} "
          f"({n_no_prior/n_total*100:>5.1f}%)")
    print()

    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    pct_relaxed = n_relaxed_match / n_total
    pct_strict = n_strict_match / n_total
    pct_no = n_no_prior / n_total

    if pct_relaxed > 0.30:
        recoverable = pct_relaxed * 100
        print(f"→ FL_DATE TIMEZONE BUG CONFIRMED.")
        print(f"  {recoverable:.0f}% of NO_PRIOR cases have a prior in the last 24h that")
        print(f"  the current code MISSES because of fl_date local-tz mismatch.")
        print()
        print(f"  Combined with strict matches ({pct_strict*100:.0f}%), TOTAL recoverable")
        print(f"  lineage coverage = {(pct_strict + pct_relaxed)*100:.0f}%")
        print(f"  vs current production = {pct_strict*100:.0f}%")
        print()
        print(f"  FIX: change lineage.py grouping from (TAIL, fl_date) to (TAIL, last 24h UTC)")
        print(f"  Cost: 5 lines of code, $0 AeroAPI, no retraining needed if same flights")
        print(f"        are picked up.")
    elif pct_no > 0.50:
        print(f"→ REAL COVERAGE GAP. {pct_no*100:.0f}% of targets have NO prior at all.")
        print(f"  Multi-hub backfill helps marginally; need broader carrier-hub coverage.")
    else:
        print(f"→ Mixed: {pct_relaxed*100:.0f}% recoverable via fl_date fix, "
              f"{pct_no*100:.0f}% true gap.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
