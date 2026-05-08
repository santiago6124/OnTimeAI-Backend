"""Why is prev_arr_delay_tail NaN for 91.5% of live predictions?

Two possible causes:
  A) Coverage gap — the previous same-tail same-day leg doesn't exist in our actuals
     table at all. Cause: not pulling enough hubs / wrong time windows.
  B) Observability rule — the prior leg exists but its actual arrival time is AFTER
     the current flight's scheduled departure. Cause: tight rotations + rule by design.

This script samples 30 predictions, finds each one's "expected prior leg" in the DB
(if any), and reports the breakdown.

Usage:
    python3 scripts/diagnose_lineage_root_cause.py --since 2026-05-04 --until 2026-05-08
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--sample-size", type=int, default=50)
    args = p.parse_args()

    conn = sqlite3.connect("live_data.db")

    # Load a sample of predictions with their flight metadata
    targets = pd.read_sql_query(
        f"""SELECT p.fa_flight_id, f.tail_num, f.fl_date, f.op_carrier,
                   f.origin, f.dest, f.scheduled_off_utc, f.scheduled_in_utc
            FROM predictions p
            JOIN flights f ON f.fa_flight_id = p.fa_flight_id
            WHERE p.predicted_at_utc >= ? AND p.predicted_at_utc <= ?
              AND f.tail_num IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?""",
        conn, params=(f"{args.since}T00:00:00", f"{args.until}T23:59:59", args.sample_size),
    )
    if targets.empty:
        print("No targets found in window")
        return 1

    print(f"Sampled {len(targets)} predictions with valid TAIL_NUM\n")

    n_have_prior_in_actuals = 0
    n_observable = 0
    n_unobservable = 0
    n_no_prior = 0

    case_breakdown = []

    for _, target in targets.iterrows():
        tail = target["tail_num"]
        fl_date = target["fl_date"]
        sched_off = pd.to_datetime(target["scheduled_off_utc"], utc=True, format="ISO8601")

        # Find prior leg = same tail + same fl_date + scheduled before this one
        prior = pd.read_sql_query(
            """SELECT f.fa_flight_id, f.scheduled_off_utc, f.scheduled_in_utc,
                      a.actual_in_utc, a.arr_delay_min
               FROM flights f
               LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
               WHERE f.tail_num = ?
                 AND f.fl_date = ?
                 AND f.scheduled_off_utc < ?
               ORDER BY f.scheduled_off_utc DESC
               LIMIT 1""",
            conn,
            params=(tail, fl_date, target["scheduled_off_utc"]),
        )

        if prior.empty:
            n_no_prior += 1
            case = "NO_PRIOR_IN_DB"
        else:
            prior_row = prior.iloc[0]
            n_have_prior_in_actuals += 1
            if pd.isna(prior_row["actual_in_utc"]):
                n_unobservable += 1  # prior exists but no actuals → can't compute delay
                case = "PRIOR_NO_ACTUALS"
            else:
                actual_arr = pd.to_datetime(prior_row["actual_in_utc"], utc=True, format="ISO8601")
                if actual_arr <= sched_off:
                    n_observable += 1
                    case = "OBSERVABLE_OK"
                else:
                    n_unobservable += 1
                    case = "PRIOR_LANDED_LATE"
            case_breakdown.append({
                "tail": tail, "carrier": target["op_carrier"],
                "current": f"{target['origin']}→{target['dest']} {sched_off.strftime('%H:%M')}",
                "prior_arr": prior_row["actual_in_utc"][:16] if prior_row["actual_in_utc"] else "(no actual)",
                "case": case,
            })

    n_total = len(targets)
    print("=" * 70)
    print(f"RESULTS for {n_total} sampled predictions")
    print("=" * 70)
    print(f"  No prior leg in DB:       {n_no_prior:>4} ({n_no_prior/n_total*100:>5.1f}%)  ← Coverage gap")
    print(f"  Prior in DB but unsettled:{n_unobservable - sum(1 for c in case_breakdown if c['case']=='PRIOR_LANDED_LATE'):>4}")
    print(f"  Prior landed AFTER cur dep:{sum(1 for c in case_breakdown if c['case']=='PRIOR_LANDED_LATE'):>3} (Observability rule kills it)")
    print(f"  Observable OK (lineage works): {n_observable:>4} ({n_observable/n_total*100:>5.1f}%)")
    print()

    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    if n_no_prior / n_total > 0.6:
        print("→ DOMINANT CAUSE: data coverage gap. >60% of predictions don't have")
        print("  ANY prior same-tail same-day leg in the DB.")
        print("  FIX: multi-hub backfill (pull from carriers' real hubs, not just KATL).")
    elif (n_unobservable + n_no_prior) / n_total > 0.6 and n_no_prior / n_total < 0.3:
        print("→ DOMINANT CAUSE: observability rule + tight rotations.")
        print("  Most prior legs exist but land too late to be 'observable'.")
        print("  FIX: relax the observability rule, or use scheduled vs actual arrival.")
    elif n_observable / n_total > 0.4:
        print("→ Some lineage works ({:.0f}%). The remaining is mixed coverage + observability.".format(
            n_observable / n_total * 100,
        ))
    else:
        print("→ Mixed causes — both coverage and observability matter.")

    print("\nSample of cases (first 15):")
    for c in case_breakdown[:15]:
        print(f"  {c['carrier']} {c['current']:<22} prior_arr={c['prior_arr']}  case={c['case']}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
