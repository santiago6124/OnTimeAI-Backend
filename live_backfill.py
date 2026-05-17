"""One-time seed of the live SQLite DB from the existing 2025 master.

Picks the last N days of the 2025 master and writes them as `flights` + `actuals`
so the first live runs have lineage/rolling history available.

Usage:
    python3 live_backfill.py                  # default: last 14 days of 2025
    python3 live_backfill.py --days 7
    python3 live_backfill.py --master path/to/master.csv --days 30
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ontimeai.live import open_db, PROJECT_ROOT


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path,
                   default=PROJECT_ROOT / "dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv")
    p.add_argument("--days", type=int, default=14,
                   help="How many trailing days of the master to seed")
    args = p.parse_args()

    print(f"Reading {args.master}")
    df = pd.read_csv(args.master, parse_dates=["FL_DATE"], low_memory=False)
    df = df.sort_values("FL_DATE").reset_index(drop=True)

    cutoff = df["FL_DATE"].max() - pd.Timedelta(days=args.days)
    seed = df[df["FL_DATE"] >= cutoff].copy()
    seed = seed[(seed["CANCELLED"].fillna(0) == 0) &
                (seed["DIVERTED"].fillna(0) == 0) &
                seed["ARR_DELAY"].notna()].copy()
    print(f"Seeding {len(seed):,} flights from {cutoff.date()} → {df['FL_DATE'].max().date()}")

    conn = open_db()
    now = datetime.now(timezone.utc).isoformat()

    # Build a synthetic fa_flight_id from BTS fields (deterministic & unique enough)
    seed["fa_flight_id"] = (
        "bts_" + seed["FL_DATE"].dt.strftime("%Y%m%d") + "_" +
        seed["OP_CARRIER"].astype(str) + seed["OP_CARRIER_FL_NUM"].astype("Int64").astype(str) +
        "_" + seed["ORIGIN"].astype(str) + "_" + seed["DEST"].astype(str)
    )

    # Compute scheduled_off_utc from EVENT_ORIGIN_UTC (which is built from FL_DATE + CRS_DEP_MIN)
    crs_min = pd.to_numeric(seed["CRS_DEP_MIN"], errors="coerce")
    if "EVENT_ORIGIN_UTC" in seed.columns:
        sched_off = pd.to_datetime(seed["EVENT_ORIGIN_UTC"], errors="coerce")
    else:
        sched_off = pd.NaT
    if "EVENT_DEST_UTC" in seed.columns:
        sched_on = pd.to_datetime(seed["EVENT_DEST_UTC"], errors="coerce")
    else:
        sched_on = pd.NaT

    flights_rows = []
    actuals_rows = []
    for _, r in seed.iterrows():
        fid = r["fa_flight_id"]
        sof = sched_off.loc[r.name].isoformat() if pd.notna(sched_off.loc[r.name]) else None
        son = sched_on.loc[r.name].isoformat() if pd.notna(sched_on.loc[r.name]) else None
        flights_rows.append((
            fid, None, str(r["OP_CARRIER"]) if pd.notna(r["OP_CARRIER"]) else None,
            str(int(r["OP_CARRIER_FL_NUM"])) if pd.notna(r["OP_CARRIER_FL_NUM"]) else None,
            r["TAIL_NUM"] if pd.notna(r["TAIL_NUM"]) else None,
            str(r["ORIGIN"]), str(r["DEST"]), None,
            r["FL_DATE"].strftime("%Y-%m-%d"),
            int(crs_min.loc[r.name]) if pd.notna(crs_min.loc[r.name]) else None,
            sof, sof, son, son,
            float(r["CRS_ELAPSED_TIME"]) if pd.notna(r["CRS_ELAPSED_TIME"]) else None,
            float(r["DISTANCE"]) if pd.notna(r["DISTANCE"]) else None,
            None, 0, 0, now, now,
        ))
        actuals_rows.append((
            fid, None, None, None, None,
            float(r["ARR_DELAY"]),
            float(r["DEP_DELAY"]) if "DEP_DELAY" in r and pd.notna(r["DEP_DELAY"]) else None,
            0, 0, now,
        ))

    cur = conn.executemany(
        """INSERT OR REPLACE INTO flights
           (fa_flight_id, ident_iata, op_carrier, flight_number, tail_num,
            origin, dest, inbound_fa_flight_id, fl_date, crs_dep_min,
            scheduled_out_utc, scheduled_off_utc, scheduled_on_utc, scheduled_in_utc,
            crs_elapsed_min, distance, aircraft_type, cancelled, diverted,
            first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        flights_rows,
    )
    print(f"  flights upserted: {cur.rowcount:,}")
    cur = conn.executemany(
        """INSERT OR REPLACE INTO actuals
           (fa_flight_id, actual_out_utc, actual_off_utc, actual_on_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        actuals_rows,
    )
    print(f"  actuals upserted: {cur.rowcount:,}")
    conn.commit()

    n_flights = conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    n_actuals = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
    print(f"\nDB now has {n_flights:,} flights, {n_actuals:,} actuals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
