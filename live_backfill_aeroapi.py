"""Backfill the live SQLite history buffer with completed flights from AeroAPI.

Pulls /airports/KATL/flights/departures and /airports/KATL/flights/arrivals
for the last N days (default 2), maps them to our schema, and inserts into
both `flights` and `actuals` tables. This populates the recent-history window
that lineage/rolling features need to work.

Usage:
    python3 live_backfill_aeroapi.py                  # last 2 days
    python3 live_backfill_aeroapi.py --days 3
    python3 live_backfill_aeroapi.py --days 2 --skip-arrivals  # only departures
    python3 live_backfill_aeroapi.py --dry-run        # estimate cost, no call
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from ontimeai.live import (
    open_db, fetch_airport_flights,
    aeroapi_to_flight_row, upsert_flights, upsert_actuals_from_aeroapi,
)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--airport", default="KATL")
    p.add_argument("--days", type=int, default=2,
                   help="How many days back to backfill (default 2; ~$1 per day)")
    p.add_argument("--max-pages", type=int, default=80,
                   help="Max client-side pages per day-endpoint (each ~15 flights)")
    p.add_argument("--skip-arrivals", action="store_true",
                   help="Only pull departures (saves ~50%% of calls)")
    p.add_argument("--skip-departures", action="store_true",
                   help="Only pull arrivals")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    end = now

    print(f"Backfill window (UTC): {_iso(start)} → {_iso(end)} ({args.days} days)")
    if args.dry_run:
        per_day = 50  # rough estimate: ~750 flights/day / 15 per page = 50 pages
        endpoints = (0 if args.skip_departures else 1) + (0 if args.skip_arrivals else 1)
        est = args.days * per_day * endpoints
        print(f"Estimated AeroAPI calls: ~{est} (cost ~${est * 0.005:.2f} USD)")
        return 0

    conn = open_db()

    total_flights_upserted = 0
    total_actuals_upserted = 0

    for day_offset in range(args.days):
        day_start = start + timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)
        if day_end > end:
            day_end = end
        print(f"\n=== Day {day_offset + 1}/{args.days}: {_iso(day_start)} → {_iso(day_end)} ===")

        if not args.skip_departures:
            print(f"  [departures] pulling completed departures from {args.airport}")
            try:
                deps = fetch_airport_flights(args.airport, "departures",
                                             _iso(day_start), _iso(day_end), args.max_pages)
                print(f"    got {len(deps)} flights")
                rows = [r for r in (aeroapi_to_flight_row(rec) for rec in deps) if r]
                n_f = upsert_flights(conn, rows)
                actuals = [d for d in deps if d.get("actual_in") or d.get("actual_off")]
                n_a = upsert_actuals_from_aeroapi(conn, actuals)
                total_flights_upserted += n_f
                total_actuals_upserted += n_a
                print(f"    upserted {n_f} flights, {n_a} actuals")
            except Exception as e:
                print(f"    [FAIL] {e}")
            time.sleep(1)

        if not args.skip_arrivals:
            print(f"  [arrivals] pulling completed arrivals at {args.airport}")
            try:
                arrs = fetch_airport_flights(args.airport, "arrivals",
                                             _iso(day_start), _iso(day_end), args.max_pages)
                print(f"    got {len(arrs)} flights")
                rows = [r for r in (aeroapi_to_flight_row(rec) for rec in arrs) if r]
                n_f = upsert_flights(conn, rows)
                actuals = [a for a in arrs if a.get("actual_in") or a.get("actual_off")]
                n_a = upsert_actuals_from_aeroapi(conn, actuals)
                total_flights_upserted += n_f
                total_actuals_upserted += n_a
                print(f"    upserted {n_f} flights, {n_a} actuals")
            except Exception as e:
                print(f"    [FAIL] {e}")
            time.sleep(1)

    n_flights = conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    n_actuals = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
    n_recent = conn.execute(
        "SELECT COUNT(*) FROM flights WHERE scheduled_off_utc >= ?",
        ((datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),),
    ).fetchone()[0]
    print(f"\nDone. Backfill upserted: {total_flights_upserted} flights, {total_actuals_upserted} actuals")
    print(f"DB totals: {n_flights:,} flights | {n_actuals:,} actuals | {n_recent:,} flights in last 7 days (lineage window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
