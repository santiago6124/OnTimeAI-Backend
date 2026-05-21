"""Backfill weather data for known gap periods in live_data.db.

Fills weather observations for days where the live pipeline was down or hadn't
started yet. Uses IEM METAR (free, no auth required).

Known gaps:
  - May 1-3 2026: pipeline not yet active, no weather
  - May 9-11 2026: pipeline outage, no weather
  - Any day with flights but zero weather observations

Usage:
    python3 scripts/backfill_gaps.py                     # auto-detect gaps
    python3 scripts/backfill_gaps.py --dates 2026-05-01 2026-05-02 2026-05-03
    python3 scripts/backfill_gaps.py --dry-run            # show what would be done
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.live import open_db, fetch_iem_obs, upsert_weather, AIRPORTS


def detect_weather_gaps(conn) -> list[str]:
    """Find dates with flights but zero or very few weather observations."""
    rows = conn.execute(
        """SELECT f.fl_date, count(f.fa_flight_id) as n_flights,
                  (SELECT count(*) FROM weather_obs
                   WHERE date(valid_utc) = f.fl_date) as n_wx
           FROM flights f
           GROUP BY f.fl_date
           HAVING n_flights > 10
           ORDER BY f.fl_date"""
    ).fetchall()

    gaps = []
    for fl_date, n_flights, n_wx in rows:
        if n_wx < 50:  # threshold: fewer than 50 wx obs is basically empty
            gaps.append(fl_date)
            print(f"  GAP: {fl_date} — {n_flights} flights, {n_wx} wx obs")

    return gaps


def get_active_airports(conn, date: str) -> set[str]:
    """Get airports that appear in flights for a given date."""
    rows = conn.execute(
        """SELECT DISTINCT origin FROM flights WHERE fl_date = ?
           UNION
           SELECT DISTINCT dest FROM flights WHERE fl_date = ?""",
        (date, date),
    ).fetchall()
    airports = {r[0] for r in rows if r[0]}
    return airports & AIRPORTS


def backfill_weather_for_date(conn, date_str: str, dry_run: bool = False) -> int:
    """Fetch and upsert IEM weather for a single date."""
    dt = pd.Timestamp(date_str)
    start = pd.Timestamp(f"{date_str} 00:00:00")
    end = pd.Timestamp(f"{date_str} 23:59:59")

    # Get airports active on this date
    airports = get_active_airports(conn, date_str)
    if not airports:
        print(f"  {date_str}: no flights, skipping")
        return 0

    print(f"  {date_str}: fetching weather for {len(airports)} airports...")
    if dry_run:
        print(f"  [dry-run] would fetch IEM for {sorted(airports)}")
        return 0

    wx = fetch_iem_obs(airports, start, end)
    if wx.empty:
        print(f"  {date_str}: no weather data returned from IEM")
        return 0

    n = upsert_weather(conn, wx)
    print(f"  {date_str}: upserted {n} weather observations ({len(airports)} stations)")
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill weather gaps in live_data.db")
    p.add_argument("--dates", nargs="*", default=None,
                   help="Specific dates to backfill (YYYY-MM-DD). Auto-detects if omitted.")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = p.parse_args()

    conn = open_db()

    if args.dates:
        gap_dates = args.dates
        print(f"Manually specified dates: {gap_dates}")
    else:
        print("Auto-detecting weather gaps...\n")
        gap_dates = detect_weather_gaps(conn)

    if not gap_dates:
        print("\nNo weather gaps detected. ✅")
        return 0

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Backfilling {len(gap_dates)} days...\n")
    total_wx = 0
    for date_str in gap_dates:
        n = backfill_weather_for_date(conn, date_str, args.dry_run)
        total_wx += n

    print(f"\nDone. Total weather observations added: {total_wx:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
