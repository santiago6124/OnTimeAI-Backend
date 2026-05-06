"""One-off utility to settle pending predictions.

The standard `live_pull.py` only pulls KATL arrivals (which only settles
ARR_TO_ATL predictions). DEP_FROM_ATL predictions need the `departures`
endpoint instead — where AeroAPI also returns `actual_in` + `arrival_delay`
for flights that have landed at their destination.

Usage:
    python3 scripts/settle_pending_predictions.py
    python3 scripts/settle_pending_predictions.py --hours 24
    python3 scripts/settle_pending_predictions.py --max-pages 5
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure repo root on path so `ontimeai` package imports work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ontimeai.live import (
    open_db,
    fetch_airport_flights,
    upsert_actuals_from_aeroapi,
)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--airport", default="KATL")
    p.add_argument("--hours", type=int, default=24,
                   help="Window backwards from now (default 24h)")
    p.add_argument("--max-pages", type=int, default=5)
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=args.hours)

    print(f"Settling predictions from {_iso(start)} → {_iso(now)}")
    conn = open_db()

    # Show the gap before — match by stable_id (fa_flight_id has unstable suffix
    # between /scheduled_* and /departures-/arrivals endpoints).
    pre = conn.execute(
        "SELECT COUNT(*) FROM predictions p "
        "LEFT JOIN actuals a ON a.stable_id = p.stable_id "
        "WHERE a.actual_in_utc IS NULL OR a.arr_delay_min IS NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"Before: {pre} of {total} predictions are missing actuals")

    n_calls = 0
    n_actuals = 0

    # ---- 1. arrivals endpoint (settles ARR_TO_ATL preds) ----
    print(f"\n[1] AeroAPI arrivals at {args.airport}...")
    arrived = fetch_airport_flights(args.airport, "arrivals",
                                    _iso(start), _iso(now), args.max_pages)
    n_calls += min(len(arrived) // 15 + 1, args.max_pages)
    arr_filt = [r for r in arrived if r.get("actual_in")]
    n = upsert_actuals_from_aeroapi(conn, arr_filt)
    print(f"   pulled {len(arrived)} arrivals → wrote {n} actuals")
    n_actuals += n

    # ---- 2. departures endpoint (settles DEP_FROM_ATL preds) ----
    # AeroAPI Flight object from /departures includes actual_off + actual_in
    # + arrival_delay if the flight has already landed at destination.
    print(f"\n[2] AeroAPI departures from {args.airport}...")
    departed = fetch_airport_flights(args.airport, "departures",
                                     _iso(start), _iso(now), args.max_pages)
    n_calls += min(len(departed) // 15 + 1, args.max_pages)
    dep_filt = [r for r in departed if r.get("actual_in")]
    print(f"   pulled {len(departed)} departures, "
          f"{len(dep_filt)} have landed at destination")
    n = upsert_actuals_from_aeroapi(conn, dep_filt)
    print(f"   wrote {n} actuals (DEP_FROM_ATL settled)")
    n_actuals += n

    # Show the gap after — same stable_id-based match.
    post = conn.execute(
        "SELECT COUNT(*) FROM predictions p "
        "LEFT JOIN actuals a ON a.stable_id = p.stable_id "
        "WHERE a.actual_in_utc IS NULL OR a.arr_delay_min IS NULL"
    ).fetchone()[0]
    settled = conn.execute(
        "SELECT COUNT(*) FROM predictions p "
        "INNER JOIN actuals a ON a.stable_id = p.stable_id "
        "WHERE p.stable_id IS NOT NULL "
        "  AND a.actual_in_utc IS NOT NULL "
        "  AND a.arr_delay_min IS NOT NULL "
        "  AND COALESCE(a.cancelled,0)=0 AND COALESCE(a.diverted,0)=0"
    ).fetchone()[0]
    print(f"\nAfter: {post} predictions still missing actuals "
          f"(was {pre}, settled {pre - post})")
    print(f"Total settled & evaluable: {settled} of {total}")
    print(f"AeroAPI cost estimate: {n_calls} calls × $0.005 = ${n_calls * 0.005:.2f} USD")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
