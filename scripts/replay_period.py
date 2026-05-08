"""Replay live pipeline over a historical date range using AeroAPI backfill.

Phase 1: Pull KATL departures + arrivals for [start - history_days, end + 1]
         to populate flights + actuals tables with rich history.
Phase 2: For each scheduled flight in [start, end], simulate prediction at
         scheduled_off_utc - 2h, using the live pipeline (build_inference_frame,
         cold-deck fallback, model.predict, isotonic calibrator). Persist with
         threshold_strategy='replay_quantile@X' to distinguish from real cron.
Phase 3: Settle uses the actuals already pulled in Phase 1.
Phase 4: Tell user to run analyze_live_period.py for the metrics.

Usage:
    python3 scripts/replay_period.py --start 2026-05-01 --end 2026-05-06 \
        --history-days 7 --max-pages 25

Cost (approximate, $0.005 per AeroAPI call):
    Phase 1: 60 calls/day × (history_days + replay_days) ≈ $0.30/day pulled
    Phase 2: free (no AeroAPI calls — uses model on data from DB)
    For history=7 + replay=3 days: ~$3.00 USD
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ontimeai.config import ARTIFACTS_DIR
from ontimeai.lineage_fallback import load_lookups
from ontimeai.live import (
    aeroapi_to_flight_row,
    build_inference_frame,
    fetch_airport_flights,
    open_db,
    stable_id,
    upsert_actuals_from_aeroapi,
    upsert_flights,
)
from ontimeai.model import (
    load_artifact,
    predict_label,
    predict_proba,
    quantile_threshold,
)
from predict import prepare_inference_frame


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _backfill_day(
    conn, airports: list[str], day_start: datetime, max_pages: int,
) -> tuple[int, int, int]:
    """Pull departures + arrivals for one day from each airport in `airports`.

    Returns (n_flights, n_actuals, n_calls_est).
    """
    day_end = day_start + timedelta(days=1)
    n_flights = 0
    n_actuals = 0
    n_calls = 0
    for airport in airports:
        for kind in ("departures", "arrivals"):
            try:
                flights = fetch_airport_flights(
                    airport, kind, _iso(day_start), _iso(day_end), max_pages,
                )
            except Exception as e:
                print(f"      {airport} {kind}: FAIL ({e})")
                continue
            n_calls += min(max_pages, (len(flights) + 14) // 15) or 1

            rows = [
                r for r in (aeroapi_to_flight_row(f, strict_airport_filter=False)
                            for f in flights) if r
            ]
            upsert_flights(conn, rows)
            n_flights += len(rows)

            landed = [f for f in flights if f.get("actual_in")]
            upsert_actuals_from_aeroapi(conn, landed)
            n_actuals += len(landed)

            time.sleep(1.5)  # be kind to AeroAPI
    return n_flights, n_actuals, n_calls


def _replay_day(
    conn, day_start: datetime, meta: dict, fallback,
    target_pos_rate: float, history_days: int,
) -> tuple[int, dict]:
    """Simulate predictions for all KATL-touching flights scheduled in [day, day+1)."""
    day_end = day_start + timedelta(days=1)

    # Targets = flights from the universe scheduled in this day
    rows = conn.execute(
        "SELECT fa_flight_id FROM flights "
        "WHERE scheduled_off_utc >= ? AND scheduled_off_utc < ? "
        "  AND (origin = 'ATL' OR dest = 'ATL') "
        "  AND COALESCE(cancelled, 0) = 0 AND COALESCE(diverted, 0) = 0",
        (day_start.isoformat(), day_end.isoformat()),
    ).fetchall()
    target_ids = [r[0] for r in rows]
    if not target_ids:
        return 0, {}

    df = build_inference_frame(conn, target_ids, history_days=history_days)
    if df.empty:
        return 0, {}

    target_mask = df["fa_flight_id"].isin(target_ids) & df["ARR_DELAY"].isna()
    n_targets = int(target_mask.sum())
    if n_targets == 0:
        return 0, {}

    X = prepare_inference_frame(df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback)
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in X.columns:
        if c in cat_cols_set:
            continue
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    proba = predict_proba(meta["booster"], X)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)

    target_proba = proba[target_mask.to_numpy()]
    if target_pos_rate > 0 and target_proba.size >= 5:
        threshold_used = quantile_threshold(target_proba, target_pos_rate)
        threshold_strategy = f"replay_quantile@{target_pos_rate:.2f}"
    else:
        threshold_used = float(meta["threshold"])
        threshold_strategy = "replay_artifact"
    labels = predict_label(proba, threshold_used, "binary")

    stats = {
        "n": n_targets,
        "mean_proba": float(target_proba.mean()),
        "std_proba": float(target_proba.std()),
        "threshold": float(threshold_used),
        "pos_pred_rate": float((target_proba >= threshold_used).mean()),
    }

    # Persist with predicted_at_utc = scheduled_off - 2h
    pred_rows = []
    for i in df.index[target_mask]:
        sched_off = df.loc[i, "EVENT_ORIGIN_UTC"]
        if pd.isna(sched_off):
            continue
        pred_at = (pd.Timestamp(sched_off) - pd.Timedelta(hours=2)).isoformat()
        pred_rows.append((
            df.loc[i, "fa_flight_id"], stable_id(df.loc[i, "fa_flight_id"]),
            pred_at, float(proba[i]), int(labels[i]),
            float(threshold_used), threshold_strategy,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
            threshold_used, threshold_strategy) VALUES (?,?,?,?,?,?,?)""",
        pred_rows,
    )
    conn.commit()
    return len(pred_rows), stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First day to predict (YYYY-MM-DD)")
    p.add_argument("--end", help="Last day (default: same as --start)")
    p.add_argument("--history-days", type=int, default=7,
                   help="Days of history to backfill before --start (default 7)")
    p.add_argument("--max-pages", type=int, default=25,
                   help="AeroAPI cursor pagination cap per endpoint (default 25)")
    p.add_argument("--airport", default="KATL",
                   help="Target airport for predictions (default KATL)")
    p.add_argument("--backfill-airports",
                   default="KATL,KDFW,KCLT,KORD,KDEN,KMIA,KJFK,KLGA,KBOS,KFLL,KMCO,KLAX,KIAH,KEWR,KPHX",
                   help="Comma-separated ICAO codes to backfill (default top-15 US hubs). "
                        "Wider coverage = better lineage features for multi-hub carriers.")
    p.add_argument("--artifact", default=str(ARTIFACTS_DIR / "4year_v4_full"))
    p.add_argument("--target-pos-rate", type=float, default=0.22)
    p.add_argument("--skip-backfill", action="store_true",
                   help="Skip Phase 1 (use existing DB history)")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end or args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    history_start = start - timedelta(days=args.history_days)

    # AeroAPI free tier limits history to 10 days back from "now". Cap automatically
    # with 1 day of margin (so use 9 days max effective).
    now = datetime.now(timezone.utc)
    earliest_allowed = (now - timedelta(days=9)).replace(hour=0, minute=0, second=0, microsecond=0)
    if history_start < earliest_allowed:
        new_history_days = max(0, (start - earliest_allowed).days)
        print(f"WARNING: history_start {history_start.date()} is older than AeroAPI's 10-day limit.")
        print(f"  Capping history-days from {args.history_days} to {new_history_days}")
        history_start = earliest_allowed
    if start < earliest_allowed:
        new_start = earliest_allowed + timedelta(days=1)
        print(f"WARNING: --start {start.date()} is too old. Using {new_start.date()}.")
        start = new_start

    print(f"Replay window:    {start.date()} → {end.date()} ({(end - start).days + 1} days)")
    print(f"History backfill: {history_start.date()} → {start.date()} ({(start - history_start).days} days)")

    conn = open_db()

    total_calls = 0
    total_flights = 0
    total_actuals = 0

    backfill_airports = [a.strip().upper() for a in args.backfill_airports.split(",") if a.strip()]
    if not args.skip_backfill:
        print(f"\n=== Phase 1: backfilling history from {len(backfill_airports)} airports ===")
        print(f"  Airports: {', '.join(backfill_airports)}")
        cur = history_start
        while cur < end + timedelta(days=1):
            print(f"\n  {cur.date()}:")
            t0 = time.time()
            n_f, n_a, n_c = _backfill_day(conn, backfill_airports, cur, args.max_pages)
            total_flights += n_f
            total_actuals += n_a
            total_calls += n_c
            print(f"    total: {n_f} flights, {n_a} actuals  "
                  f"({n_c} API calls, {time.time() - t0:.0f}s, "
                  f"running cost ~${total_calls * 0.005:.2f})")
            cur += timedelta(days=1)
        print(f"\nTotal Phase 1: {total_flights:,} flights, {total_actuals:,} actuals")
        print(f"Estimated cost: ~${total_calls * 0.005:.2f} USD")
    else:
        print("\n=== Phase 1: SKIPPED (--skip-backfill) ===")

    print("\n=== Phase 2: replaying predictions (offline, no API cost) ===")
    meta = load_artifact(Path(args.artifact))
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = load_lookups(fallback_path) if fallback_path.exists() else None
    if fallback is None:
        print("  WARNING: no cold-deck lookup at artifacts/lineage_fallback.joblib")

    cur = start
    grand_n = 0
    grand_stats = []
    while cur < end + timedelta(days=1):
        n, stats = _replay_day(
            conn, cur, meta, fallback, args.target_pos_rate, args.history_days,
        )
        if n > 0:
            print(f"  {cur.date()}: {n:>4} predictions  "
                  f"mean_proba={stats['mean_proba']:.3f}  std={stats['std_proba']:.3f}  "
                  f"threshold={stats['threshold']:.3f}  "
                  f"pos_pred_rate={stats['pos_pred_rate']:.3f}")
            grand_stats.append((cur.date(), stats))
        else:
            print(f"  {cur.date()}: no targets (no scheduled flights or empty inference frame)")
        grand_n += n
        cur += timedelta(days=1)

    print(f"\n=== Summary ===")
    print(f"Total replayed predictions: {grand_n}")
    if grand_stats:
        all_means = np.array([s[1]['mean_proba'] for s in grand_stats])
        all_stds = np.array([s[1]['std_proba'] for s in grand_stats])
        print(f"Across days — mean(proba)= {all_means.mean():.3f}±{all_means.std():.3f}, "
              f"mean(std_proba)= {all_stds.mean():.3f}")
        if all_stds.mean() > 0.15:
            print("→ Std looks healthy (>0.15) — model is discriminating!")
        elif all_stds.mean() > 0.10:
            print("→ Std moderate (0.10-0.15) — partial improvement vs prior live runs")
        else:
            print("→ Std still low (<0.10) — feature degradation persists, deeper investigation needed")
    print(f"\nNext: run analyze")
    print(f"  python scripts/analyze_live_period.py --since {start.date()} --until {end.date()}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
