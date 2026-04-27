"""Per-tick pipeline: pull schedules + actuals + weather, predict, store.

Designed to be cron'd every 30 min. Idempotent: safe to re-run.

Usage:
    python3 live_pull.py                     # default: scheduled +0..+4h, actuals -6..0h
    python3 live_pull.py --schedule-hours 6  # look 6h ahead instead
    python3 live_pull.py --no-weather        # skip IEM pull
    python3 live_pull.py --dry-run           # print plan, skip writes
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from ontimeai.live import (
    open_db, fetch_airport_flights, fetch_iem_obs,
    aeroapi_to_flight_row, upsert_flights, upsert_actuals_from_aeroapi, upsert_weather,
    build_inference_frame, AIRPORTS,
)
from ontimeai.model import load_artifact, predict_label, predict_proba
from predict import prepare_inference_frame
from ontimeai.config import ARTIFACTS_DIR


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=ARTIFACTS_DIR / "4year_v3_full")
    p.add_argument("--airport", default="KATL")
    p.add_argument("--schedule-hours", type=int, default=2,
                   help="Pull scheduled departures for the next N hours")
    p.add_argument("--actuals-hours", type=int, default=4,
                   help="Pull arrivals from the past N hours to settle actuals")
    p.add_argument("--max-pages", type=int, default=2,
                   help="Max client-side cursor pages per endpoint (1 page ≈ 15 flights)")
    p.add_argument("--skip-arrivals-sched", action="store_true",
                   help="Skip the scheduled_arrivals endpoint (saves API calls)")
    p.add_argument("--skip-actuals", action="store_true",
                   help="Skip the arrivals endpoint (saves API calls)")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    now = datetime.now(timezone.utc)
    sched_start = now
    sched_end = now + timedelta(hours=args.schedule_hours)
    arr_start = now - timedelta(hours=args.actuals_hours)
    arr_end = now

    print(f"Tick {now.isoformat()}")
    print(f"  schedule window: {_iso(sched_start)} → {_iso(sched_end)}")
    print(f"  actuals window:  {_iso(arr_start)} → {_iso(arr_end)}")

    if args.dry_run:
        print("\n[dry-run] would call AeroAPI scheduled_departures + arrivals + IEM")
        return 0

    conn = open_db()
    cur = conn.execute(
        "INSERT INTO runs (started_utc) VALUES (?)", (now.isoformat(),),
    )
    run_id = cur.lastrowid
    conn.commit()

    # ---- 1. scheduled departures (KATL) ----
    print("\n[1] AeroAPI scheduled_departures...")
    sched = fetch_airport_flights(args.airport, "scheduled_departures",
                                  _iso(sched_start), _iso(sched_end), args.max_pages)
    print(f"   pulled {len(sched)} scheduled departures")
    sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in sched) if r]
    n_sched = upsert_flights(conn, sched_rows)
    print(f"   upserted {n_sched} flights to DB (after ATL+known-airports filter)")

    arr_sched_rows: list[dict] = []
    arr_sched: list[dict] = []
    if not args.skip_arrivals_sched:
        # ---- 2. scheduled arrivals (KATL) — captures FLOW=ARR_TO_ATL ----
        print("\n[2] AeroAPI scheduled_arrivals...")
        arr_sched = fetch_airport_flights(args.airport, "scheduled_arrivals",
                                          _iso(sched_start), _iso(sched_end), args.max_pages)
        print(f"   pulled {len(arr_sched)} scheduled arrivals")
        arr_sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in arr_sched) if r]
        n_arr_sched = upsert_flights(conn, arr_sched_rows)
        print(f"   upserted {n_arr_sched} flights to DB")
    else:
        print("\n[2] (skipped scheduled_arrivals)")

    n_act = 0
    if not args.skip_actuals:
        # ---- 3. completed arrivals → actuals ----
        print("\n[3] AeroAPI arrivals (completed)...")
        arrived = fetch_airport_flights(args.airport, "arrivals",
                                        _iso(arr_start), _iso(arr_end), args.max_pages)
        print(f"   pulled {len(arrived)} arrivals")
        arrived_filt = [r for r in arrived if r.get("actual_in")]
        n_act = upsert_actuals_from_aeroapi(conn, arrived_filt)
        print(f"   wrote {n_act} actuals")
    else:
        print("\n[3] (skipped actuals)")
        arrived = []

    # ---- 4. weather ----
    n_wx = 0
    if not args.no_weather:
        print("\n[4] IEM METAR refresh...")
        wx = fetch_iem_obs(AIRPORTS, sched_start - timedelta(hours=2),
                           sched_end + timedelta(hours=2))
        if not wx.empty:
            n_wx = upsert_weather(conn, wx)
            print(f"   upserted {n_wx} weather observations")

    # ---- 5. predict scheduled flights ----
    print("\n[5] Building features and predicting...")
    target_ids = [r["fa_flight_id"] for r in sched_rows + arr_sched_rows]
    if not target_ids:
        print("   no flights to predict")
        conn.execute(
            "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?, actuals_updated=?, weather_obs_added=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), len(sched) + len(arr_sched), 0, n_act, n_wx, run_id),
        )
        conn.commit()
        return 0

    df = build_inference_frame(conn, target_ids, history_days=7)
    if df.empty:
        print("   inference frame empty")
        return 0

    target_mask = df["fa_flight_id"].isin(target_ids) & df["ARR_DELAY"].isna()
    print(f"   {target_mask.sum()} target rows | {(~target_mask).sum()} history rows for lineage")

    meta = load_artifact(args.artifact)
    X = prepare_inference_frame(df, meta["feature_cols"], meta["cat_mapping"])
    # Coerce non-categorical object columns to numeric (live path may have all-NaN
    # weather columns that pd.NA leaves as object, which LightGBM rejects).
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in X.columns:
        if c in cat_cols_set:
            continue
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    proba = predict_proba(meta["booster"], X)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)
    labels = predict_label(proba, meta["threshold"], "binary")

    # Persist only target predictions
    pred_now = datetime.now(timezone.utc).isoformat()
    pred_rows = [
        (df.loc[i, "fa_flight_id"], pred_now, float(proba[i]), int(labels[i]))
        for i in df.index[target_mask]
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, predicted_at_utc, proba_delay, predicted_delay)
           VALUES (?,?,?,?)""",
        pred_rows,
    )
    conn.commit()
    print(f"   wrote {len(pred_rows)} predictions")

    conn.execute(
        "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?, actuals_updated=?, weather_obs_added=? WHERE run_id=?",
        (datetime.now(timezone.utc).isoformat(),
         len(sched) + len(arr_sched) + len(arrived),
         len(pred_rows), n_act, n_wx, run_id),
    )
    conn.commit()

    print(f"\nDone. run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
