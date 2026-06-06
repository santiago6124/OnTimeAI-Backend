"""Per-tick pipeline: pull schedules + actuals + weather, predict, store.

Designed to be cron'd every 30 min. Idempotent: safe to re-run.

Data source modes (env var `LIVE_DATA_SOURCE`):
    - aeroapi (default): pulls all data from AeroAPI per tick (~$0.10/tick)
    - harvester: skips AeroAPI; reads flights/actuals already populated by
                 OnTimeAI-Scrapper into the shared bucket DB (~$0.001/tick).

Usage:
    python3 live_pull.py                            # default: AeroAPI mode
    LIVE_DATA_SOURCE=harvester python3 live_pull.py # uses harvester buffer
    python3 live_pull.py --schedule-hours 6         # look 6h ahead instead
    python3 live_pull.py --no-weather               # skip IEM pull
    python3 live_pull.py --dry-run                  # print plan, skip writes
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ontimeai.live import (
    open_db, fetch_airport_flights, fetch_iem_obs,
    aeroapi_to_flight_row, upsert_flights, upsert_actuals_from_aeroapi, upsert_weather,
    build_inference_frame, chain_walk_inbound, AIRPORTS, stable_id,
    snapshot_nas_status, latest_nas_status, gdp_post_prediction_adjust,
    compute_atl_arrival_congestion, carrier_delay_rate_bayesian,
    intermediate_dep_delay_adjust, compute_adsb_eta_delay, adsb_eta_adjust,
    compute_adsb_holding_min, adsb_holding_adjust,
)
from ontimeai.lineage_fallback import load_lookups, build_live_turnaround_lookups
from ontimeai.model import load_artifact, predict_label, predict_proba, quantile_threshold
from predict import prepare_inference_frame
from ontimeai.config import ARTIFACTS_DIR


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=ARTIFACTS_DIR / "4year_v9")
    p.add_argument("--airport", default="KATL")
    p.add_argument("--schedule-hours", type=int, default=6,
                   help="Pull scheduled departures for the next N hours")
    p.add_argument("--actuals-hours", type=int, default=4,
                   help="Pull arrivals from the past N hours to settle actuals")
    p.add_argument("--actuals-offset-hours", type=int, default=0,
                   help="Shift the actuals window back by N hours (e.g. --actuals-hours 3 "
                        "--actuals-offset-hours 6 covers the window 9h..6h ago)")
    p.add_argument("--max-pages", type=int, default=2,
                   help="Max client-side cursor pages per endpoint (1 page ≈ 15 flights)")
    p.add_argument("--skip-arrivals-sched", action="store_true",
                   help="Skip the scheduled_arrivals endpoint (saves API calls)")
    p.add_argument("--skip-actuals", action="store_true",
                   help="Skip the arrivals endpoint (saves API calls)")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--chain-walk-max",
        type=int,
        default=20,
        help=(
            "Max AeroAPI calls per tick to chain-walk `inbound_fa_flight_id` "
            "→ hydrates prev_arr_delay_tail without waiting for backfill. "
            "Set to 0 to disable. Each call costs ~$0.005."
        ),
    )
    p.add_argument(
        "--target-pos-rate",
        type=float,
        default=0.22,
        help=(
            "Target predicted-positive rate for quantile threshold (default 0.22, "
            "matches v4_full test base rate). Set to 0 to fall back to artifact threshold."
        ),
    )
    args = p.parse_args()

    # LIVE_DATA_SOURCE: env-var switch entre AeroAPI (default) y harvester (buffer GCS).
    # Fase 4 del plan: depreca AeroAPI cuando el harvester demuestre paridad de AUC.
    data_source = os.getenv("LIVE_DATA_SOURCE", "aeroapi").lower()
    if data_source not in ("aeroapi", "harvester"):
        raise SystemExit(f"LIVE_DATA_SOURCE='{data_source}' inválido; usar 'aeroapi' o 'harvester'")
    is_harvester_mode = data_source == "harvester"

    now = datetime.now(timezone.utc)
    sched_start = now
    sched_end = now + timedelta(hours=args.schedule_hours)
    arr_end   = now - timedelta(hours=args.actuals_offset_hours)
    arr_start = arr_end - timedelta(hours=args.actuals_hours)

    print(f"Tick {now.isoformat()}")
    print(f"  data source:     {data_source}")
    print(f"  schedule window: {_iso(sched_start)} → {_iso(sched_end)}")
    print(f"  actuals window:  {_iso(arr_start)} → {_iso(arr_end)}")

    if args.dry_run:
        if is_harvester_mode:
            print("\n[dry-run] would read flights/actuals from shared bucket DB + IEM weather")
        else:
            print("\n[dry-run] would call AeroAPI scheduled_departures + arrivals + IEM")
        return 0

    conn = open_db()
    cur = conn.execute(
        "INSERT INTO runs (started_utc) VALUES (?)", (now.isoformat(),),
    )
    run_id = cur.lastrowid
    conn.commit()

    sched: list[dict] = []
    arr_sched: list[dict] = []
    arrived: list[dict] = []
    sched_rows: list[dict] = []
    arr_sched_rows: list[dict] = []
    n_act = 0
    n_chain_calls = 0
    n_chain_actuals = 0

    if is_harvester_mode:
        # ---- harvester mode: leemos del buffer poblado por OnTimeAI-Scrapper ----
        print("\n[1-3b] harvester mode: leyendo flights del buffer GCS (no AeroAPI calls)")
        # Vuelos KATL en la ventana sched — el harvester ya los pobló
        # (origin='ATL' OR dest='ATL' en schema IATA del scrapper)
        # SQLite datetime() normaliza ambos formatos:
        #   '2026-05-21T20:25:00'       (AeroAPI, sin TZ)
        #   '2026-05-22T22:00:00+00:00' (FR24, con TZ)
        cur = conn.execute(
            """
            SELECT fa_flight_id, origin, dest, tail_num, op_carrier,
                   scheduled_out_utc, scheduled_in_utc
            FROM flights
            WHERE (origin = 'ATL' OR dest = 'ATL')
              AND scheduled_out_utc IS NOT NULL
              AND datetime(scheduled_out_utc) >= datetime(?)
              AND datetime(scheduled_out_utc) < datetime(?)
            ORDER BY scheduled_out_utc
            """,
            (_iso(sched_start), _iso(sched_end)),
        )
        cols = [d[0] for d in cur.description]
        sched_rows = [dict(zip(cols, r)) for r in cur.fetchall() if r[0]]
        # Heurística para mantener compat con el código downstream:
        # tratamos los vuelos cuyo destino es ATL como "arr_sched" y los demás como sched.
        arr_sched_rows = [r for r in sched_rows if r.get("dest") == "ATL"]
        sched_rows = [r for r in sched_rows if r.get("origin") == "ATL"]
        print(f"   buffer hits: {len(sched_rows)} departures + {len(arr_sched_rows)} arrivals (ventana sched)")
        # n_act y chain_walk: ambos los hace el harvester, no replicamos
    else:
        # ---- AeroAPI mode (default, comportamiento original) ----
        print("\n[1] AeroAPI scheduled_departures...")
        try:
            sched = fetch_airport_flights(args.airport, "scheduled_departures",
                                          _iso(sched_start), _iso(sched_end), args.max_pages)
            print(f"   pulled {len(sched)} scheduled departures")
            sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in sched) if r]
            n_sched = upsert_flights(conn, sched_rows)
            print(f"   upserted {n_sched} flights to DB (after ATL+known-airports filter)")
        except RuntimeError as _e:
            print(f"   [1] skipped (rate-limited): {_e}")
            sched = []

        if not args.skip_arrivals_sched:
            # ---- 2. scheduled arrivals (KATL) — captures FLOW=ARR_TO_ATL ----
            time.sleep(5)
            print("\n[2] AeroAPI scheduled_arrivals...")
            try:
                arr_sched = fetch_airport_flights(args.airport, "scheduled_arrivals",
                                                  _iso(sched_start), _iso(sched_end), args.max_pages)
                print(f"   pulled {len(arr_sched)} scheduled arrivals")
                arr_sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in arr_sched) if r]
                n_arr_sched = upsert_flights(conn, arr_sched_rows)
                print(f"   upserted {n_arr_sched} flights to DB")
            except RuntimeError as _e:
                print(f"   [2] skipped (rate-limited): {_e}")
                arr_sched = []
        else:
            print("\n[2] (skipped scheduled_arrivals)")
            arr_sched = []

        # ---- 2b. Intermediate actuals (Tier 3 #1): some scheduled_* records
        # already have actual_off because the target took off but hasn't
        # arrived yet. Persist them so the prediction phase can boost proba
        # using the empirical P(arr_delay|dep_delay) relationship.
        intermediate_recs = [
            r for r in (sched + arr_sched)
            if r.get("actual_off")
        ]
        if intermediate_recs:
            n_intermediate = upsert_actuals_from_aeroapi(conn, intermediate_recs)
            print(f"\n[2b] intermediate dep_delay captures: {n_intermediate} flights already departed")

        if not args.skip_actuals:
            # Cap actuals at 4 pages — sufficient for last 4h of ATL arrivals,
            # and avoids burning API quota needed for scheduled endpoints.
            actuals_pages = min(args.max_pages, 4)

            # ---- 3. completed arrivals to KATL → actuals (settles ARR_TO_ATL preds) ----
            time.sleep(5)
            print("\n[3] AeroAPI arrivals (completed at KATL)...")
            try:
                arrived = fetch_airport_flights(args.airport, "arrivals",
                                                _iso(arr_start), _iso(arr_end), actuals_pages)
                print(f"   pulled {len(arrived)} arrivals")
                arrived_filt = [r for r in arrived if r.get("actual_in")]
                n_act_arr = upsert_actuals_from_aeroapi(conn, arrived_filt)
                print(f"   wrote {n_act_arr} actuals")
            except RuntimeError as _e:
                print(f"   [3] skipped (rate-limited): {_e}")
                n_act_arr = 0

            # ---- 3a. completed departures from KATL → actuals (settles DEP_FROM_ATL preds)
            # Non-fatal: if rate-limited after steps 1+2+3, log and continue.
            time.sleep(5)
            print("\n[3a] AeroAPI departures (completed from KATL)...")
            try:
                departed = fetch_airport_flights(args.airport, "departures",
                                                 _iso(arr_start), _iso(arr_end), actuals_pages)
                landed = [r for r in departed if r.get("actual_in")]
                print(f"   pulled {len(departed)} departures, {len(landed)} have landed at destination")
                n_act_dep = upsert_actuals_from_aeroapi(conn, landed)
                print(f"   wrote {n_act_dep} actuals")
            except RuntimeError as _e:
                print(f"   [3a] skipped (rate-limited): {_e}")
                n_act_dep = 0
            n_act = n_act_arr + n_act_dep
        else:
            print("\n[3] (skipped actuals)")

        # ---- 3b. chain-walk inbound_fa_flight_id → hydrate lineage on demand ----
        if args.chain_walk_max > 0:
            print("\n[3b] Chain-walk inbound_fa_flight_id...")
            n_chain_calls, n_chain_actuals = chain_walk_inbound(
                conn,
                sched_rows + arr_sched_rows,
                max_calls=args.chain_walk_max,
            )
            print(
                f"   chain-walk: {n_chain_calls} AeroAPI calls "
                f"(~${n_chain_calls * 0.005:.2f} USD), {n_chain_actuals} actuals hydrated"
            )
        else:
            print("\n[3b] (chain-walk disabled)")

    # ---- 4. weather ----
    n_wx = 0
    if not args.no_weather:
        print("\n[4] IEM METAR refresh...")
        # Only fetch weather for airports that appear in today's flights (~40-50),
        # not the full 326-airport universe (would exceed 300s task timeout).
        active_airports: set[str] = set()
        for r in sched_rows + arr_sched_rows:
            if r.get("origin"):
                active_airports.add(r["origin"])
            if r.get("dest"):
                active_airports.add(r["dest"])
        active_airports &= AIRPORTS  # keep only ones we have IEM network info for
        print(f"   fetching weather for {len(active_airports)} airports (from today's flights)")
        wx = fetch_iem_obs(active_airports, sched_start - timedelta(hours=2),
                           sched_end + timedelta(hours=2))
        if not wx.empty:
            n_wx = upsert_weather(conn, wx)
            print(f"   upserted {n_wx} weather observations")

    # ---- 4a. Bootstrap unseen tails (Layer 2 lineage fix) ----
    # When a tail appears in today's schedule but is not in `tail_lineage_cache`,
    # insert a placeholder row with hydrated_until in the past. The harvester's
    # `select_tails_to_hydrate` prioritizes never > expired, so unseen tails get
    # chain-walked within ~30 min of first appearance instead of relying on
    # the fallback for that tail's entire schedule.
    target_tails = set()
    for r in sched_rows + arr_sched_rows:
        t = r.get("tail_num") or r.get("registration")
        if t and str(t).strip():
            target_tails.add(str(t).strip())
    if target_tails:
        placeholders = ",".join(["?"] * len(target_tails))
        cur = conn.execute(
            f"SELECT tail FROM tail_lineage_cache WHERE tail IN ({placeholders})",
            list(target_tails),
        )
        cached_tails = {r[0] for r in cur.fetchall()}
        new_tails = target_tails - cached_tails
        if new_tails:
            conn.executemany(
                "INSERT OR IGNORE INTO tail_lineage_cache(tail, hydrated_until, "
                "last_pull_source, last_pull_ok, consecutive_failures) "
                "VALUES (?, '1970-01-01T00:00:00+00:00', 'bootstrap-request', 0, 0)",
                [(t,) for t in new_tails],
            )
            conn.commit()
            print(f"\n[4a] bootstrap: queued {len(new_tails)} unseen tails for next harvester tick")
        else:
            print(f"\n[4a] bootstrap: all {len(target_tails)} target tails already cached")

    # ---- 4b. FAA NAS Status snapshot (Tier 2 #K) ----
    # Capture programs (GDP/GS/Closure) active right now. Persists into
    # nas_status table for both historical dataset building and immediate
    # post-prediction adjustment below.
    n_nas = snapshot_nas_status(conn)
    if n_nas > 0:
        print(f"\n[4b] NAS snapshot: {n_nas} airports under active programs")
    else:
        print("\n[4b] NAS snapshot: no active programs (or fetch failed)")

    # ---- 5. predict scheduled flights ----
    print("\n[5] Building features and predicting...")
    # Decouple the prediction target from this cycle's fetch. Predicting only the
    # flights pulled this tick means each flight is scored ~once, minutes before
    # departure (the scheduled_departures feed only surfaces near-term flights).
    # Instead, every cycle re-predict ALL upcoming ATL departures whose departure
    # is still AHEAD — so a fresh PRE-departure prediction always exists and
    # refines as the gate nears.
    #
    # "Still upcoming" = wheels-off not yet recorded (actual_off_utc IS NULL) AND
    # the best available departure estimate is in the future. We use
    # COALESCE(estimated_out_utc, scheduled_out_utc) so the predicate is
    # delay-aware: a delayed flight whose estimated_out slips into the future
    # stays in the target, while a flight that already departed but hasn't been
    # settled yet (actual_off lag) is correctly excluded — without this, ~125
    # departed-but-unsettled flights/cycle would get spurious post-departure
    # predictions.
    horizon_h = int(os.getenv("PREDICT_HORIZON_HOURS", str(args.schedule_hours)))
    db_dep_rows = conn.execute(
        """
        SELECT f.fa_flight_id
        FROM flights f
        LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
        WHERE f.origin = 'ATL'
          AND f.scheduled_out_utc IS NOT NULL
          AND a.actual_off_utc IS NULL
          AND datetime(COALESCE(f.estimated_out_utc, f.scheduled_out_utc)) > datetime(?)
          AND datetime(f.scheduled_out_utc) <= datetime(?, ?)
          AND COALESCE(f.cancelled, 0) = 0
        """,
        (_iso(now), _iso(now), f"+{horizon_h} hours"),
    ).fetchall()
    # Union the standing upcoming-departures set with this cycle's freshly fetched
    # flights (keeps arrivals + brand-new departures not yet committed-visible).
    target_ids = list(
        {r[0] for r in db_dep_rows}
        | {r["fa_flight_id"] for r in sched_rows + arr_sched_rows}
    )
    print(f"   target: {len(db_dep_rows)} standing upcoming ATL deps + "
          f"{len(sched_rows) + len(arr_sched_rows)} freshly fetched → {len(target_ids)} unique")
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

    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = None
    # Lineage fallback is default-on. The DISABLE_LINEAGE_FALLBACK env var was a
    # diagnostic flag during the SIGSEGV hunt (root cause turned out to be CRLF
    # in model.lgb, not the fallback). Keep the override path for repro
    # scenarios but never default to bypassing — without the fallback, NaN
    # lineage features force the model to extrapolate via TAIL_DELAY_DECAY and
    # other proxies, which sobre-estimates and degrades live AUC.
    if os.getenv("DISABLE_LINEAGE_FALLBACK", "").lower() in ("1", "true", "yes"):
        print("   lineage fallback disabled via DISABLE_LINEAGE_FALLBACK env (DIAGNOSTIC ONLY — remove this env var in production)")
    elif fallback_path.exists():
        try:
            fallback = load_lookups(fallback_path)
        except Exception as e:
            print(f"   ⚠ fallback load failed (pickle/pandas version mismatch): {e}")
            print("   proceeding without lineage fallback")
    else:
        print(f"   ⚠ lineage fallback artifact missing at {fallback_path} — predictions will use NaN priors (degrades AUC)")
    if fallback is not None:
        print(f"   loaded cold-deck fallback ({fallback_path.name})")

    # Layer 3 fallback: enrich `fallback` with live-DB turnaround aggregations
    # so prev_turnaround_tail_min cascades through (carrier, route, hour)
    # buckets rather than collapsing to the 60-min constant. Computed each tick
    # from the local copy of `flights` -- cheap (~50ms) and self-updating.
    if fallback is None:
        fallback = {}
    try:
        live_turnaround = build_live_turnaround_lookups(conn, days=14)
        fallback.update(live_turnaround)
        meta_t = live_turnaround.get("_meta", {})
        print(
            f"   live turnaround lookups: n_rows={meta_t.get('n_rows', 0)} "
            f"global={live_turnaround.get('global_turnaround_mean', 60.0):.1f}min "
            f"carrier_buckets={len(live_turnaround.get('turnaround_carrier_mean', []))} "
            f"route_buckets={len(live_turnaround.get('turnaround_carrier_route_mean', []))}"
        )
    except Exception as e:
        print(f"   ⚠ live turnaround lookup build failed: {e} (cascade disabled, falls back to 60.0)")

    # ---- Feature NaN logging & Quality assertions ----
    X_raw = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=None,
    )
    
    target_idx = df.index[target_mask]
    if not target_idx.empty:
        X_raw_target = X_raw.loc[target_idx].copy()
        
        # Ensure numeric columns are parsed as numeric for accurate NaN checks
        cat_cols_set = set(meta.get("cat_cols", []))
        for c in X_raw_target.columns:
            if c not in cat_cols_set:
                X_raw_target[c] = pd.to_numeric(X_raw_target[c], errors="coerce")
        
        print("   Raw Feature NaN Rates (before cold-deck fallback) for target flights:")
        col_nan_rates = X_raw_target.isna().mean()
        sorted_col_nans = sorted(col_nan_rates.items(), key=lambda x: x[1], reverse=True)
        for col, rate in sorted_col_nans:
            if rate > 0.0:
                print(f"     - {col}: {rate:.1%}")
                
        nan_rates_per_flight = X_raw_target.isna().mean(axis=1)
        too_many_nans = nan_rates_per_flight[nan_rates_per_flight > 0.6]
        if not too_many_nans.empty:
            print(f"   Quality Assertion: Skipping prediction for {len(too_many_nans)} flights with >60% NaN features:")
            for idx, rate in too_many_nans.items():
                fl_id = df.loc[idx, "fa_flight_id"]
                print(f"     - {fl_id}: {rate:.1%} NaN features")
            
            # Exclude flights that failed quality assertion from prediction and database insert
            target_mask = target_mask & (~df.index.isin(too_many_nans.index))
            print(f"   {target_mask.sum()} target rows remaining after quality filtering")
    # --------------------------------------------------

    X = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )
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

    # Threshold strategy: quantile-target on the target subset (robust to live
    # distribution shift); fall back to the artifact's static threshold when
    # --target-pos-rate=0 or when the target batch is too small to estimate.
    target_proba = proba[target_mask.to_numpy()]
    if args.target_pos_rate > 0 and target_proba.size >= 5:
        threshold_used = quantile_threshold(target_proba, args.target_pos_rate)
        threshold_strategy = f"quantile@{args.target_pos_rate:.2f}"
    else:
        threshold_used = float(meta["threshold"])
        threshold_strategy = "artifact"
    labels = predict_label(proba, threshold_used, "binary")
    print(
        f"   threshold strategy={threshold_strategy} value={threshold_used:.4f} "
        f"| proba_target n={target_proba.size} mean={target_proba.mean():.3f} "
        f"std={target_proba.std():.3f} pos_pred_rate="
        f"{(target_proba >= threshold_used).mean():.3f}"
        if target_proba.size > 0
        else f"   threshold strategy={threshold_strategy} value={threshold_used:.4f} (no targets)"
    )

    # Post-prediction GDP adjustment (Tier 2 #K). The v9 model was trained
    # without GDP_FLAG in its feature_cols, so even if the live feature
    # pipeline computes GDP_FLAG it doesn't reach the booster. We compensate
    # by adjusting `proba` after the fact when ORIGIN or DEST is under a
    # program. `proba_raw` is preserved so we can A/B test the lift.
    nas_state = latest_nas_status(conn, max_age_minutes=30)
    gdp_adjust_enabled = os.getenv("GDP_ADJUST", "1").lower() in ("1", "true", "yes")
    if nas_state and gdp_adjust_enabled:
        affected = [a for a in nas_state if a in {df.loc[i, "origin"] for i in df.index[target_mask]} | {df.loc[i, "dest"] for i in df.index[target_mask]}]
        if affected:
            print(f"   GDP adjustment: {len(affected)} target-relevant airports under program: {affected}")
    elif not gdp_adjust_enabled:
        print("   GDP adjustment disabled via GDP_ADJUST env")

    # Pre-fetch intermediate dep_delay for all target stable_ids in one query.
    target_stable_ids = [stable_id(df.loc[i, "fa_flight_id"]) for i in df.index[target_mask]]
    dep_delay_map: dict[str, float] = {}
    if target_stable_ids:
        placeholders = ",".join("?" for _ in target_stable_ids)
        for row in conn.execute(
            f"""SELECT stable_id, departure_delay_min FROM actuals
               WHERE stable_id IN ({placeholders})
                 AND departure_delay_min IS NOT NULL
                 AND actual_in_utc IS NULL""",  # only flights that took off but haven't landed
            target_stable_ids,
        ).fetchall():
            dep_delay_map[row[0]] = float(row[1])

    dep_adjust_enabled = os.getenv("DEP_DELAY_ADJUST", "1").lower() in ("1", "true", "yes")
    if dep_delay_map and dep_adjust_enabled:
        print(f"   intermediate dep_delay available for {len(dep_delay_map)} targets")

    adsb_enabled = os.getenv("ADSB_ADJUST", "1").lower() in ("1", "true", "yes")

    # Persist only target predictions
    pred_now = datetime.now(timezone.utc).isoformat()
    pred_rows: list[tuple] = []
    for i in df.index[target_mask]:
        proba_raw = float(proba[i])
        origin = df.loc[i, "origin"]
        dest = df.loc[i, "dest"]
        gdp_orig = nas_state.get(origin, {}).get("delay_min", 0.0) if nas_state else 0.0
        gdp_dest = nas_state.get(dest, {}).get("delay_min", 0.0) if nas_state else 0.0

        # Adjustment chain: raw → GDP → intermediate dep_delay → ADS-B ETA.
        proba_after_gdp = (
            gdp_post_prediction_adjust(proba_raw, gdp_orig, gdp_dest)
            if gdp_adjust_enabled
            else proba_raw
        )
        sid = stable_id(df.loc[i, "fa_flight_id"])
        dep_delay = dep_delay_map.get(sid)  # None if target hasn't departed yet
        proba_after_dep = (
            intermediate_dep_delay_adjust(proba_after_gdp, dep_delay)
            if dep_adjust_enabled
            else proba_after_gdp
        )
        # Tier 3 #3 — ADS-B ETA boost (only for arrivals to ATL in air now)
        adsb_delay = (
            compute_adsb_eta_delay(
                conn,
                tail_num=df.loc[i, "tail_num"],
                scheduled_in_utc=df.loc[i, "scheduled_in_utc"],
                dest=df.loc[i, "dest"],
            )
            if adsb_enabled
            else None
        )
        proba_after_eta = (
            adsb_eta_adjust(proba_after_dep, adsb_delay) if adsb_enabled else proba_after_dep
        )

        # Mid win #5 — ADS-B holding pattern detection (orbiting near ATL)
        holding_min = (
            compute_adsb_holding_min(
                conn,
                tail_num=df.loc[i, "tail_num"],
                dest=df.loc[i, "dest"],
            )
            if adsb_enabled
            else None
        )
        proba_adj = (
            adsb_holding_adjust(proba_after_eta, holding_min) if adsb_enabled else proba_after_eta
        )
        label_adj = int(proba_adj >= threshold_used)

        # Diagnostic features (Tier 2 #I, #J) — computed live, NOT in
        # feature_cols of v9. Persisted for future v9.1 retrain analysis.
        atl_window = compute_atl_arrival_congestion(
            conn, df.loc[i, "scheduled_in_utc"], window_minutes=30,
        )
        carrier_smooth = carrier_delay_rate_bayesian(
            conn, df.loc[i, "op_carrier"], df.loc[i, "scheduled_off_utc"],
            window_hours=24.0, alpha=20.0, prior_window_hours=168.0,
        )
        pred_rows.append((
            df.loc[i, "fa_flight_id"], sid,
            pred_now, float(proba_adj), label_adj,
            float(threshold_used), threshold_strategy,
            proba_raw, float(gdp_orig), float(gdp_dest),
            int(atl_window), (float(carrier_smooth) if carrier_smooth is not None else None),
            (float(dep_delay) if dep_delay is not None else None),
            (float(adsb_delay) if adsb_delay is not None else None),
            (float(holding_min) if holding_min is not None else None),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
            threshold_used, threshold_strategy,
            proba_raw, gdp_orig_delay_min, gdp_dest_delay_min,
            atl_arrivals_in_window_30min, carrier_delay_rate_smooth,
            intermediate_dep_delay_min, adsb_eta_delay_min, adsb_holding_min)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        pred_rows,
    )
    conn.commit()
    # r[3]=proba_delay (final), r[7]=proba_raw, r[12]=intermediate_dep_delay_min,
    # r[13]=adsb_eta_delay_min, r[14]=adsb_holding_min
    n_any = sum(1 for r in pred_rows if r[7] is not None and abs(r[7] - r[3]) > 1e-6)
    n_dep = sum(1 for r in pred_rows if r[12] is not None and r[12] > 5)
    n_adsb_available = sum(1 for r in pred_rows if r[13] is not None)
    n_adsb_boost = sum(1 for r in pred_rows if r[13] is not None and r[13] > 5)
    n_holding = sum(1 for r in pred_rows if r[14] is not None and r[14] >= 5)
    print(f"   wrote {len(pred_rows)} predictions ({n_any} adjusted, "
          f"{n_dep} via dep_delay, {n_adsb_available} with adsb_eta "
          f"of which {n_adsb_boost} boosted, {n_holding} in holding pattern)")

    # ---- SHAP top-K persistence (Fix D in FIXES_PLAN.md) ----
    # Compute SHAP values for target rows only, persist top-15 by |shap|.
    # Disabled if SHAP_TOPK=0. Failure is non-fatal (predictions already written).
    shap_topk = int(os.getenv("SHAP_TOPK", "15"))
    if shap_topk > 0 and pred_rows:
        try:
            target_indices = df.index[target_mask]
            X_target = X.loc[target_indices]
            # LightGBM native: pred_contrib=True returns per-row SHAP vector + bias term.
            contribs = meta["booster"].predict(X_target, pred_contrib=True)
            # contribs shape: (n_targets, n_features + 1) — last col is bias
            feat_names = list(X_target.columns)
            shap_matrix = contribs[:, :-1]

            shap_rows: list[tuple] = []
            for row_idx, df_idx in enumerate(target_indices):
                fa_id = df.loc[df_idx, "fa_flight_id"]
                row_shap = shap_matrix[row_idx]
                # Rank by absolute contribution
                abs_order = np.argsort(-np.abs(row_shap))[:shap_topk]
                for rank, feat_idx in enumerate(abs_order, start=1):
                    fname = feat_names[feat_idx]
                    sval = float(row_shap[feat_idx])
                    fval = X_target.iloc[row_idx][fname]
                    fval_str = "NaN" if pd.isna(fval) else str(fval)
                    shap_rows.append((fa_id, pred_now, fname, sval, fval_str, rank))

            conn.executemany(
                """INSERT OR REPLACE INTO prediction_shap
                   (fa_flight_id, predicted_at_utc, feature_name, shap_value,
                    feature_value, rank)
                   VALUES (?,?,?,?,?,?)""",
                shap_rows,
            )
            conn.commit()
            print(f"   wrote {len(shap_rows)} SHAP values (top-{shap_topk} × {len(pred_rows)} preds)")
        except Exception as e:
            print(f"   ⚠ SHAP persistence failed (non-fatal): {type(e).__name__}: {e}")

    conn.execute(
        "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?, actuals_updated=?, weather_obs_added=? WHERE run_id=?",
        (datetime.now(timezone.utc).isoformat(),
         len(sched) + len(arr_sched) + len(arrived),
         len(pred_rows), n_act, n_wx, run_id),
    )
    conn.commit()
    # Close the connection so SQLite flushes cached pages to disk BEFORE the
    # subsequent GCS upload in live_job.py reads /tmp/live_data.db.
    conn.close()

    print(f"\nDone. run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
