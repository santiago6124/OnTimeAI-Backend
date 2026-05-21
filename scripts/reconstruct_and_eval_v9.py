"""Reconstruct database and evaluate v9 on all live data (May 4 to May 8, 2026) with zero leakage.

This script loads the historical live snapshots (May 4-8, 2026), downloads
the public IEM weather data for the exact period, and generates predictions for
model v9 under a strict, time-locked, leak-free retrospective backtest.
"""
from __future__ import annotations

import os
import json
import sqlite3
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score, accuracy_score
)

# Set warnings to ignore to keep output clean
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "live_backtest.db"

# Re-use the airport list and timezones from ontimeai
from ontimeai.live import AIRPORTS, TZ_BY_AIRPORT, fetch_iem_obs, upsert_weather, stable_id
from ontimeai.model import load_artifact, predict_proba, predict_label
from predict import prepare_inference_frame

# DB schema definition
SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    fa_flight_id TEXT PRIMARY KEY,
    stable_id TEXT,
    ident_iata TEXT,
    op_carrier TEXT,
    flight_number TEXT,
    tail_num TEXT,
    origin TEXT,
    dest TEXT,
    inbound_fa_flight_id TEXT,
    fl_date TEXT,
    crs_dep_min INTEGER,
    scheduled_out_utc TEXT,
    scheduled_off_utc TEXT,
    scheduled_on_utc TEXT,
    scheduled_in_utc TEXT,
    crs_elapsed_min REAL,
    distance REAL,
    aircraft_type TEXT,
    cancelled INTEGER,
    diverted INTEGER,
    first_seen_utc TEXT,
    last_updated_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_flights_date_dep ON flights(fl_date, scheduled_off_utc);
CREATE INDEX IF NOT EXISTS idx_flights_tail ON flights(tail_num, scheduled_off_utc);
CREATE INDEX IF NOT EXISTS idx_flights_carrier ON flights(op_carrier, scheduled_off_utc);
CREATE INDEX IF NOT EXISTS idx_flights_stable ON flights(stable_id);

CREATE TABLE IF NOT EXISTS actuals (
    fa_flight_id TEXT PRIMARY KEY,
    stable_id TEXT,
    actual_out_utc TEXT,
    actual_off_utc TEXT,
    actual_on_utc TEXT,
    actual_in_utc TEXT,
    arr_delay_min REAL,
    departure_delay_min REAL,
    cancelled INTEGER,
    diverted INTEGER,
    settled_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actuals_stable ON actuals(stable_id);

CREATE TABLE IF NOT EXISTS weather_obs (
    station TEXT NOT NULL,
    valid_utc TEXT NOT NULL,
    tmpc REAL, dwpc REAL, relh REAL, drct REAL, sknt REAL, alti REAL,
    p01m REAL, vsby REAL, gust REAL, wxcodes TEXT,
    wx_precip_flag INTEGER, wx_low_vis_flag INTEGER, wx_strong_wind_flag INTEGER,
    PRIMARY KEY (station, valid_utc)
);
CREATE INDEX IF NOT EXISTS idx_weather_station_time ON weather_obs(station, valid_utc);
"""

def build_flights_table_rows(preds_df: pd.DataFrame, actuals_df: pd.DataFrame) -> list[dict]:
    """Merge predictions and actuals metadata to reconstruct flight records."""
    flights_map = {}
    
    # 1. Load distance lookup for distance backfill
    dist_lookup_path = PROJECT_ROOT / "artifacts" / "distance_lookup.csv"
    dist_dict = {}
    if dist_lookup_path.exists():
        dist_df = pd.read_csv(dist_lookup_path)
        dist_dict = dict(zip(zip(dist_df["ORIGIN"], dist_df["DEST"]), dist_df["DISTANCE"]))

    def process_row(fid, ident, tail, carrier, origin, dest, sched_off, sched_on, sched_in, inbound, actype, canc, div):
        if not fid or fid in flights_map:
            return
        
        origin = str(origin).upper().strip()
        dest = str(dest).upper().strip()
        
        # Determine times
        if pd.isna(sched_off) or not sched_off:
            return  # Needs scheduled_off_utc to build features
        
        sched_off_ts = pd.to_datetime(sched_off, utc=True).tz_convert(None)
        
        if pd.isna(sched_in) or not sched_in:
            if not pd.isna(sched_on) and sched_on:
                sched_in_ts = pd.to_datetime(sched_on, utc=True).tz_convert(None) + timedelta(minutes=10)
            else:
                sched_in_ts = sched_off_ts + timedelta(minutes=120)
        else:
            sched_in_ts = pd.to_datetime(sched_in, utc=True).tz_convert(None)
            
        elapsed_min = float((sched_in_ts - sched_off_ts).total_seconds() / 60)
        
        # Local date and crs_dep_min
        tz = ZoneInfo(TZ_BY_AIRPORT.get(origin, "UTC"))
        local_dt = sched_off_ts.tz_localize("UTC").astimezone(tz)
        fl_date = local_dt.strftime("%Y-%m-%d")
        crs_dep_min = local_dt.hour * 60 + local_dt.minute
        
        # Flight number
        fl_num = "".join(filter(str.isdigit, str(ident))) if pd.notna(ident) else None
        
        # Distance
        dist = dist_dict.get((origin, dest))
        if dist is None:
            # Fallback approximate distance formula
            dist = 500.0 if origin != dest else 0.0
            
        flights_map[fid] = {
            "fa_flight_id": fid,
            "stable_id": stable_id(fid),
            "ident_iata": ident if pd.notna(ident) else None,
            "op_carrier": carrier if pd.notna(carrier) else None,
            "flight_number": fl_num,
            "tail_num": tail if pd.notna(tail) else None,
            "origin": origin,
            "dest": dest,
            "inbound_fa_flight_id": inbound if pd.notna(inbound) else None,
            "fl_date": fl_date,
            "crs_dep_min": crs_dep_min,
            "scheduled_out_utc": (sched_off_ts - timedelta(minutes=10)).isoformat(),
            "scheduled_off_utc": sched_off_ts.isoformat(),
            "scheduled_on_utc": (sched_in_ts - timedelta(minutes=10)).isoformat(),
            "scheduled_in_utc": sched_in_ts.isoformat(),
            "crs_elapsed_min": elapsed_min,
            "distance": float(dist),
            "aircraft_type": actype if pd.notna(actype) else None,
            "cancelled": 1 if canc == 1 else 0,
            "diverted": 1 if div == 1 else 0,
            "first_seen_utc": sched_off,
            "last_updated_utc": sched_off
        }

    # Process all rows in predictions snapshot
    for r in preds_df.itertuples():
        process_row(
            r.fa_flight_id, getattr(r, "ident_iata", None), getattr(r, "tail_num", None),
            getattr(r, "op_carrier", None), getattr(r, "origin", None), getattr(r, "dest", None),
            getattr(r, "scheduled_off_utc", None), getattr(r, "scheduled_on_utc", None),
            None, getattr(r, "inbound_fa_flight_id", None), getattr(r, "aircraft_type", None),
            0, 0
        )
        
    # Process all rows in actuals snapshot (many history flights only present here)
    for r in actuals_df.itertuples():
        # Derive a scheduled_off_utc from scheduled_in_utc for history if missing
        sched_in = getattr(r, "scheduled_in_utc", None)
        if pd.isna(sched_in) or not sched_in:
            continue
        sched_in_ts = pd.to_datetime(sched_in, utc=True).tz_convert(None)
        sched_off_ts = sched_in_ts - timedelta(minutes=120)  # default duration estimate
        
        process_row(
            r.fa_flight_id, getattr(r, "ident_iata", None), getattr(r, "tail_num", None),
            getattr(r, "op_carrier", None), getattr(r, "origin", None), getattr(r, "dest", None),
            sched_off_ts.isoformat(), None, sched_in, None, None,
            getattr(r, "cancelled", 0), getattr(r, "diverted", 0)
        )
        
    return list(flights_map.values())

def populate_database():
    """Populates the database live_backtest.db from snapshot CSVs and downloads weather."""
    print("[1/4] Loading CSV Snapshots...")
    preds_path = PROJECT_ROOT / "artifacts" / "live_snapshots" / "predictions_2026-05-08_final.csv"
    actuals_path = PROJECT_ROOT / "artifacts" / "live_snapshots" / "actuals_2026-05-08_final.csv"
    
    preds_df = pd.read_csv(preds_path)
    actuals_df = pd.read_csv(actuals_path)
    
    # Recreate DB file
    if DB_PATH.exists():
        DB_PATH.unlink()
        
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    conn.commit()
    
    print("[2/4] Reconstructing Flights and Actuals tables...")
    flight_rows = build_flights_table_rows(preds_df, actuals_df)
    
    # Insert flights
    payload_flights = [
        (r["fa_flight_id"], r["stable_id"], r["ident_iata"], r["op_carrier"], r["flight_number"],
         r["tail_num"], r["origin"], r["dest"], r["inbound_fa_flight_id"], r["fl_date"], r["crs_dep_min"],
         r["scheduled_out_utc"], r["scheduled_off_utc"], r["scheduled_on_utc"], r["scheduled_in_utc"],
         r["crs_elapsed_min"], r["distance"], r["aircraft_type"], r["cancelled"], r["diverted"],
         r["first_seen_utc"], r["last_updated_utc"])
        for r in flight_rows
    ]
    conn.executemany(
        """INSERT INTO flights VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        payload_flights
    )
    
    # Insert actuals
    payload_actuals = []
    for r in actuals_df.itertuples():
        act_in = getattr(r, "actual_in_utc", None)
        if pd.notna(act_in) and act_in:
            ts = pd.to_datetime(act_in, utc=True).tz_convert(None)
            settled = (ts + timedelta(minutes=30)).isoformat()
        else:
            sched_in = getattr(r, "scheduled_in_utc", None)
            if pd.notna(sched_in) and sched_in:
                ts = pd.to_datetime(sched_in, utc=True).tz_convert(None)
                settled = (ts + timedelta(minutes=30)).isoformat()
            else:
                settled = "2026-05-08T23:59:59"
                
        payload_actuals.append((
            r.fa_flight_id, stable_id(r.fa_flight_id),
            getattr(r, "actual_out_utc", None), getattr(r, "actual_off_utc", None),
            getattr(r, "actual_on_utc", None), getattr(r, "actual_in_utc", None),
            getattr(r, "arr_delay_min", None), getattr(r, "departure_delay_min", None),
            int(getattr(r, "cancelled", 0)), int(getattr(r, "diverted", 0)),
            settled
        ))
    conn.executemany(
        """INSERT INTO actuals VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        payload_actuals
    )
    conn.commit()
    print(f"      Upserted {len(flight_rows)} flights and {len(payload_actuals)} actuals.")
    
    print("[3/4] Fetching Weather observations from IEM (May 3 - May 9, 2026)...")
    active_airports = set(preds_df["origin"].dropna().unique()) | set(preds_df["dest"].dropna().unique())
    from ontimeai.live import NETWORK_BY_AIRPORT
    legacy_hubs = {"ATL", "LGA", "MCO", "FLL", "MIA", "DFW", "DCA", "EWR", "TPA", "ORD", "PHL", "DEN", "LAX", "BWI", "LAS", "BOS"}
    active_airports = {ap for ap in active_airports if ap in NETWORK_BY_AIRPORT and ap in legacy_hubs}
    print(f"      Restricting weather fetch to {len(active_airports)} active ASOS hubs: {active_airports}")
    wx = fetch_iem_obs(active_airports, pd.Timestamp("2026-05-03"), pd.Timestamp("2026-05-09"))
    if not wx.empty:
        n_wx = upsert_weather(conn, wx)
        print(f"      Upserted {n_wx} weather observations.")
    else:
        print("      ⚠️ Weather fetch failed or returned empty. Inbound falling back to cold-deck.")
        
    conn.close()
    print("[4/4] Database Populated Successfully.")

# Custom weather merge function with temporal lock to prevent future weather leakage
def _merge_weather_asof_leak_free(df: pd.DataFrame, wx: pd.DataFrame, station_col: str, time_col: str, prefix: str, max_valid_utc: str) -> pd.DataFrame:
    """ASOF join that strictly limits weather observations to those valid BEFORE the prediction time."""
    if df[time_col].isna().all():
        return df
    
    # Filter weather observations to only those valid BEFORE the prediction time
    wx_filtered = wx[wx["valid_utc"] <= max_valid_utc].copy()
    if wx_filtered.empty:
        # Fill with NaN if no weather was historically available before prediction time
        for col in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes", "precip_flag", "low_vis_flag", "strong_wind_flag"]:
            df[f"{prefix}_WX_{col.upper()}"] = pd.NA
        df[f"{prefix}_WX_MATCH_GAP_MIN"] = pd.NA
        return df

    # Prepare ASOF join keys
    df["_join_time"] = pd.to_datetime(df[time_col])
    wx_filtered["_join_time"] = wx_filtered["valid"]
    
    # Rename columns in weather observations before merging
    rename_cols = {
        "valid": f"{prefix}_WX_VALID_UTC",
        "wxcodes": f"{prefix}_WX_CODES",
        "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
        "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
        "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
    }
    for c in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]:
        rename_cols[c] = f"{prefix}_WX_{c.upper()}"
        
    wx_merged = wx_filtered.rename(columns=rename_cols)
    
    wx_merged["station"] = wx_merged["station"].astype(str)
    df_sorted = df.sort_values("_join_time").copy()
    df_sorted[station_col] = df_sorted[station_col].astype(str)
    
    merged = pd.merge_asof(
        df_sorted,
        wx_merged.sort_values("_join_time"),
        on="_join_time",
        left_by=station_col,
        right_by="station",
        direction="nearest",
        tolerance=pd.Timedelta(hours=2)
    )
    
    # Compute gap
    gap_min = (merged["_join_time"] - pd.to_datetime(merged[f"{prefix}_WX_VALID_UTC"])).dt.total_seconds() / 60
    merged[f"{prefix}_WX_MATCH_GAP_MIN"] = gap_min.abs()
    
    merged = merged.drop(columns=["_join_time"])
    return merged.sort_index()

def build_inference_frame_leak_free_fast(
    fa_flight_id: str, predicted_at_utc: str, flights_df: pd.DataFrame, 
    history_pool_df: pd.DataFrame, weather_df: pd.DataFrame, history_days: int = 7
) -> pd.DataFrame:
    """Build inference frame for a specific flight with a hard temporal lock using preloaded DataFrames.
    
    Filters actuals table to ONLY records settled before `predicted_at_utc`.
    This mimics EXACTLY what was known in production at the prediction moment T.
    """
    # 1. Pull the target flight
    target = flights_df[flights_df["fa_flight_id"] == fa_flight_id].copy()
    if target.empty:
        return pd.DataFrame()
        
    # 2. Pull history flights that settled BEFORE prediction time T (Strict Leak-Free constraint)
    earliest = (pd.to_datetime(predicted_at_utc, utc=True) - timedelta(days=history_days)).isoformat()
    history = history_pool_df[
        (history_pool_df["scheduled_off_utc"] >= earliest) &
        (history_pool_df["settled_at_utc"] <= predicted_at_utc)
    ].copy()
    
    df = pd.concat([
        history.assign(_role="history"),
        target.assign(_role="target", arr_delay_min=np.nan),
    ], ignore_index=True)
    
    # 3. Map to model schema
    df["FL_DATE"] = pd.to_datetime(df["fl_date"])
    df["YEAR"] = df["FL_DATE"].dt.year
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.weekday + 1
    df["OP_CARRIER"] = df["op_carrier"].astype("string")
    df["TAIL_NUM"] = df["tail_num"].astype("string")
    df["OP_CARRIER_FL_NUM"] = pd.to_numeric(df["flight_number"], errors="coerce")
    df["ORIGIN"] = df["origin"].astype("string")
    df["DEST"] = df["dest"].astype("string")
    df["CRS_DEP_MIN"] = pd.to_numeric(df["crs_dep_min"], errors="coerce")
    df["CRS_DEP_TIME"] = (df["CRS_DEP_MIN"] // 60) * 100 + (df["CRS_DEP_MIN"] % 60)
    df["CRS_ELAPSED_TIME"] = pd.to_numeric(df["crs_elapsed_min"], errors="coerce")
    df["DISTANCE"] = pd.to_numeric(df["distance"], errors="coerce")
    df["CANCELLED"] = df["cancelled"].fillna(0).astype(int)
    df["DIVERTED"] = df["diverted"].fillna(0).astype(int)
    df["ARR_DELAY"] = pd.to_numeric(df["arr_delay_min"], errors="coerce")
    
    df["EVENT_ORIGIN_UTC"] = pd.to_datetime(df["scheduled_off_utc"], errors="coerce").astype("datetime64[ns]")
    df["EVENT_DEST_UTC"] = pd.to_datetime(df["scheduled_on_utc"], errors="coerce").astype("datetime64[ns]")
    df["DEP_LOCAL_DT"] = df["FL_DATE"] + pd.to_timedelta(df["CRS_DEP_MIN"], unit="m")
    
    df["FLOW_ATL"] = np.where(df["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    df["PAR_AIRPORT"] = np.where(df["ORIGIN"].eq("ATL"), df["DEST"], df["ORIGIN"])
    
    # 4. Leak-Free Weather Merge
    if df["EVENT_ORIGIN_UTC"].notna().any() and not weather_df.empty:
        df = _merge_weather_asof_leak_free(df, weather_df, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG", predicted_at_utc)
        df = _merge_weather_asof_leak_free(df, weather_df, "DEST", "EVENT_DEST_UTC", "DEST", predicted_at_utc)
            
    # 5. Add features
    from ontimeai.live import _add_v7_wind_features, _add_v8_features, _add_aircraft_family
    df = _add_v7_wind_features(df)
    df = _add_v8_features(df)
    df = _add_aircraft_family(df)
    
    # Lineage rollups (using only history resolved before prediction time T)
    from ontimeai.lineage import (
        add_tail_lineage_features, add_carrier_day_lag, add_origin_day_lag,
        add_carrier_rolling_features, add_origin_rolling_features, add_dest_rolling_features
    )
    from ontimeai.features import add_absorb_score
    from ontimeai.data import drop_leaky_target_columns
    df = add_tail_lineage_features(df)
    df = add_carrier_day_lag(df)
    df = add_origin_day_lag(df)
    df = add_carrier_rolling_features(df)
    df = add_origin_rolling_features(df)
    df = add_dest_rolling_features(df)
    df = add_absorb_score(df)
    df = drop_leaky_target_columns(df)
    
    return df

def run_evaluation():
    """Runs v9 Backtest over reconstructed live records and compares with live v7_recal."""
    print("\n" + "=" * 80)
    print("  RUNNING BACKTEST FOR MODEL v9 (May 4-8, 2026)")
    print("  Ensuring strict temporal-lock, leak-free feature building")
    print("=" * 80)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Pre-load entire tables into memory to bypass loop SQL overhead (100x speedup!)
    print("Pre-loading tables into memory for lightning-fast inference...")
    flights_df = pd.read_sql_query("SELECT * FROM flights", conn)
    actuals_df = pd.read_sql_query("SELECT * FROM actuals", conn)
    
    # Pre-merge flights with actuals to form history pool once
    history_pool_df = flights_df.merge(actuals_df, on="stable_id", how="inner", suffixes=("", "_act"))
    history_pool_df = history_pool_df[history_pool_df["stable_id"].notna()]
    if "cancelled" in history_pool_df.columns:
        history_pool_df["act_cancelled"] = history_pool_df["cancelled"]
    if "diverted" in history_pool_df.columns:
        history_pool_df["act_diverted"] = history_pool_df["diverted"]
        
    weather_df = pd.read_sql_query("SELECT * FROM weather_obs", conn)
    if not weather_df.empty:
        weather_df["valid"] = pd.to_datetime(weather_df["valid_utc"]).astype("datetime64[ns]")
        weather_df = weather_df.sort_values(["station", "valid"]).reset_index(drop=True)
        
    # Load v9 artifact
    meta = load_artifact(PROJECT_ROOT / "artifacts" / "4year_v9")
    fallback_path = PROJECT_ROOT / "artifacts" / "lineage_fallback.joblib"
    from ontimeai.lineage_fallback import apply_lineage_fallback, load_lookups
    try:
        fallback = load_lookups(fallback_path) if fallback_path.exists() else None
    except Exception as e:
        print(f"      ⚠️ Cold-deck fallback load failed: {e}. Proceeding without it (LightGBM handles NaNs).")
        fallback = None
    
    # Load all live predictions made during that period (mainly by v7_recal)
    preds_df = pd.read_csv(PROJECT_ROOT / "artifacts" / "live_snapshots" / "predictions_2026-05-08_final.csv")
    actuals_df_snapshot = pd.read_csv(PROJECT_ROOT / "artifacts" / "live_snapshots" / "actuals_2026-05-08_final.csv")
    
    # Filter to flights with non-cancelled, non-diverted settled ground truth
    valid_actuals = actuals_df_snapshot[(actuals_df_snapshot["arr_delay_min"].notna()) & (actuals_df_snapshot["cancelled"] == 0) & (actuals_df_snapshot["diverted"] == 0)]
    
    merged = preds_df.merge(valid_actuals, on="fa_flight_id", how="inner", suffixes=("_pred", "_act"))
    
    # De-duplicate to keep only the LATEST prediction per flight
    merged = merged.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()
    
    print(f"Total matched evaluation flights: {len(merged):,}")
    
    results = []
    
    # Run predictions flight-by-flight (time-locked feature builder)
    print("Predicting flight-by-flight (temporal lock)...")
    for idx, row in enumerate(merged.itertuples()):
        if idx > 0 and idx % 200 == 0:
            print(f"  Processed {idx}/{len(merged)} flights...")
            
        fid = row.fa_flight_id
        pred_time = row.predicted_at_utc
        
        # Build features for this specific flight using fast in-memory method
        df_inf = build_inference_frame_leak_free_fast(
            fid, pred_time, flights_df, history_pool_df, weather_df
        )
        if df_inf.empty:
            continue
            
        # Target row is the last row (we only want to predict the target)
        target_row = df_inf.iloc[[-1]]
        
        if idx == 0:
            print(f"      [DEBUG] First flight ID: {fid}")
            print(f"      [DEBUG] pred_time: {pred_time}")
            print(f"      [DEBUG] history size: {len(df_inf) - 1}")
            lineage_cols = [c for c in target_row.columns if "delay_rate" in c or "prev_arr" in c]
            print(f"      [DEBUG] Lineage features values in target row:")
            for c in lineage_cols:
                print(f"        {c}: {target_row[c].values[0]}")
        
        # Lineage fallback if features were NaN
        if fallback is not None:
            target_row = apply_lineage_fallback(target_row, fallback)
            
        X = prepare_inference_frame(
            target_row, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=None
        )
        
        # Coerce non-categorical object columns to numeric
        cat_cols_set = set(meta.get("cat_cols", []))
        for c in X.columns:
            if c in cat_cols_set:
                continue
            if X[c].dtype == object:
                X[c] = pd.to_numeric(X[c], errors="coerce")
                
        # Predict
        proba = predict_proba(meta["booster"], X)[0]
        if meta.get("calibrator") is not None and meta["target"] == "binary":
            proba = meta["calibrator"].transform(np.array([proba]))[0]
            
        results.append({
            "fa_flight_id": fid,
            "arr_delay_min": row.arr_delay_min,
            "proba_v7": row.proba_delay,
            "proba_v9": float(proba),
            "y_true": 1 if row.arr_delay_min > 15 else 0
        })
        
    conn.close()
    
    # Compute metrics
    eval_df = pd.DataFrame(results)
    print(f"\nSuccessfully predicted {len(eval_df)} flights for model v9.")
    
    # 22% quantile threshold for decision labels
    t_v7 = float(np.quantile(eval_df["proba_v7"], 1 - 0.22))
    t_v9 = float(np.quantile(eval_df["proba_v9"], 1 - 0.22))
    
    pred_v7 = (eval_df["proba_v7"] >= t_v7).astype(int)
    pred_v9 = (eval_df["proba_v9"] >= t_v9).astype(int)
    
    y = eval_df["y_true"].values
    
    metrics = {
        "v7_recal": {
            "AUC": roc_auc_score(y, eval_df["proba_v7"]),
            "Brier": brier_score_loss(y, eval_df["proba_v7"]),
            "F1": f1_score(y, pred_v7),
            "precision": precision_score(y, pred_v7),
            "recall": recall_score(y, pred_v7),
        },
        "v9": {
            "AUC": roc_auc_score(y, eval_df["proba_v9"]),
            "Brier": brier_score_loss(y, eval_df["proba_v9"]),
            "F1": f1_score(y, pred_v9),
            "precision": precision_score(y, pred_v9),
            "recall": recall_score(y, pred_v9),
        }
    }
    
    print("\n" + "=" * 80)
    print("  COMPARATIVE EVALUATION SUMMARY (May 4-8, 2026 Live Period)")
    print("=" * 80)
    print(f"Matched flights count: n = {len(eval_df)}")
    print(f"Actual delay rate (>15 min): {y.mean()*100:.2f}%")
    print("-" * 80)
    print(f"Model v7_recal (Deployed Baseline):")
    print(f"  AUC   = {metrics['v7_recal']['AUC']:.4f}")
    print(f"  Brier = {metrics['v7_recal']['Brier']:.4f}")
    print(f"  F1    = {metrics['v7_recal']['F1']:.4f}  (P={metrics['v7_recal']['precision']:.3f}, R={metrics['v7_recal']['recall']:.3f})")
    print("-" * 80)
    print(f"Model v9 (Time-Locked Backtest):")
    print(f"  AUC   = {metrics['v9']['AUC']:.4f}")
    print(f"  Brier = {metrics['v9']['Brier']:.4f}")
    print(f"  F1    = {metrics['v9']['F1']:.4f}  (P={metrics['v9']['precision']:.3f}, R={metrics['v9']['recall']:.3f})")
    print("-" * 80)
    print(f"Δ (v9 - v7_recal):")
    print(f"  ΔAUC   = {metrics['v9']['AUC'] - metrics['v7_recal']['AUC']:+.4f}")
    print(f"  ΔBrier = {metrics['v9']['Brier'] - metrics['v7_recal']['Brier']:+.4f} (lower is better)")
    print(f"  ΔF1    = {metrics['v9']['F1'] - metrics['v7_recal']['F1']:+.4f}")
    print("=" * 80)

if __name__ == "__main__":
    populate_database()
    run_evaluation()
