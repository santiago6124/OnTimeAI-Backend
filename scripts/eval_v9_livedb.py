"""Evaluate model v9 on ALL live_data.db flights — BATCH approach with leak-safe features.

Strategy to avoid O(n²) per-flight loops:
  1. Build feature matrix for ALL flights at once (lineage, weather, etc.)
     using the FULL history pool — this creates features as if we had omniscient
     knowledge.
  2. Then VALIDATE for leakage: the lineage features (rolling rates, tail chain)
     already enforce temporal ordering by design (they use EVENT_ORIGIN_UTC /
     EVENT_DEST_UTC + ARR_DELAY to compute "what was known before this flight").
  3. Weather is ASOF-merged using the flight's own departure time (not future).

The only TRUE leakage risk is if we include a flight's OWN actual delay in
its features. The lineage module already excludes self (it only looks at
prior flights in sorted order). So batch computation is safe here.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_score, recall_score,
    f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_DB = PROJECT_ROOT / "live_data.db"

from ontimeai.model import load_artifact, predict_proba
from predict import prepare_inference_frame


def main():
    conn = sqlite3.connect(str(LIVE_DB))

    # ── 1. Load everything ────────────────────────────────────────────────
    print("[1/5] Loading live_data.db...")
    flights_df = pd.read_sql("SELECT * FROM flights", conn)
    actuals_df = pd.read_sql("SELECT * FROM actuals", conn)
    preds_df   = pd.read_sql("SELECT * FROM predictions", conn)
    weather_df = pd.read_sql("SELECT * FROM weather_obs", conn)
    conn.close()

    print(f"      flights={len(flights_df):,}  actuals={len(actuals_df):,}  "
          f"preds={len(preds_df):,}  weather={len(weather_df):,}")

    # ── 2. Build evaluation set ───────────────────────────────────────────
    print("[2/5] Building evaluation set...")
    valid_act = actuals_df[
        actuals_df["arr_delay_min"].notna() &
        (actuals_df["cancelled"] == 0) &
        (actuals_df["diverted"] == 0)
    ][["fa_flight_id", "arr_delay_min", "actual_in_utc", "settled_at_utc"]].copy()

    preds_latest = (
        preds_df.sort_values("predicted_at_utc")
        .groupby("fa_flight_id", as_index=False).last()
    )
    eval_set = preds_latest.merge(valid_act, on="fa_flight_id", how="inner")
    eval_set["y_true"] = (eval_set["arr_delay_min"] > 15).astype(int)
    eval_ids = set(eval_set["fa_flight_id"])
    print(f"      Eval flights: {len(eval_set):,}  delay_rate: {eval_set['y_true'].mean()*100:.1f}%")

    # ── 3. Build full feature matrix (batch) ──────────────────────────────
    print("[3/5] Building batch feature matrix for all flights...")

    # Merge flights + actuals (for lineage computation)
    full = flights_df.merge(
        actuals_df[["fa_flight_id", "arr_delay_min", "actual_in_utc"]],
        on="fa_flight_id", how="left",
    )

    # Map to model schema
    full["FL_DATE"]   = pd.to_datetime(full["fl_date"])
    full["YEAR"]      = full["FL_DATE"].dt.year
    full["MONTH"]     = full["FL_DATE"].dt.month
    full["DAY_OF_MONTH"] = full["FL_DATE"].dt.day
    full["DAY_OF_WEEK"]  = full["FL_DATE"].dt.weekday + 1
    full["OP_CARRIER"]   = full["op_carrier"].astype("string")
    full["TAIL_NUM"]     = full["tail_num"].astype("string")
    full["OP_CARRIER_FL_NUM"] = pd.to_numeric(full["flight_number"], errors="coerce")
    full["ORIGIN"] = full["origin"].astype("string")
    full["DEST"]   = full["dest"].astype("string")
    full["CRS_DEP_MIN"] = pd.to_numeric(full["crs_dep_min"], errors="coerce")
    full["CRS_DEP_TIME"] = (full["CRS_DEP_MIN"] // 60) * 100 + (full["CRS_DEP_MIN"] % 60)
    full["CRS_ELAPSED_TIME"] = pd.to_numeric(full["crs_elapsed_min"], errors="coerce")
    full["DISTANCE"]   = pd.to_numeric(full["distance"], errors="coerce")
    full["CANCELLED"]  = full["cancelled"].fillna(0).astype(int)
    full["DIVERTED"]   = full["diverted"].fillna(0).astype(int)
    full["ARR_DELAY"]  = pd.to_numeric(full["arr_delay_min"], errors="coerce")

    full["EVENT_ORIGIN_UTC"] = pd.to_datetime(full["scheduled_off_utc"], errors="coerce").astype("datetime64[ns]")
    full["EVENT_DEST_UTC"]   = pd.to_datetime(full["scheduled_on_utc"], errors="coerce").astype("datetime64[ns]")
    full["DEP_LOCAL_DT"]     = full["FL_DATE"] + pd.to_timedelta(full["CRS_DEP_MIN"], unit="m")

    full["FLOW_ATL"]    = np.where(full["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    full["PAR_AIRPORT"] = np.where(full["ORIGIN"].eq("ATL"), full["DEST"], full["ORIGIN"])

    # ── 3a. Weather ASOF merge ────────────────────────────────────────────
    print("      Merging weather (ASOF)...")
    weather_df["valid"] = pd.to_datetime(
        weather_df["valid_utc"], format="mixed", utc=True
    ).dt.tz_localize(None).astype("datetime64[ns]")
    weather_df.sort_values(["station", "valid"], inplace=True)

    def _safe_wx_merge(df, wx, station_col, event_col, prefix):
        """NaT-safe ASOF weather merge."""
        has_time = df[event_col].notna() & df[station_col].notna()
        df_ok  = df[has_time].copy()
        df_bad = df[~has_time].copy()
        if df_ok.empty:
            return df
        parts = []
        for st in sorted(df_ok[station_col].dropna().unique()):
            lsub = df_ok[df_ok[station_col].eq(st)].sort_values(event_col).copy()
            wsub = wx[wx["station"].eq(st)].sort_values("valid").copy()
            if lsub.empty or wsub.empty:
                parts.append(lsub)
                continue
            merged = pd.merge_asof(
                lsub, wsub, left_on=event_col, right_on="valid",
                direction="nearest", tolerance=pd.Timedelta("90min"),
            )
            parts.append(merged)
        if not parts:
            return df
        out = pd.concat(parts + [df_bad], ignore_index=True)
        rename = {
            "valid": f"{prefix}_WX_VALID_UTC",
            "tmpc": f"{prefix}_WX_TMPC", "dwpc": f"{prefix}_WX_DWPC",
            "relh": f"{prefix}_WX_RELH", "drct": f"{prefix}_WX_DRCT",
            "sknt": f"{prefix}_WX_SKNT", "alti": f"{prefix}_WX_ALTI",
            "p01m": f"{prefix}_WX_P01M", "vsby": f"{prefix}_WX_VSBY",
            "gust": f"{prefix}_WX_GUST", "wxcodes": f"{prefix}_WX_CODES",
            "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
            "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
            "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
        }
        out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
        if event_col in out.columns and f"{prefix}_WX_VALID_UTC" in out.columns:
            out[f"{prefix}_WX_MATCH_GAP_MIN"] = (
                (out[event_col] - out[f"{prefix}_WX_VALID_UTC"]).abs().dt.total_seconds() / 60
            )
        return out.drop(columns=["station"], errors="ignore")

    # Origin weather
    full["_orig_station"] = full["ORIGIN"].astype(str)
    full = _safe_wx_merge(full, weather_df, "_orig_station", "EVENT_ORIGIN_UTC", "ORIG")
    # Dest weather
    full["_dest_station"] = full["DEST"].astype(str)
    full = _safe_wx_merge(full, weather_df, "_dest_station", "EVENT_DEST_UTC", "DEST")

    # ── 3b. Wind, PageRank, GDP, Aircraft ─────────────────────────────────
    print("      Adding v7/v8/v9 features...")
    from ontimeai.live import _add_v7_wind_features, _add_v8_features, _add_aircraft_family
    full = _add_v7_wind_features(full)
    full = _add_v8_features(full)
    full = _add_aircraft_family(full)

    # ── 3c. Lineage features (leak-safe by design) ────────────────────────
    print("      Computing lineage features...")
    from ontimeai.lineage import (
        add_tail_lineage_features, add_carrier_day_lag, add_origin_day_lag,
        add_carrier_rolling_features, add_origin_rolling_features, add_dest_rolling_features,
    )
    from ontimeai.features import add_absorb_score
    from ontimeai.data import drop_leaky_target_columns

    full = add_tail_lineage_features(full)
    full = add_carrier_day_lag(full)
    full = add_origin_day_lag(full)
    full = add_carrier_rolling_features(full)
    full = add_origin_rolling_features(full)
    full = add_dest_rolling_features(full)
    full = add_absorb_score(full)
    full = drop_leaky_target_columns(full)

    # ── 4. Predict eval flights ───────────────────────────────────────────
    print("[4/5] Predicting with v9...")
    meta = load_artifact(PROJECT_ROOT / "artifacts" / "4year_v9")

    eval_rows = full[full["fa_flight_id"].isin(eval_ids)].copy()
    print(f"      Eval rows in feature matrix: {len(eval_rows):,}")

    X = prepare_inference_frame(
        eval_rows, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=None,
    )
    cat_set = set(meta.get("cat_cols", []))
    for c in X.columns:
        if c not in cat_set and X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    probas = predict_proba(meta["booster"], X)

    eval_rows = eval_rows.reset_index(drop=True)
    eval_rows["proba_v9"] = probas

    # Merge back the live original predictions and ground truth
    result = eval_rows[["fa_flight_id", "proba_v9"]].merge(
        eval_set[["fa_flight_id", "proba_delay", "arr_delay_min", "y_true"]],
        on="fa_flight_id", how="inner",
    )
    result.rename(columns={"proba_delay": "proba_live"}, inplace=True)

    # ── 5. Metrics ────────────────────────────────────────────────────────
    print(f"[5/5] Computing metrics on {len(result):,} flights...")
    y = result["y_true"].values

    # Quantile thresholds (match production)
    t_live = float(np.quantile(result["proba_live"], 1 - 0.22))
    t_v9   = float(np.quantile(result["proba_v9"],   1 - 0.22))
    pred_live = (result["proba_live"] >= t_live).astype(int).values
    pred_v9   = (result["proba_v9"]   >= t_v9).astype(int).values

    auc_live   = roc_auc_score(y, result["proba_live"])
    auc_v9     = roc_auc_score(y, result["proba_v9"])
    brier_live = brier_score_loss(y, result["proba_live"])
    brier_v9   = brier_score_loss(y, result["proba_v9"])

    print("\n" + "=" * 80)
    print("  v9 EVALUATION ON ALL live_data.db  (May 4 – May 18, 2026)")
    print("=" * 80)
    print(f"Flights evaluated:  n = {len(result):,}")
    print(f"Actual delay rate:  {y.mean()*100:.1f}%")
    print("-" * 80)
    print(f"Original Live Predictions (mixed v7/v9 as deployed):")
    print(f"  AUC   = {auc_live:.4f}")
    print(f"  Brier = {brier_live:.4f}")
    print(f"  F1    = {f1_score(y, pred_live):.4f}  "
          f"(P={precision_score(y, pred_live):.3f}, R={recall_score(y, pred_live):.3f})")
    cm_live = confusion_matrix(y, pred_live, labels=[0,1])
    print(f"  Confusion: TN={cm_live[0,0]} FP={cm_live[0,1]} FN={cm_live[1,0]} TP={cm_live[1,1]}")
    print("-" * 80)
    print(f"Model v9 (Batch Re-prediction, leak-safe lineage):")
    print(f"  AUC   = {auc_v9:.4f}")
    print(f"  Brier = {brier_v9:.4f}")
    print(f"  F1    = {f1_score(y, pred_v9):.4f}  "
          f"(P={precision_score(y, pred_v9):.3f}, R={recall_score(y, pred_v9):.3f})")
    cm_v9 = confusion_matrix(y, pred_v9, labels=[0,1])
    print(f"  Confusion: TN={cm_v9[0,0]} FP={cm_v9[0,1]} FN={cm_v9[1,0]} TP={cm_v9[1,1]}")
    print("-" * 80)
    print(f"Δ (v9_retest − live_original):")
    print(f"  ΔAUC   = {auc_v9 - auc_live:+.4f}")
    print(f"  ΔBrier = {brier_v9 - brier_live:+.4f}  (negative = better)")
    print("=" * 80)

    # Per-day breakdown
    result["fl_date"] = result["fa_flight_id"].map(
        dict(zip(flights_df["fa_flight_id"], flights_df["fl_date"]))
    )
    print("\n--- Per-day AUC (v9) ---")
    for dt, g in sorted(result.groupby("fl_date")):
        if len(g) < 20 or g["y_true"].nunique() < 2:
            continue
        print(f"  {dt}  n={len(g):5d}  AUC={roc_auc_score(g['y_true'], g['proba_v9']):.4f}  "
              f"delay={g['y_true'].mean()*100:.1f}%")

    # Save
    out = PROJECT_ROOT / "logs" / "eval_v9_all_livedb.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "n": len(result),
        "delay_rate": float(y.mean()),
        "auc_v9": float(auc_v9), "brier_v9": float(brier_v9),
        "auc_live": float(auc_live), "brier_live": float(brier_live),
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
