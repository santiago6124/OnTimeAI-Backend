"""OnTimeAI — FastAPI backend.

Serves live predictions from live_data.db.  Swap models without restart:
    ACTIVE_MODEL=4year_v9_recal uvicorn api:app --reload

Endpoints:
    GET /flights            — today's scheduled flights + latest prediction
    GET /flights/{id}       — single flight detail + SHAP
    GET /metrics/summary    — today's KPI cards
    GET /metrics/hourly     — predictions grouped by departure hour
    GET /metrics/model      — active model info + live AUC
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Config ─────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "live_data.db"

MODEL_REGISTRY: dict[str, Path] = {
    "4year_v9":       Path(__file__).parent / "artifacts/4year_v9",
    "4year_v9_recal": Path(__file__).parent / "artifacts/4year_v9_recal",
    "4year_v7_recal": Path(__file__).parent / "artifacts/4year_v7_recal",
}
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "4year_v9")
ARTIFACT_PATH = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["4year_v9"])

FEATURE_LABELS: dict[str, str] = {
    "prev_arr_delay_tail":      "Demora previa del avión",
    "prev_turnaround_tail_min": "Tiempo en tierra previo",
    "tail_flights_today_prior": "Vuelos previos del avión hoy",
    "carrier_delay_rate_yday":  "Tasa demora aerolínea ayer",
    "origin_delay_rate_yday":   "Tasa demora origen ayer",
    "origin_delay_rate_1h":     "Tasa demora origen (1h)",
    "origin_delay_rate_6h":     "Tasa demora origen (6h)",
    "origin_delay_rate_24h":    "Tasa demora origen (24h)",
    "dest_delay_rate_1h":       "Tasa demora destino (1h)",
    "dest_delay_rate_6h":       "Tasa demora destino (6h)",
    "carrier_delay_rate_24h":   "Tasa demora aerolínea (24h)",
    "carrier_delay_rate_7d":    "Tasa demora aerolínea (7d)",
    "DISTANCE":                 "Distancia de vuelo",
    "CRS_ELAPSED_TIME":         "Duración programada",
    "AIRCRAFT_FAMILY":          "Tipo de aeronave",
    "PAGERANK_ORIGIN":          "Importancia origen (red)",
    "PAGERANK_DEST":            "Importancia destino (red)",
    "TAIL_DELAY_DECAY":         "Historial demoras avión",
    "tmpf_origin":              "Temperatura origen",
    "sknt_origin":              "Viento origen",
    "vsby_origin":              "Visibilidad origen",
    "alti_origin":              "Presión origen",
    "CRS_DEP_MIN_sin":          "Hora salida (cíclica)",
    "CRS_DEP_MIN_cos":          "Hora salida (cíclica)",
    "absorb_score":             "Capacidad absorción ATL",
    "congestion_score":         "Congestión aeropuerto",
}

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="OnTimeAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def risk_level(proba: float) -> str:
    if proba >= 0.35:
        return "high"
    if proba >= 0.15:
        return "medium"
    return "low"


# ── Lazy model loader ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    from ontimeai.model import load_artifact
    return load_artifact(ARTIFACT_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────

def _latest_predictions_today(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Latest prediction per flight for flights scheduled today."""
    today = today_utc()
    rows = con.execute("""
        SELECT f.fa_flight_id,
               f.ident_iata,
               f.op_carrier,
               f.flight_number,
               f.origin,
               f.dest,
               f.scheduled_out_utc,
               f.scheduled_in_utc,
               f.aircraft_type,
               p.proba_delay,
               p.predicted_delay,
               p.predicted_at_utc,
               CASE WHEN a.fa_flight_id IS NOT NULL THEN 1 ELSE 0 END AS has_actual
        FROM flights f
        JOIN (
            SELECT fa_flight_id,
                   proba_delay,
                   predicted_delay,
                   predicted_at_utc,
                   ROW_NUMBER() OVER (
                       PARTITION BY fa_flight_id
                       ORDER BY predicted_at_utc DESC
                   ) AS rn
            FROM predictions
        ) p ON p.fa_flight_id = f.fa_flight_id AND p.rn = 1
        LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
                            AND a.arr_delay_min IS NOT NULL
        WHERE f.scheduled_out_utc LIKE ? || '%'
          AND f.cancelled = 0
        ORDER BY f.scheduled_out_utc
    """, (today,)).fetchall()
    return rows


def _flight_row_to_dict(row: sqlite3.Row) -> dict:
    proba = float(row["proba_delay"])
    ident = row["ident_iata"] or row["flight_number"] or row["fa_flight_id"]
    return {
        "fa_flight_id":    row["fa_flight_id"],
        "flight_number":   ident,
        "airline_code":    row["op_carrier"] or "",
        "origin":          row["origin"] or "",
        "destination":     row["dest"] or "",
        "scheduled_out_utc": row["scheduled_out_utc"] or "",
        "scheduled_in_utc":  row["scheduled_in_utc"] or "",
        "aircraft_type":   row["aircraft_type"] or "",
        "risk":            risk_level(proba),
        "delay_probability": round(proba, 4),
        "predicted_delay": int(row["predicted_delay"]),
        "predicted_at_utc": row["predicted_at_utc"] or "",
        "has_actual":      bool(row["has_actual"]),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/flights")
def list_flights():
    con = get_db()
    try:
        rows = _latest_predictions_today(con)
        return [_flight_row_to_dict(r) for r in rows]
    finally:
        con.close()


@app.get("/flights/{fa_flight_id:path}")
def get_flight(fa_flight_id: str):
    con = get_db()
    try:
        rows = _latest_predictions_today(con)
        row = next((r for r in rows if r["fa_flight_id"] == fa_flight_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="Flight not found")

        result = _flight_row_to_dict(row)
        result["shap"] = _compute_shap(fa_flight_id)
        return result
    finally:
        con.close()


def _compute_shap(fa_flight_id: str) -> list[dict]:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ontimeai.live import open_db, build_inference_frame
        from ontimeai.lineage_fallback import load_lookups
        from ontimeai.explainability import compute_shap_values, explain_instance
        from predict import prepare_inference_frame

        meta = _load_meta()
        conn = open_db()

        df = build_inference_frame(conn, [fa_flight_id], history_days=7)
        conn.close()
        if df.empty:
            return []

        fallback_path = ARTIFACT_PATH.parent / "lineage_fallback.joblib"
        fallback = load_lookups(fallback_path) if fallback_path.exists() else None

        X = prepare_inference_frame(
            df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback
        )
        cat_cols_set = set(meta.get("cat_cols", []))
        for c in X.columns:
            if c not in cat_cols_set and X[c].dtype == object:
                import pandas as pd
                X[c] = pd.to_numeric(X[c], errors="coerce")

        target_mask = df["fa_flight_id"].isin([fa_flight_id]) & df["ARR_DELAY"].isna()
        if not target_mask.any():
            target_mask = df["fa_flight_id"].isin([fa_flight_id])

        idx = int(df.index[target_mask][0])
        row_pos = list(df.index).index(idx)

        shap_vals = compute_shap_values(meta["booster"], X)
        top = explain_instance(shap_vals, list(X.columns), row_pos, top_n=10)

        result = []
        for _, r in top.iterrows():
            feat = str(r["feature"])
            contrib = float(r["contribution"])
            result.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "contribution": round(abs(contrib), 4),
                "direction":    "positive" if contrib >= 0 else "negative",
            })
        return result
    except Exception:
        return []


@app.get("/metrics/summary")
def metrics_summary():
    con = get_db()
    try:
        rows = _latest_predictions_today(con)
        if not rows:
            return {
                "total_flights": 0, "high_risk": 0, "medium_risk": 0,
                "low_risk": 0, "avg_delay_probability": 0.0,
                "predicted_positive_rate": 0.0,
                "model_version": ACTIVE_MODEL, "last_tick_utc": "",
            }
        probas = [float(r["proba_delay"]) for r in rows]
        preds  = [int(r["predicted_delay"]) for r in rows]
        ticks  = [r["predicted_at_utc"] for r in rows if r["predicted_at_utc"]]
        return {
            "total_flights":           len(rows),
            "high_risk":               sum(1 for p in probas if p >= 0.35),
            "medium_risk":             sum(1 for p in probas if 0.15 <= p < 0.35),
            "low_risk":                sum(1 for p in probas if p < 0.15),
            "avg_delay_probability":   round(float(np.mean(probas)), 4),
            "predicted_positive_rate": round(float(np.mean(preds)), 4),
            "model_version":           ACTIVE_MODEL,
            "last_tick_utc":           max(ticks) if ticks else "",
        }
    finally:
        con.close()


@app.get("/metrics/hourly")
def metrics_hourly():
    con = get_db()
    try:
        rows = _latest_predictions_today(con)
        buckets: dict[str, dict] = {}
        for r in rows:
            sched = r["scheduled_out_utc"] or ""
            if len(sched) >= 16:
                hour = sched[11:13] + ":00"
            else:
                hour = "??"
            if hour not in buckets:
                buckets[hour] = {"hour": hour, "total": 0, "high_risk": 0, "sum_proba": 0.0}
            b = buckets[hour]
            b["total"] += 1
            b["sum_proba"] += float(r["proba_delay"])
            if float(r["proba_delay"]) >= 0.35:
                b["high_risk"] += 1

        result = []
        for hour in sorted(buckets):
            b = buckets[hour]
            result.append({
                "hour":      hour,
                "total":     b["total"],
                "high_risk": b["high_risk"],
                "avg_proba": round(b["sum_proba"] / b["total"], 4) if b["total"] else 0.0,
            })
        return result
    finally:
        con.close()


@app.get("/metrics/model")
def metrics_model():
    try:
        meta = _load_meta()
        threshold = float(meta.get("threshold", 0.0))
    except Exception:
        threshold = 0.0

    # Live AUC/Brier from last 7 days (deduplicated per flight)
    live_auc = live_brier = n_actuals = None
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        con = get_db()
        import pandas as pd
        df = pd.read_sql("""
            SELECT p.fa_flight_id, p.proba_delay, p.predicted_at_utc,
                   CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
            FROM predictions p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0
              AND p.predicted_at_utc >= datetime('now', '-7 days')
            ORDER BY p.predicted_at_utc
        """, con)
        con.close()
        if not df.empty:
            df = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()
            y = df["delayed"].to_numpy(dtype=int)
            p = df["proba_delay"].to_numpy(dtype=float)
            if len(y) >= 30 and y.sum() >= 5:
                live_auc   = round(float(roc_auc_score(y, p)), 4)
                live_brier = round(float(brier_score_loss(y, p)), 4)
                n_actuals  = int(len(y))
    except Exception:
        pass

    version = ACTIVE_MODEL.replace("4year_", "").replace("_", "-")
    return {
        "active_model": ACTIVE_MODEL,
        "version":      version,
        "live_auc":     live_auc,
        "live_brier":   live_brier,
        "n_actuals":    n_actuals,
        "threshold":    threshold,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
