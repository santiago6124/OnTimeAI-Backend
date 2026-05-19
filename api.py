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
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local", override=False)
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import secrets
from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ── Config ─────────────────────────────────────────────────────────────────

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
_TMP_DB = Path("/tmp/live_data.db")
_BUNDLED_DB = Path(__file__).parent / "live_data.db"
DB_PATH = _TMP_DB if GCS_BUCKET else _BUNDLED_DB
_DB_REFRESH_INTERVAL = 1800  # refresh from GCS every 30 min
_db_last_refresh: float = 0.0


def _refresh_db_from_gcs() -> None:
    global _db_last_refresh
    import time
    if not GCS_BUCKET:
        return
    if time.time() - _db_last_refresh < _DB_REFRESH_INTERVAL:
        return
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        blob = client.bucket(GCS_BUCKET).blob("live_data.db")
        blob.download_to_filename(str(_TMP_DB))
        _db_last_refresh = time.time()
    except Exception as e:
        print(f"[db_refresh] failed: {e}")

MODEL_REGISTRY: dict[str, Path] = {
    "4year_v9":       Path(__file__).parent / "artifacts/4year_v9",
    "4year_v9_recal": Path(__file__).parent / "artifacts/4year_v9_recal",
    "4year_v7_recal": Path(__file__).parent / "artifacts/4year_v7_recal",
}
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "4year_v9")
ARTIFACT_PATH = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["4year_v9"])

# ── Auth ───────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "ontimeai-dev-secret-change-in-prod-32chars")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

_API_USER = os.getenv("API_USERNAME", "admin")
_API_PASS = os.getenv("API_PASSWORD", "ontimeai2026")


def _verify_password(plain: str) -> bool:
    return secrets.compare_digest(plain.encode(), _API_PASS.encode())

_PUBLIC_PATHS = {"/auth/login", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith("/redoc"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        token = auth[7:]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except JWTError:
            return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)
        return await call_next(request)


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Feature labels ─────────────────────────────────────────────────────────

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if GCS_BUCKET:
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            blob = client.bucket(GCS_BUCKET).blob("live_data.db")
            blob.download_to_filename(str(_TMP_DB))
            mb = _TMP_DB.stat().st_size / 1e6
            print(f"[startup] DB downloaded from gs://{GCS_BUCKET}/live_data.db ({mb:.1f} MB)")
        except Exception as e:
            print(f"[startup] GCS download failed, using bundled DB: {e}")
            import shutil
            shutil.copy(_BUNDLED_DB, _TMP_DB)
    yield


app = FastAPI(title="OnTimeAI API", version="1.0.0", lifespan=lifespan)
# Auth is inner; CORS is outer so it wraps 401 responses too
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> sqlite3.Connection:
    _refresh_db_from_gcs()
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


# ── Auth routes ────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(body: LoginRequest):
    if body.username != _API_USER or not _verify_password(body.password):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    token = jwt.encode(
        {"sub": body.username, "exp": expire},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def auth_me(request: Request):
    auth = request.headers.get("Authorization", "")
    token = auth[7:]
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return {"username": payload.get("sub")}


# ── Protected routes ────────────────────────────────────────────────────────

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


@app.get("/test-cases")
def test_cases():
    """Formal test case results for thesis validation (CP-01, CP-02)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score, brier_score_loss

    con = get_db()
    try:
        cp01_row = con.execute("""
            SELECT p.fa_flight_id, p.proba_delay, a.arr_delay_min,
                   f.ident_iata, f.op_carrier, f.origin, f.dest,
                   f.scheduled_out_utc, f.aircraft_type
            FROM (
                SELECT fa_flight_id, proba_delay,
                       ROW_NUMBER() OVER (
                           PARTITION BY fa_flight_id ORDER BY predicted_at_utc DESC
                       ) AS rn
                FROM predictions
            ) p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            JOIN flights  f ON f.fa_flight_id = p.fa_flight_id
            WHERE p.rn = 1
              AND p.proba_delay >= 0.35
              AND a.arr_delay_min > 15
              AND a.cancelled = 0
              AND f.origin = 'ATL'
            ORDER BY p.proba_delay DESC
            LIMIT 1
        """).fetchone()

        cp01 = None
        if cp01_row:
            shap = _compute_shap(cp01_row["fa_flight_id"])
            ident = cp01_row["ident_iata"] or cp01_row["fa_flight_id"]
            cp01 = {
                "fa_flight_id":      cp01_row["fa_flight_id"],
                "flight_number":     ident,
                "airline_code":      cp01_row["op_carrier"] or "",
                "origin":            cp01_row["origin"] or "",
                "destination":       cp01_row["dest"] or "",
                "scheduled_out_utc": cp01_row["scheduled_out_utc"] or "",
                "predicted_proba":   round(float(cp01_row["proba_delay"]), 4),
                "predicted_risk":    risk_level(float(cp01_row["proba_delay"])),
                "actual_delay_min":  int(cp01_row["arr_delay_min"]),
                "shap":              shap,
                "passed":            True,
            }

        df = pd.read_sql("""
            SELECT p.fa_flight_id, p.proba_delay, p.predicted_at_utc,
                   CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
            FROM predictions p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0
        """, con)

        cp02: dict[str, Any] = {
            "n_actuals": 0, "auc": None, "brier": None,
            "actual_delay_rate": None, "passed": False,
        }
        if not df.empty:
            df = (df.sort_values("predicted_at_utc")
                    .groupby("fa_flight_id", as_index=False)
                    .last())
            y      = df["delayed"].to_numpy(dtype=int)
            p_vals = df["proba_delay"].to_numpy(dtype=float)
            if len(y) >= 30 and y.sum() >= 5:
                auc   = float(roc_auc_score(y, p_vals))
                brier = float(brier_score_loss(y, p_vals))
                cp02 = {
                    "n_actuals":         int(len(y)),
                    "auc":               round(auc, 4),
                    "brier":             round(brier, 4),
                    "actual_delay_rate": round(float(y.mean()), 4),
                    "passed":            auc >= 0.70 and brier <= 0.15,
                }

        return {"cp01": cp01, "cp02": cp02}
    finally:
        con.close()


@app.get("/weather/{airport_code}")
def weather(airport_code: str):
    """Latest METAR observation for an airport from the stored weather_obs table."""
    con = get_db()
    try:
        row = con.execute("""
            SELECT station, valid_utc, tmpc, dwpc, relh, drct, sknt, alti,
                   p01m, vsby, gust, wxcodes,
                   wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag
            FROM weather_obs
            WHERE station = ?
            ORDER BY valid_utc DESC
            LIMIT 1
        """, (airport_code.upper(),)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No weather data for {airport_code}")
        return {
            "airport_code":      row["station"],
            "valid_utc":         row["valid_utc"],
            "temperature_c":     row["tmpc"],
            "dewpoint_c":        row["dwpc"],
            "humidity_pct":      row["relh"],
            "wind_direction":    row["drct"],
            "wind_knots":        row["sknt"],
            "gust_knots":        row["gust"],
            "altimeter_inhg":    row["alti"],
            "precip_mm":         row["p01m"],
            "visibility_miles":  row["vsby"],
            "wx_codes":          row["wxcodes"],
            "precip_flag":       bool(row["wx_precip_flag"]),
            "low_visibility":    bool(row["wx_low_vis_flag"]),
            "strong_wind":       bool(row["wx_strong_wind_flag"]),
        }
    finally:
        con.close()


@app.get("/operations/{airport_code}")
def operations(airport_code: str):
    """Today's operational delay stats for flights departing or arriving at an airport."""
    con = get_db()
    try:
        today = today_utc()
        rows = con.execute("""
            SELECT f.origin, f.dest, p.proba_delay, p.predicted_delay
            FROM flights f
            JOIN (
                SELECT fa_flight_id, proba_delay, predicted_delay,
                       ROW_NUMBER() OVER (
                           PARTITION BY fa_flight_id ORDER BY predicted_at_utc DESC
                       ) AS rn
                FROM predictions
            ) p ON p.fa_flight_id = f.fa_flight_id AND p.rn = 1
            WHERE f.scheduled_out_utc LIKE ? || '%'
              AND f.cancelled = 0
              AND (f.origin = ? OR f.dest = ?)
        """, (today, airport_code.upper(), airport_code.upper())).fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No flight data for {airport_code} today")

        probas = [float(r["proba_delay"]) for r in rows]
        departures = [r for r in rows if r["origin"] == airport_code.upper()]
        high_risk  = sum(1 for p in probas if p >= 0.35)
        total      = len(rows)

        return {
            "airport_code":          airport_code.upper(),
            "date_utc":              today,
            "total_flights":         total,
            "departures":            len(departures),
            "arrivals":              total - len(departures),
            "high_risk_count":       high_risk,
            "delay_rate":            round(sum(r["predicted_delay"] for r in rows) / total, 4),
            "avg_delay_probability": round(float(np.mean(probas)), 4),
            "congestion_level":      "high" if total > 80 else "medium" if total > 40 else "low",
        }
    finally:
        con.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
