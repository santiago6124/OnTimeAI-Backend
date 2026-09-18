"""
Tests de los endpoints HTTP, contra un TestClient de FastAPI.

Este archivo estuvo excluido del CI porque fallaba al COLECTAR, antes de correr
un solo test. Tres razones, todas a nivel de modulo:

  1. `client.post("/auth/login", ...)` se ejecutaba al importar, y la tabla
     `users` todavia no existia: `TestClient(app)` a secas no corre el
     `lifespan`, que es quien llama a `_init_users_db()`.
  2. Se autenticaba con "admin"/"ontimeai2026", credenciales que se rotaron
     cuando los secretos pasaron a Secret Manager. El login devolvia 401 y los
     tests corrian sin token.
  3. El `lifespan` corre migraciones sobre `DB_PATH`, que sin GCS_BUCKET apunta
     al `live_data.db` versionado: los tests ensuciaban un archivo del repo.

Ahora todo eso vive en el fixture `client`, que usa el TestClient como context
manager —asi corre el lifespan completo— y redirige ambas bases a un temporal.

Era el unico archivo de tests fuera del CI, y justo el de los endpoints. El
costo se vio el 15/09: el PR #40 paso CI con /metrics/summary devolviendo 500
en produccion, porque ningun test ejecutaba esa consulta. Ver issue #20.
"""
from __future__ import annotations

import os
import shutil

import pytest
from fastapi.testclient import TestClient


def _sembrar_vuelo_de_hoy(db) -> None:
    """Un vuelo dentro de la ventana activa, con prediccion, SHAP y resultado.

    El `live_data.db` bundleado es un snapshot fijo: no tiene vuelos de hoy, asi
    que los tests que verifican el esquema de /flights se saltaban con "No
    flights in DB today". Eran justamente los que comprueban que cada campo
    llegue, que es lo que se rompe en una regresion.

    Se siembra en la copia temporal, nunca en el archivo del repo.
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    ahora = datetime.now(timezone.utc)
    sale = (ahora + timedelta(hours=2)).isoformat()
    llega = (ahora + timedelta(hours=4)).isoformat()
    predicho = (ahora - timedelta(minutes=10)).isoformat()
    fid = "TEST-SEED-0001"

    con = sqlite3.connect(db)
    try:
        con.execute(
            """INSERT OR REPLACE INTO flights
               (fa_flight_id, stable_id, ident_iata, op_carrier, flight_number,
                tail_num, origin, dest, fl_date, scheduled_out_utc,
                scheduled_in_utc, aircraft_type, cancelled, diverted,
                first_seen_utc, last_updated_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fid, fid, "DL9999", "DL", "9999", "N999DL", "ATL", "MIA",
             sale[:10], sale, llega, "B738", 0, 0, predicho, predicho),
        )
        con.execute(
            """INSERT OR REPLACE INTO predictions
               (fa_flight_id, stable_id, predicted_at_utc, proba_delay,
                predicted_delay, threshold_used, threshold_strategy,
                prediction_phase)
               VALUES (?,?,?,?,?,?,?,?)""",
            (fid, fid, predicho, 0.42, 1, 0.10, "quantile@0.22", "PRE_DEPARTURE"),
        )
        for rank, (feature, valor) in enumerate(
            (("DEP_HOUR", 0.21), ("ORIG_WX_SKNT", -0.08), ("CARRIER", 0.05)), start=1
        ):
            con.execute(
                """INSERT OR REPLACE INTO prediction_shap
                   (fa_flight_id, predicted_at_utc, feature_name, shap_value,
                    feature_value, rank)
                   VALUES (?,?,?,?,?,?)""",
                (fid, predicho, feature, valor, "12", rank),
            )
        con.commit()
    finally:
        con.close()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient autenticado, con bases temporales.

    De ambito `module` porque levantar el lifespan copia la base (~81 MB) y
    corre migraciones: hacerlo por test multiplicaria eso por veinte sin
    aportar aislamiento, ya que ningun test de este archivo escribe.
    """
    import api

    tmp = tmp_path_factory.mktemp("api")
    originales = (api.DB_PATH, api.USERS_DB_PATH)

    # El lifespan corre migraciones sobre DB_PATH; si apuntara al bundleado,
    # los tests dejarian el archivo del repo modificado.
    db = tmp / "live_data.db"
    shutil.copy(api.DB_PATH, db)
    _sembrar_vuelo_de_hoy(db)
    api.DB_PATH = db
    api.USERS_DB_PATH = tmp / "users.db"

    try:
        # Como context manager para que corra el lifespan, que es quien crea la
        # tabla `users` y da de alta los usuarios semilla desde el entorno.
        with TestClient(api.app) as c:
            credenciales = {
                "username": os.environ["API_USERNAME"],
                "password": os.environ["API_PASSWORD"],
            }
            r = c.post("/auth/login", json=credenciales)
            assert r.status_code == 200, (
                f"login fallo con {r.status_code}: {r.text}. "
                "Las credenciales salen del entorno (ver tests/conftest.py), "
                "no de un valor fijo."
            )
            c.headers.update({"Authorization": f"Bearer {r.json()['access_token']}"})
            yield c
    finally:
        api.DB_PATH, api.USERS_DB_PATH = originales


# ── /flights ───────────────────────────────────────────────────────────────

def test_flights_returns_list(client):
    r = client.get("/flights")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_flights_schema(client):
    r = client.get("/flights")
    data = r.json()
    if not data:
        pytest.skip("No flights in DB today")
    flight = data[0]
    for key in ("fa_flight_id", "flight_number", "origin", "destination",
                "delay_probability", "risk", "predicted_delay",
                "estimated_out_utc", "estimated_in_utc",
                "actual_out_utc", "actual_off_utc",
                "actual_on_utc", "actual_in_utc"):
        assert key in flight, f"missing key: {key}"


def test_flights_risk_values(client):
    r = client.get("/flights")
    for f in r.json():
        assert f["risk"] in ("low", "medium", "high")
        assert 0.0 <= f["delay_probability"] <= 1.0


# ── /flights/{id} ──────────────────────────────────────────────────────────

def test_flight_detail_404(client):
    r = client.get("/flights/NONEXISTENT-ID-XYZ")
    assert r.status_code == 404


def test_flight_detail_has_shap(client):
    flights = client.get("/flights").json()
    if not flights:
        pytest.skip("No flights in DB today")
    fid = flights[0]["fa_flight_id"]
    r = client.get(f"/flights/{fid}")
    assert r.status_code == 200
    assert "shap" in r.json()
    assert isinstance(r.json()["shap"], list)


def test_flight_history_includes_cycle_explanation(client):
    flights = client.get("/flights").json()
    if not flights:
        pytest.skip("No flights in DB today")
    fid = flights[0]["fa_flight_id"]
    r = client.get(f"/flight-history/{fid}")
    assert r.status_code == 200
    history = r.json()
    if not history:
        pytest.skip("No prediction history for current flight")

    cycle = history[-1]
    for key in (
        "predicted_at_utc", "delay_probability", "base_probability",
        "operational_adjustment", "predicted_delay", "threshold_used",
        "threshold_strategy", "prediction_phase", "operational_context", "shap",
    ):
        assert key in cycle, f"missing history key: {key}"
    assert isinstance(cycle["shap"], list)
    if cycle["shap"]:
        for key in ("feature", "label", "contribution", "direction", "value"):
            assert key in cycle["shap"][0]


# ── /metrics/summary ───────────────────────────────────────────────────────

def test_metrics_summary_keys(client):
    r = client.get("/metrics/summary")
    assert r.status_code == 200
    data = r.json()
    for key in ("total_flights", "high_risk", "medium_risk", "low_risk",
                "avg_delay_probability", "model_version"):
        assert key in data


def test_metrics_summary_counts_consistent(client):
    data = client.get("/metrics/summary").json()
    assert data["high_risk"] + data["medium_risk"] + data["low_risk"] == data["total_flights"]


# ── /metrics/hourly ────────────────────────────────────────────────────────

def test_metrics_hourly_returns_list(client):
    r = client.get("/metrics/hourly")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_metrics_hourly_schema(client):
    data = client.get("/metrics/hourly").json()
    for bucket in data:
        assert "hour" in bucket
        assert "total" in bucket
        assert "avg_proba" in bucket
        assert 0.0 <= bucket["avg_proba"] <= 1.0


# ── /metrics/model ─────────────────────────────────────────────────────────

def test_metrics_model_keys(client):
    r = client.get("/metrics/model")
    assert r.status_code == 200
    data = r.json()
    for key in ("active_model", "version", "threshold"):
        assert key in data


# ── /weather/{airport_code} ────────────────────────────────────────────────

def test_weather_atl(client):
    r = client.get("/weather/ATL")
    assert r.status_code == 200
    data = r.json()
    for key in ("airport_code", "valid_utc", "temperature_c", "wind_knots", "visibility_miles"):
        assert key in data


def test_weather_404(client):
    r = client.get("/weather/ZZZZ")
    assert r.status_code == 404


def test_weather_lowercase_normalized(client):
    r = client.get("/weather/atl")
    assert r.status_code == 200
    assert r.json()["airport_code"] == "ATL"


# ── /operations/{airport_code} ─────────────────────────────────────────────

def test_operations_atl(client):
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    assert r.status_code == 200
    data = r.json()
    for key in ("airport_code", "total_flights", "departures", "arrivals",
                "delay_rate", "avg_delay_probability", "congestion_level"):
        assert key in data


def test_operations_counts_consistent(client):
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    data = r.json()
    assert data["departures"] + data["arrivals"] == data["total_flights"]


def test_operations_congestion_values(client):
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    assert r.json()["congestion_level"] in ("low", "medium", "high")


def test_operations_404(client):
    r = client.get("/operations/ZZZZ")
    assert r.status_code == 404


# ── DELETE /users/me ───────────────────────────────────────────────────────

def test_delete_me_removes_the_account(client):
    """Alta, baja y comprobacion de que la sesion vieja ya no sirve.

    Es el unico test del archivo que escribe en users.db, y deja la tabla
    como la encontro: el usuario que crea es el que borra.
    """
    from starlette.testclient import TestClient
    import api

    creado = client.post(
        "/admin/users",
        json={"username": "baja-temporal", "password": "solo-para-este-test"},
    )
    assert creado.status_code == 201, creado.text

    # Cliente aparte para no pisar el Authorization del fixture compartido.
    with TestClient(api.app) as propio:
        login = propio.post(
            "/auth/login",
            json={"username": "baja-temporal", "password": "solo-para-este-test"},
        )
        assert login.status_code == 200, login.text
        propio.headers.update(
            {"Authorization": f"Bearer {login.json()['access_token']}"}
        )
        assert propio.get("/auth/me").status_code == 200

        assert propio.delete("/users/me").status_code == 204

        # El JWT sigue firmado y vigente, pero la cuenta no esta: sesion vencida.
        assert propio.get("/auth/me").status_code == 401
        assert propio.delete("/users/me").status_code == 404

    lista = client.get("/admin/users")
    assert lista.status_code == 200
    assert "baja-temporal" not in {u["username"] for u in lista.json()}
