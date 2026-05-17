"""API endpoint tests using FastAPI TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ── /flights ───────────────────────────────────────────────────────────────

def test_flights_returns_list():
    r = client.get("/flights")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_flights_schema():
    r = client.get("/flights")
    data = r.json()
    if not data:
        pytest.skip("No flights in DB today")
    flight = data[0]
    for key in ("fa_flight_id", "flight_number", "origin", "destination",
                "delay_probability", "risk", "predicted_delay"):
        assert key in flight, f"missing key: {key}"


def test_flights_risk_values():
    r = client.get("/flights")
    for f in r.json():
        assert f["risk"] in ("low", "medium", "high")
        assert 0.0 <= f["delay_probability"] <= 1.0


# ── /flights/{id} ──────────────────────────────────────────────────────────

def test_flight_detail_404():
    r = client.get("/flights/NONEXISTENT-ID-XYZ")
    assert r.status_code == 404


def test_flight_detail_has_shap():
    flights = client.get("/flights").json()
    if not flights:
        pytest.skip("No flights in DB today")
    fid = flights[0]["fa_flight_id"]
    r = client.get(f"/flights/{fid}")
    assert r.status_code == 200
    assert "shap" in r.json()
    assert isinstance(r.json()["shap"], list)


# ── /metrics/summary ───────────────────────────────────────────────────────

def test_metrics_summary_keys():
    r = client.get("/metrics/summary")
    assert r.status_code == 200
    data = r.json()
    for key in ("total_flights", "high_risk", "medium_risk", "low_risk",
                "avg_delay_probability", "model_version"):
        assert key in data


def test_metrics_summary_counts_consistent():
    data = client.get("/metrics/summary").json()
    assert data["high_risk"] + data["medium_risk"] + data["low_risk"] == data["total_flights"]


# ── /metrics/hourly ────────────────────────────────────────────────────────

def test_metrics_hourly_returns_list():
    r = client.get("/metrics/hourly")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_metrics_hourly_schema():
    data = client.get("/metrics/hourly").json()
    for bucket in data:
        assert "hour" in bucket
        assert "total" in bucket
        assert "avg_proba" in bucket
        assert 0.0 <= bucket["avg_proba"] <= 1.0


# ── /metrics/model ─────────────────────────────────────────────────────────

def test_metrics_model_keys():
    r = client.get("/metrics/model")
    assert r.status_code == 200
    data = r.json()
    for key in ("active_model", "version", "threshold"):
        assert key in data


# ── /weather/{airport_code} ────────────────────────────────────────────────

def test_weather_atl():
    r = client.get("/weather/ATL")
    assert r.status_code == 200
    data = r.json()
    for key in ("airport_code", "valid_utc", "temperature_c", "wind_knots", "visibility_miles"):
        assert key in data


def test_weather_404():
    r = client.get("/weather/ZZZZ")
    assert r.status_code == 404


def test_weather_lowercase_normalized():
    r = client.get("/weather/atl")
    assert r.status_code == 200
    assert r.json()["airport_code"] == "ATL"


# ── /operations/{airport_code} ─────────────────────────────────────────────

def test_operations_atl():
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    assert r.status_code == 200
    data = r.json()
    for key in ("airport_code", "total_flights", "departures", "arrivals",
                "delay_rate", "avg_delay_probability", "congestion_level"):
        assert key in data


def test_operations_counts_consistent():
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    data = r.json()
    assert data["departures"] + data["arrivals"] == data["total_flights"]


def test_operations_congestion_values():
    r = client.get("/operations/ATL")
    if r.status_code == 404:
        pytest.skip("No flight data for ATL today")
    assert r.json()["congestion_level"] in ("low", "medium", "high")


def test_operations_404():
    r = client.get("/operations/ZZZZ")
    assert r.status_code == 404
