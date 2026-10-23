"""ontimeai/live.py: cliente AeroAPI, paginación, holding ADS-B y features v8 (CP-COB-05).

Cuatro tramos que no tenía ningún test: los reintentos de `_api_get` ante
429 y su fallo cerrado ante 4xx/5xx, la paginación por cursor de
`fetch_airport_flights`, la heurística de holding de `compute_adsb_holding_min`
(cada condición de descarte y el caso positivo) y la degradación de
`_add_v8_features` cuando faltan el artefacto de PageRank o el cliente GDP.
Todo con dobles: sin red, sin archivos.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ontimeai import live


class RespuestaFalsa:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


@pytest.fixture
def sin_esperas(monkeypatch):
    esperas: list[float] = []
    monkeypatch.setattr(live.time, "sleep", lambda s: esperas.append(s))
    return esperas


# ── _api_get ────────────────────────────────────────────────────────────────

def test_api_get_devuelve_el_json_y_manda_la_clave(monkeypatch, sin_esperas):
    # Arrange
    llamadas = []
    monkeypatch.setattr(live.requests, "get",
                        lambda url, headers, params, timeout: llamadas.append((url, headers, params)) or RespuestaFalsa(200, {"ok": 1}))
    # Act
    out = live._api_get("/airports/KATL/flights/arrivals", {"start": "a"}, key="clave-de-prueba")
    # Assert
    assert out == {"ok": 1}
    url, headers, params = llamadas[0]
    assert url.endswith("/airports/KATL/flights/arrivals") and headers["x-apikey"] == "clave-de-prueba"
    assert params == {"start": "a"} and sin_esperas == []


def test_api_get_reintenta_ante_429_con_backoff(monkeypatch, sin_esperas):
    respuestas = iter([RespuestaFalsa(429), RespuestaFalsa(200, {"ok": 2})])
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: next(respuestas))
    assert live._api_get("/x", key="k", backoff_s=15.0) == {"ok": 2}
    assert sin_esperas == [15.0]


def test_api_get_falla_cerrado_ante_error_http(monkeypatch, sin_esperas):
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: RespuestaFalsa(503, text="unavailable"))
    with pytest.raises(RuntimeError, match="AeroAPI 503"):
        live._api_get("/x", key="k")


def test_api_get_agota_los_reintentos(monkeypatch, sin_esperas):
    monkeypatch.setattr(live.requests, "get", lambda *a, **k: RespuestaFalsa(429))
    with pytest.raises(RuntimeError, match="exhausted"):
        live._api_get("/x", key="k", retries=2, backoff_s=1.0)
    assert sin_esperas == [1.0, 2.0]  # backoff exponencial


# ── fetch_airport_flights ───────────────────────────────────────────────────

def _paginas(monkeypatch, paginas: dict[str | None, dict]):
    pedidos = []

    def falso(path, params, key):
        pedidos.append(dict(params))
        return paginas[params.get("cursor")]

    monkeypatch.setattr(live, "_api_get", falso)
    return pedidos


def test_fetch_airport_flights_sigue_el_cursor_hasta_el_final(monkeypatch, sin_esperas):
    pedidos = _paginas(monkeypatch, {
        None: {"arrivals": [{"id": 1}], "links": {"next": "/airports/KATL/flights/arrivals?cursor=abc"}},
        "abc": {"arrivals": [{"id": 2}], "links": None},
    })
    out = live.fetch_airport_flights("KATL", "arrivals", "s", "e", key="k")
    assert [f["id"] for f in out] == [1, 2]
    assert pedidos[1]["cursor"] == "abc" and sin_esperas == [6.0]


def test_fetch_airport_flights_respeta_max_pages(monkeypatch, sin_esperas):
    _paginas(monkeypatch, {
        None: {"departures": [{"id": 1}], "links": {"next": "/x?cursor=p2"}},
        "p2": {"departures": [{"id": 2}], "links": {"next": "/x?cursor=p3"}},
        "p3": {"departures": [{"id": 3}], "links": None},
    })
    out = live.fetch_airport_flights("KATL", "departures", "s", "e", max_pages=2, key="k")
    assert [f["id"] for f in out] == [1, 2]


def test_fetch_airport_flights_corta_si_el_next_no_trae_cursor(monkeypatch, sin_esperas):
    _paginas(monkeypatch, {None: {"arrivals": [{"id": 1}], "links": {"next": "/x?otro=1"}}})
    assert len(live.fetch_airport_flights("KATL", "arrivals", "s", "e", key="k")) == 1


def test_fetch_airport_flights_rechaza_un_tipo_desconocido():
    with pytest.raises(AssertionError):
        live.fetch_airport_flights("KATL", "llegadas", "s", "e", key="k")


# ── compute_adsb_holding_min ────────────────────────────────────────────────

ATL_LAT, ATL_LON = 33.6367, -84.4281


def _posiciones(puntos: list[tuple[float, float, int, int]]):
    """(lat, lon, on_ground, minutos_atras) → tabla aircraft_position en memoria."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE aircraft_position(lat REAL, lon REAL, on_ground INTEGER, registration TEXT, captured_at_utc TEXT)")
    ahora = datetime.now(timezone.utc)
    for lat, lon, en_tierra, hace in puntos:
        conn.execute("INSERT INTO aircraft_position VALUES (?,?,?,?,?)",
                     (lat, lon, en_tierra, "N828NW", (ahora - timedelta(minutes=hace)).isoformat()))
    return conn


# Órbita a ~30 nm al norte de ATL (0,5° de latitud ≈ 30 nm), 10 minutos, poco progreso neto.
ORBITA = [(ATL_LAT + 0.50, ATL_LON - 0.05, 0, 10), (ATL_LAT + 0.52, ATL_LON + 0.05, 0, 7),
          (ATL_LAT + 0.49, ATL_LON + 0.08, 0, 4), (ATL_LAT + 0.51, ATL_LON - 0.02, 0, 0)]


def test_holding_detectado_devuelve_los_minutos_observados():
    conn = _posiciones(ORBITA)
    minutos = live.compute_adsb_holding_min(conn, "n828nw", dest="ATL")
    assert minutos is not None and 9.5 <= minutos <= 10.5


@pytest.mark.parametrize("tail, dest", [(None, "ATL"), ("", "ATL"), ("N828NW", "MIA"), ("N828NW", None)])
def test_holding_sin_matricula_o_fuera_de_atl_es_none(tail, dest):
    assert live.compute_adsb_holding_min(_posiciones(ORBITA), tail, dest=dest) is None


def test_holding_sin_tabla_es_none():
    conn = sqlite3.connect(":memory:")
    assert live.compute_adsb_holding_min(conn, "N828NW", dest="ATL") is None


def test_holding_con_pocas_observaciones_es_none():
    assert live.compute_adsb_holding_min(_posiciones(ORBITA[:3]), "N828NW", dest="ATL") is None


def test_holding_con_una_observacion_en_tierra_es_none():
    puntos = ORBITA[:3] + [(ATL_LAT, ATL_LON, 1, 0)]
    assert live.compute_adsb_holding_min(_posiciones(puntos), "N828NW", dest="ATL") is None


def test_holding_lejos_del_aeropuerto_es_crucero_no_espera():
    lejos = [(lat + 2.0, lon, g, h) for lat, lon, g, h in ORBITA]  # ~150 nm
    assert live.compute_adsb_holding_min(_posiciones(lejos), "N828NW", dest="ATL") is None


def test_holding_con_mucho_progreso_neto_es_aproximacion():
    aproximacion = [(ATL_LAT + 0.9, ATL_LON, 0, 10), (ATL_LAT + 0.7, ATL_LON, 0, 7),
                    (ATL_LAT + 0.5, ATL_LON, 0, 4), (ATL_LAT + 0.3, ATL_LON, 0, 0)]
    assert live.compute_adsb_holding_min(_posiciones(aproximacion), "N828NW", dest="ATL") is None


def test_holding_con_mucha_dispersion_es_transito():
    transito = [(ATL_LAT + 0.9, ATL_LON, 0, 10), (ATL_LAT + 0.3, ATL_LON + 0.5, 0, 7),
                (ATL_LAT + 0.9, ATL_LON - 0.5, 0, 4), (ATL_LAT + 0.9, ATL_LON, 0, 0)]
    assert live.compute_adsb_holding_min(_posiciones(transito), "N828NW", dest="ATL") is None


def test_holding_demasiado_breve_es_none():
    breve = [(lat, lon, g, h * 0.3) for lat, lon, g, h in ORBITA]  # 3 minutos de ventana
    assert live.compute_adsb_holding_min(_posiciones(breve), "N828NW", dest="ATL") is None


# ── _add_v8_features ────────────────────────────────────────────────────────

def test_v8_sin_artefacto_ni_cliente_gdp_rellena_con_neutros(monkeypatch):
    # Arrange: el módulo de PageRank no importa y el cliente GDP falla
    monkeypatch.setitem(sys.modules, "feature_engineering_v7.v8_features", None)

    class GdpRoto:
        def add_gdp_features(self, df):
            raise RuntimeError("FAA caída")

    monkeypatch.setattr(live, "_GDP_CLIENT", GdpRoto())
    df = pd.DataFrame({"ORIGIN": ["ATL"], "DEST": ["MIA"]})
    # Act
    with pytest.warns(UserWarning):
        out = live._add_v8_features(df)
    # Assert: la inferencia nunca se bloquea
    assert out["ORIGIN_PAGERANK"].tolist() == [0.0] and out["DEST_PAGERANK"].tolist() == [0.0]
    assert out["GDP_FLAG"].tolist() == [0] and out["GDP_DELAY_MIN"].tolist() == [0.0]


def test_v8_usa_el_cliente_gdp_cuando_responde(monkeypatch):
    monkeypatch.setitem(sys.modules, "feature_engineering_v7.v8_features", None)

    class GdpOk:
        def add_gdp_features(self, df):
            return df.assign(GDP_FLAG=1, GDP_DELAY_MIN=45.0)

    monkeypatch.setattr(live, "_GDP_CLIENT", GdpOk())
    with pytest.warns(UserWarning):
        out = live._add_v8_features(pd.DataFrame({"ORIGIN": ["ATL"], "DEST": ["MIA"]}))
    assert out["GDP_FLAG"].tolist() == [1] and out["GDP_DELAY_MIN"].tolist() == [45.0]
