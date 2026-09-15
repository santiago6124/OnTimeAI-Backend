"""
La cache de SHAP se busca por `fa_flight_id`, y eso la vuelve fragil a los
reapuntados de id.

El harvester reconcilia los placeholders `SYN-` de captura de legs futuros
contra el vuelo real. Hasta el 15/09 repuntaba `predictions` pero no
`prediction_shap`, asi que la prediccion se mudaba al id real y su explicacion
se quedaba en el id muerto. La consulta no la encontraba y el detalle del vuelo
salia sin factores.

Medido antes del arreglo: el 95,4% de `prediction_shap` tenia un id `SYN-`, y el
98% de las predicciones con vuelo real no tenian ninguna fila. Ver issue #5.
"""
from __future__ import annotations

import sqlite3

import pytest

import api


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute(
        """CREATE TABLE prediction_shap (
             fa_flight_id TEXT NOT NULL,
             predicted_at_utc TEXT NOT NULL,
             feature_name TEXT NOT NULL,
             shap_value REAL NOT NULL,
             feature_value TEXT,
             rank INTEGER NOT NULL,
             PRIMARY KEY (fa_flight_id, predicted_at_utc, feature_name)
           )"""
    )
    return c


def _shap(con, fid, at, feature, valor, rank):
    con.execute(
        "INSERT INTO prediction_shap VALUES (?,?,?,?,?,?)",
        (fid, at, feature, valor, "12", rank),
    )


def test_devuelve_los_factores_ordenados_por_rank(con) -> None:
    at = "2026-09-15T01:00:00+00:00"
    _shap(con, "REAL1", at, "DEP_HOUR", 0.30, 1)
    _shap(con, "REAL1", at, "CARRIER", -0.20, 2)
    con.commit()

    out = api._load_cached_shap(con, "REAL1")
    assert [f["feature"] for f in out] == ["DEP_HOUR", "CARRIER"]
    assert out[0]["direction"] == "positive"
    assert out[1]["direction"] == "negative"
    assert out[1]["contribution"] == pytest.approx(0.20), "la contribucion es el valor absoluto"


def test_toma_solo_el_ciclo_mas_reciente(con) -> None:
    _shap(con, "REAL1", "2026-09-15T01:00:00+00:00", "DEP_HOUR", 0.10, 1)
    _shap(con, "REAL1", "2026-09-15T02:00:00+00:00", "WEATHER", 0.50, 1)
    con.commit()

    out = api._load_cached_shap(con, "REAL1")
    assert [f["feature"] for f in out] == ["WEATHER"]


def test_un_shap_bajo_otro_id_no_se_encuentra(con) -> None:
    """
    La forma exacta del bug de #5: la explicacion existe, pero quedo guardada
    contra el id del placeholder mientras su prediccion se mudaba al id real.

    Lo arregla el harvester al reconciliar, no esta consulta: buscar tambien por
    `stable_id` taparia el sintoma y dejaria filas duplicadas acumulandose.
    """
    at = "2026-09-15T01:00:00+00:00"
    _shap(con, "SYN-DL456-ATL-JFK-2026-09-15", at, "DEP_HOUR", 0.30, 1)
    con.commit()

    assert api._load_cached_shap(con, "REAL1") == []


def test_sin_la_tabla_devuelve_vacio_en_vez_de_romper(con) -> None:
    con.execute("DROP TABLE prediction_shap")
    con.commit()
    assert api._load_cached_shap(con, "REAL1") == []


def test_sin_la_tabla_deja_rastro_en_el_log(con, capsys) -> None:
    """Tragarse el fallo en silencio es lo que hizo dificil diagnosticar #5."""
    con.execute("DROP TABLE prediction_shap")
    con.commit()
    api._load_cached_shap(con, "REAL1")
    assert "REAL1" in capsys.readouterr().out
