"""
`departure_delay_min` no significa lo mismo segun quien la escribio.

    aeroapi : demora de puerta        (actual_out - scheduled_out)
    fr24    : despegue - horario de puerta, o sea puerta MAS rodaje

FR24 no expone gate-out. Medido sobre 410.650 filas fr24, la mediana de
`departure_delay_min - arr_delay_min` da +31,2 min contra +7,0 en aeroapi: ese
delta es el rodaje de ATL.

`intermediate_dep_delay_adjust` usa bandas validadas contra BTS para demora de
puerta. Alimentado con la columna de fr24, un vuelo que empujaba en horario y
rodaba media hora caia en la banda 30-60 y saltaba a p=0.90. Ver issue #11.
"""
from __future__ import annotations

import sqlite3

import pytest

from live_pull import load_gate_departure_delays


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute(
        """CREATE TABLE actuals (
             stable_id TEXT PRIMARY KEY,
             source_provider TEXT,
             actual_in_utc TEXT,
             departure_delay_min REAL
           )"""
    )
    return c


def _fila(con, sid, provider, dep_delay, actual_in=None) -> None:
    con.execute(
        "INSERT INTO actuals (stable_id, source_provider, actual_in_utc,"
        " departure_delay_min) VALUES (?,?,?,?)",
        (sid, provider, actual_in, dep_delay),
    )


def test_toma_la_demora_de_puerta_de_aeroapi(con) -> None:
    _fila(con, "A1", "aeroapi", 22.0)
    delays, _ = load_gate_departure_delays(con, ["A1"])
    assert delays == {"A1": 22.0}


def test_descarta_fr24_aunque_tenga_la_columna(con) -> None:
    """El caso de #11: 31 min de rodaje leidos como 31 min de demora."""
    _fila(con, "F1", "fr24", 31.0)
    delays, _ = load_gate_departure_delays(con, ["F1"])
    assert delays == {}, "una demora wheels-off no puede alimentar bandas de puerta"


def test_descarta_las_filas_sin_proveedor(con) -> None:
    # 14.217 filas en produccion tienen source_provider nulo y una mediana
    # intermedia (+17 min): no hay forma de saber que semantica tienen.
    _fila(con, "N1", None, 25.0)
    delays, _ = load_gate_departure_delays(con, ["N1"])
    assert delays == {}


def test_excluye_los_que_ya_aterrizaron(con) -> None:
    """El ajuste es para vuelos en el aire; si ya aterrizo hay resultado real."""
    _fila(con, "A1", "aeroapi", 22.0, actual_in="2026-09-15T01:00:00+00:00")
    delays, aterrizados = load_gate_departure_delays(con, ["A1"])
    assert delays == {}
    assert aterrizados == {"A1"}


def test_los_aterrizados_se_cuentan_sin_importar_el_proveedor(con) -> None:
    # La fase POST_LANDING no depende de la semantica de la demora de salida.
    _fila(con, "F1", "fr24", 31.0, actual_in="2026-09-15T01:00:00+00:00")
    _, aterrizados = load_gate_departure_delays(con, ["F1"])
    assert aterrizados == {"F1"}


def test_mezcla_realista(con) -> None:
    _fila(con, "A1", "aeroapi", 22.0)
    _fila(con, "A2", "aeroapi", None)
    _fila(con, "F1", "fr24", 31.0)
    _fila(con, "F2", "fr24", 8.0, actual_in="2026-09-15T01:00:00+00:00")
    delays, aterrizados = load_gate_departure_delays(con, ["A1", "A2", "F1", "F2"])
    assert delays == {"A1": 22.0}
    assert aterrizados == {"F2"}


def test_sin_ids_no_consulta(con) -> None:
    # Sin el corte temprano, el IN () quedaria vacio y SQLite lo rechaza.
    assert load_gate_departure_delays(con, []) == ({}, set())
