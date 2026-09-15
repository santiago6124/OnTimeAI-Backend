"""
La ventana de vuelos activos no puede retener vuelos que ya aterrizaron.

`_latest_predictions_active()` incluye, ademas de la ventana temporal, los
vuelos programados en las ultimas 24 h que todavia no salieron. Esa segunda
condicion se testeaba contra `actual_out_utc`, que poblaba AeroAPI. Desde Fase 4
FR24 es el 96% de la muestra y no expone gate-out: deja el campo nulo siempre,
la condicion se cumplia para todos, y la ventana retenia las 24 h enteras.

Medido en produccion antes del arreglo: 492 de 637 vuelos con salida programada
hace mas de 6 h, y de esos el 99% ya habia despegado y el 98% ya habia
aterrizado con resultado real. Ver issue #51.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import api
from ontimeai.live import SCHEMA


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _ahora(horas: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=horas)).isoformat()


def _vuelo(con, fid, *, sched_out, actual_out=None, actual_off=None,
           actual_on=None, actual_in=None, arr_delay=None):
    con.execute(
        "INSERT INTO flights (fa_flight_id, ident_iata, op_carrier, flight_number,"
        " origin, dest, scheduled_out_utc, scheduled_in_utc, cancelled,"
        " first_seen_utc, last_updated_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (fid, "DL1", "DL", "1", "ATL", "MIA", sched_out, sched_out, 0,
         sched_out, sched_out),
    )
    con.execute(
        "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
        " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
        (fid, fid, sched_out, 0.1, 0),
    )
    if any(v is not None for v in (actual_out, actual_off, actual_on, actual_in, arr_delay)):
        con.execute(
            "INSERT INTO actuals (fa_flight_id, stable_id, actual_out_utc,"
            " actual_off_utc, actual_on_utc, actual_in_utc, arr_delay_min,"
            " cancelled, diverted, settled_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (fid, fid, actual_out, actual_off, actual_on, actual_in, arr_delay,
             0, 0, sched_out),
        )
    con.commit()


def _ids(con) -> set[str]:
    return {r["fa_flight_id"] for r in api._latest_predictions_active(con)}


class TestVuelosViejos:
    def test_uno_que_aterrizo_sale_de_la_ventana(self, con) -> None:
        """
        El caso de #51. Tipico de FR24: sin gate-out, con despegue y
        aterrizaje. Antes del arreglo la ventana lo retenia 24 h.
        """
        _vuelo(con, "ATERRIZO", sched_out=_ahora(-10),
               actual_off=_ahora(-9.5), actual_in=_ahora(-8), arr_delay=12.0)
        assert "ATERRIZO" not in _ids(con)

    def test_uno_que_despego_pero_no_llego_tambien_sale(self, con) -> None:
        # Esta en el aire: ya no es un vuelo "pendiente de salir".
        _vuelo(con, "EN_AIRE", sched_out=_ahora(-10), actual_off=_ahora(-9.5))
        assert "EN_AIRE" not in _ids(con)

    def test_uno_con_gate_out_de_aeroapi_sale(self, con) -> None:
        _vuelo(con, "AEROAPI", sched_out=_ahora(-10),
               actual_out=_ahora(-9.8), actual_off=_ahora(-9.5))
        assert "AEROAPI" not in _ids(con)


class TestVuelosVigentes:
    def test_uno_demorado_en_salir_sigue_apareciendo(self, con) -> None:
        """
        La razon de existir de la clausula: programado hace horas, sin ninguna
        senal de salida. Es justo el que el operador necesita ver.
        """
        _vuelo(con, "DEMORADO", sched_out=_ahora(-10))
        assert "DEMORADO" in _ids(con)

    def test_uno_de_la_ventana_normal_aparece(self, con) -> None:
        _vuelo(con, "PROXIMO", sched_out=_ahora(2))
        assert "PROXIMO" in _ids(con)

    def test_uno_reciente_ya_despegado_aparece_por_la_ventana_temporal(self, con) -> None:
        # Despego hace un rato pero sigue dentro de las -6 h: se muestra por la
        # primera condicion, no por la de "pendiente".
        _vuelo(con, "RECIENTE", sched_out=_ahora(-2), actual_off=_ahora(-1.8))
        assert "RECIENTE" in _ids(con)


class TestLimite:
    def test_mas_alla_de_24h_no_vuelve_aunque_no_tenga_senal(self, con) -> None:
        _vuelo(con, "ANTIGUO", sched_out=_ahora(-30))
        assert "ANTIGUO" not in _ids(con)
