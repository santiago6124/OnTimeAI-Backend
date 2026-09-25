"""
No volver a pedir el clima que ya tenemos fresco.

IEM publica METAR una vez por hora y el ciclo corre cada 15 minutos: tres de
cada cuatro pedidos traian el mismo dato. Eso era gratis mientras eran ~6
aeropuertos.

Al empezar a descubrir el horario futuro (Scrapper #11) pasaron a ser 135, con
46 pedidos de red. IEM limita por tasa y cada pedido limitado espera 5 segundos,
asi que el ciclo salto de 4,8 a 13,5 minutos contra un scheduler de 15 — a
minuto y medio de que dos jobs se pisaran.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from live_pull import WEATHER_FRESH_MINUTES, _airports_with_fresh_weather


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE weather_obs (station TEXT, valid_utc TEXT)")
    return c


def _obs(con, station, hace_minutos):
    cuando = datetime.now(timezone.utc) - timedelta(minutes=hace_minutos)
    # Con `T`, que es como lo guarda produccion: `2026-09-25T14:56:00`. El test
    # anterior usaba espacio y por eso no vio el bug que rompio el clima.
    con.execute("INSERT INTO weather_obs VALUES (?,?)",
                (station, cuando.strftime("%Y-%m-%dT%H:%M:%S")))
    con.commit()


def test_una_observacion_reciente_cuenta_como_fresca(con) -> None:
    _obs(con, "ATL", 10)
    assert _airports_with_fresh_weather(con, 50) == {"ATL"}


def test_una_observacion_vieja_no(con) -> None:
    _obs(con, "ATL", 90)
    assert _airports_with_fresh_weather(con, 50) == set()


def test_vale_la_mas_reciente_de_cada_estacion(con) -> None:
    """Una estacion con historia vieja y una observacion nueva esta fresca."""
    _obs(con, "ATL", 600)
    _obs(con, "ATL", 5)
    assert _airports_with_fresh_weather(con, 50) == {"ATL"}


def test_separa_estacion_por_estacion(con) -> None:
    _obs(con, "ATL", 5)
    _obs(con, "MIA", 200)
    assert _airports_with_fresh_weather(con, 50) == {"ATL"}


def test_sin_tabla_no_rompe(con) -> None:
    """Base recien creada: se piden todos, que es el comportamiento anterior."""
    c = sqlite3.connect(":memory:")
    assert _airports_with_fresh_weather(c, 50) == set()


def test_las_fechas_se_leen_como_utc(con) -> None:
    """
    `valid_utc` se guarda sin offset. Leerlo como hora local daria tres horas de
    mas en Argentina y ningun aeropuerto figuraria fresco.
    """
    _obs(con, "ATL", 30)
    assert "ATL" in _airports_with_fresh_weather(con, 50)


def test_una_observacion_vieja_con_T_no_pasa_por_fresca(con) -> None:
    """
    El bug que rompio el clima en produccion el 25/09.

    La columna guarda `2026-09-25T14:56:00` y `datetime('now', ...)` devuelve
    `2026-09-25 18:26:04`. Comparadas como texto, la `T` (0x54) le gana al
    espacio (0x20), asi que cualquier observacion —de hace tres horas o de hace
    tres dias— daba "fresca". El clima dejo de pedirse por completo: 138 de 148
    estaciones figuraban frescas cuando la verdad eran 0.
    """
    _obs(con, "ATL", 240)   # cuatro horas: viejisima
    assert _airports_with_fresh_weather(con, 50) == set(), (
        "una observacion de hace 4 h no puede figurar fresca"
    )


def test_el_formato_del_test_es_el_de_produccion(con) -> None:
    """
    Habia un test para el bug de arriba y paso igual, porque insertaba las
    fechas con espacio en vez de con T. Usaba un formato que produccion no usa,
    y por eso no vio nada.
    """
    _obs(con, "ATL", 10)
    guardado = con.execute("SELECT valid_utc FROM weather_obs").fetchone()[0]
    assert "T" in guardado, "produccion guarda con T; el test tiene que imitarlo"


def test_el_umbral_deja_margen_para_el_ciclo_siguiente() -> None:
    """IEM publica cada hora; el umbral tiene que estar por debajo."""
    assert 30 <= WEATHER_FRESH_MINUTES < 60
