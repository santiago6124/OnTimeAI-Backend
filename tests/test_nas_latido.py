"""El latido de la FAA: distinguir "no hay programas" de "no contesta".

`nas_status` guarda una fila por aeropuerto bajo programa. Un cielo sin
programas no deja rastro, que para el modelo esta bien —no hay nada que
informar— pero vuelve la tabla inservible para vigilar la fuente.

La madrugada del 27/09 la alarma sono tras 499 minutos sin filas. No habia nada
roto: no hubo un solo programa activo en Estados Unidos en ocho horas. En esa
misma ventana `weather_obs` escribio 1.242 filas, `actuals` 15.463 y
`predictions` 5.300 — el pipeline estaba entero.

`faa_nas_fetch` registra la consulta, no su contenido. Lo delicado es cuando
se escribe: `GdpClient._refresh()` degrada en silencio, asi que el camino
obvio —escribir el latido despues de `_ensure_fresh()`— marcaria como viva una
fuente caida y dejaria al watchdog ciego, que es peor que la falsa alarma.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _con() -> sqlite3.Connection:
    """Una base como la arma produccion.

    Va `open_db` y no `executescript(SCHEMA)` a mano: varias columnas de
    `nas_status` —`end_time_utc`, `max_delay_min`— no estan en el SCHEMA sino
    en las migraciones, asi que una base armada solo con el SCHEMA no es la que
    corre en produccion. De paso esto verifica que `faa_nas_fetch` nazca por el
    camino real de inicializacion.
    """
    from ontimeai.live import open_db

    return open_db(":memory:")


class _ClienteFalso:
    """Un GdpClient de mentira: lo que devuelve y si la consulta funciono."""

    def __init__(self, cache: dict, ok: bool) -> None:
        self._cache = cache
        self.last_refresh_ok = ok

    def _ensure_fresh(self) -> None:  # lo llama snapshot_nas_status
        pass


@pytest.fixture
def parchear(monkeypatch):
    """Reemplaza el GdpClient real para no salir a la red en un test."""

    def _aplicar(cache: dict, ok: bool):
        import feature_engineering_v7.gdp_scraper as scraper

        monkeypatch.setattr(
            scraper, "GdpClient", lambda *a, **k: _ClienteFalso(cache, ok)
        )

    return _aplicar


def test_cielo_despejado_deja_latido(parchear) -> None:
    """Sin programas activos hay latido igual: la FAA contesto."""
    from ontimeai.live import snapshot_nas_status

    parchear({}, ok=True)
    con = _con()

    assert snapshot_nas_status(con) == 0
    latidos = con.execute("SELECT checked_at_utc, airports FROM faa_nas_fetch").fetchall()
    assert len(latidos) == 1
    # Cero aeropuertos es un dato, no una ausencia: es lo que separa esta noche
    # tranquila de una FAA caida.
    assert latidos[0][1] == 0
    assert con.execute("SELECT COUNT(*) FROM nas_status").fetchone()[0] == 0


def test_fuente_caida_no_deja_latido(parchear) -> None:
    """
    El caso que da sentido a todo esto.

    `_refresh()` no lanza cuando falla la red ni cuando falla el parseo: avisa
    con `warnings.warn`, deja el cache como estaba y vuelve. Asi que el `try`
    de `snapshot_nas_status` no se entera, y el cache vacio de una caida es
    identico al de una noche sin programas.

    Si el latido se escribiera por haber llegado hasta aca, el watchdog veria
    la fuente siempre viva y una caida real de la FAA no dispararia nada.
    """
    from ontimeai.live import snapshot_nas_status

    parchear({}, ok=False)
    con = _con()

    assert snapshot_nas_status(con) == 0
    assert con.execute("SELECT COUNT(*) FROM faa_nas_fetch").fetchone()[0] == 0


def test_con_programas_escribe_las_dos_tablas(parchear) -> None:
    from ontimeai.live import snapshot_nas_status

    parchear(
        {
            "LGA": {"type": "Ground Delay", "delay_min": 45.0, "reason": "WEATHER",
                    "end_time_utc": None, "max_delay_min": 90.0},
            "ORD": {"type": "Ground Stop", "delay_min": 90.0, "reason": "VOLUME",
                    "end_time_utc": None, "max_delay_min": 90.0},
        },
        ok=True,
    )
    con = _con()

    assert snapshot_nas_status(con) == 2
    assert con.execute("SELECT airports FROM faa_nas_fetch").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM nas_status").fetchone()[0] == 2


def test_el_watchdog_mira_el_latido_y_no_los_programas() -> None:
    """
    La alarma sigue llamandose `nas_status` —es lo que le importa a quien la
    lee— pero la tabla que consulta es otra. Si volvieran a coincidir, una
    noche sin demoras en el pais volveria a sonar.
    """
    import api

    tabla, columna, _tolerado, _fuente = api.SOURCE_FRESHNESS["nas_status"]
    assert tabla == "faa_nas_fetch"
    assert columna == "checked_at_utc"


def test_el_watchdog_reporta_bajo_el_nombre_de_la_alarma() -> None:
    """La clave del resultado es el nombre, no la tabla leida."""
    import api

    con = _con()
    con.execute(
        "INSERT INTO faa_nas_fetch (checked_at_utc, airports) VALUES (datetime('now'), 0)"
    )
    con.commit()

    fuentes = api._source_freshness(con)
    assert "nas_status" in fuentes
    assert "faa_nas_fetch" not in fuentes
    assert fuentes["nas_status"]["stale"] is False


def test_el_flag_del_scraper_arranca_en_falso() -> None:
    """
    Un cliente recien construido no consulto nada todavia. Si arrancara en
    True, un fallo de red en el primer `_refresh` —que no toca el flag por no
    llegar a ejecutarse— pasaria por consulta exitosa.
    """
    from feature_engineering_v7.gdp_scraper import GdpClient

    assert GdpClient().last_refresh_ok is False
