"""
`stable_id` tiene que distinguir instancias distintas del mismo vuelo.

Backend #64: consultar el historial de `SYN-DL2595-ATL-BOS-2026-09-16` devolvia
87 ciclos de un mes entero, porque la funcion recortaba a `SYN-DL2595` y con eso
colapsaba todas las corridas diarias de ese numero de vuelo en un identificador.
En produccion habia 1.141 identificadores asi, el peor con 27 instancias
distintas adentro.

La funcion existe para sacarle a AeroAPI un sufijo que cambia entre endpoints.
El error fue recortar por "tiene al menos dos segmentos" en vez de por la forma
que realmente tiene ese formato.
"""
from __future__ import annotations

import sqlite3

import pytest

import ontimeai.live as live
from ontimeai.live import SCHEMA, _repair_stable_ids, stable_id


class TestQueRecorta:
    def test_le_saca_a_aeroapi_el_sufijo_que_cambia(self) -> None:
        # El mismo vuelo fisico, como lo devuelven dos endpoints distintos.
        programado = stable_id("AAL1811-1777701683-airline-1446p")
        despegado = stable_id("AAL1811-1777701683-airline-1447p")
        assert programado == despegado == "AAL1811-1777701683"

    def test_no_toca_los_ids_sinteticos_del_harvester(self) -> None:
        fid = "SYN-DL2595-ATL-BOS-2026-09-16"
        assert stable_id(fid) == fid

    def test_dos_dias_del_mismo_vuelo_no_se_confunden(self) -> None:
        """El caso que reporto el issue: DL2595 vuela ATL-BOS todos los dias."""
        hoy = stable_id("SYN-DL2595-ATL-BOS-2026-09-16")
        ayer = stable_id("SYN-DL2595-ATL-BOS-2026-09-15")
        viejo = stable_id("SYN-DL2595-ATL-BOS-2026-08-15")
        assert len({hoy, ayer, viejo}) == 3

    def test_no_toca_los_hexadecimales_de_fr24(self) -> None:
        # El 99% de la base: sin guiones, ya son estables.
        assert stable_id("413898b0") == "413898b0"

    @pytest.mark.parametrize("vacio", ["", None])
    def test_vacio_devuelve_none(self, vacio) -> None:
        assert stable_id(vacio) is None

    def test_el_criterio_es_la_forma_y_no_el_prefijo(self) -> None:
        """
        Se mira si el segundo segmento son digitos —el timestamp de AeroAPI— y
        no si empieza con `SYN-`. Un formato nuevo cualquiera tiene que quedar
        entero sin que nadie se acuerde de agregarlo a una lista.
        """
        assert stable_id("NUEVO-PROVEEDOR-XYZ-2026") == "NUEVO-PROVEEDOR-XYZ-2026"
        assert stable_id("ABC-20260916-cosa") == "ABC-20260916"


class TestReparacion:
    """Las filas ya escritas con el valor degenerado hay que corregirlas."""

    @pytest.fixture(autouse=True)
    def _sin_guardado(self):
        """El guardado es por proceso; cada test necesita su propia pasada."""
        live._stable_ids_revisados = False
        yield
        live._stable_ids_revisados = False

    @pytest.fixture
    def con(self):
        c = sqlite3.connect(":memory:")
        c.executescript(SCHEMA)
        return c

    def _sembrar(self, con, fid, stable):
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
            (fid, stable, "2026-09-16T12:00:00+00:00", 0.3, 0),
        )
        con.commit()

    def test_corrige_un_sintetico_degenerado(self, con) -> None:
        fid = "SYN-DL2595-ATL-BOS-2026-09-16"
        self._sembrar(con, fid, "SYN-DL2595")

        assert _repair_stable_ids(con) == 1
        guardado = con.execute(
            "SELECT stable_id FROM predictions WHERE fa_flight_id = ?", (fid,)
        ).fetchone()[0]
        assert guardado == fid

    def test_no_toca_los_de_aeroapi(self, con) -> None:
        self._sembrar(con, "AAL1811-1777701683-airline-1446p", "AAL1811-1777701683")
        assert _repair_stable_ids(con) == 0

    def test_es_idempotente(self, con) -> None:
        """Corre en cada arranque: la segunda vez no puede tocar nada."""
        self._sembrar(con, "SYN-DL2595-ATL-BOS-2026-09-16", "SYN-DL2595")
        assert _repair_stable_ids(con) == 1
        assert _repair_stable_ids(con) == 0
        assert _repair_stable_ids(con) == 0

    def test_separa_instancias_que_estaban_juntas(self, con) -> None:
        """Lo que el issue vino a arreglar, de punta a punta."""
        for dia in ("2026-09-16", "2026-09-15", "2026-08-15"):
            self._sembrar(con, f"SYN-DL2595-ATL-BOS-{dia}", "SYN-DL2595")

        antes = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE stable_id = 'SYN-DL2595'"
        ).fetchone()[0]
        assert antes == 3, "las tres instancias compartian identificador"

        _repair_stable_ids(con)

        # Consultar una instancia ahora devuelve una sola fila, no tres.
        n = con.execute(
            "SELECT COUNT(*) FROM predictions WHERE stable_id = ?",
            ("SYN-DL2595-ATL-BOS-2026-09-16",),
        ).fetchone()[0]
        assert n == 1
        distintos = con.execute(
            "SELECT COUNT(DISTINCT stable_id) FROM predictions"
        ).fetchone()[0]
        assert distintos == 3


class TestGuardadoPorProceso:
    """
    `open_db` se llama por request —`_compute_shap` abre una conexion por
    vuelo—, asi que la reparacion no puede escanear las tres tablas cada vez.
    """

    def test_no_repite_el_escaneo_en_la_misma_corrida(self) -> None:
        live._stable_ids_revisados = False
        con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA)
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
            ("SYN-DL2595-ATL-BOS-2026-09-16", "SYN-DL2595",
             "2026-09-16T12:00:00+00:00", 0.3, 0),
        )
        con.commit()

        assert _repair_stable_ids(con) == 1
        # Sin el guardado esto volveria a escanear; con el, sale de una.
        assert _repair_stable_ids(con) == 0
        live._stable_ids_revisados = False
