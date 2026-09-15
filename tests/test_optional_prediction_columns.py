"""
La API tiene que poder leer una base que todavia no paso por las migraciones.

Las columnas que se fueron sumando a `predictions` —`threshold_used`,
`proba_raw`, `prediction_phase` y el resto— las crea un `ALTER TABLE` que corre
al abrir la base para escritura, o sea en el job. La API la abre en modo lectura
y puede estar sirviendo una que no paso por ahi: la bundleada en la imagen, que
es el fallback de arranque.

El 15/09 una consulta referencio `p.threshold_used` a secas y /metrics/summary y
/flights devolvieron 500 con `OperationalError: no such column` hasta el
rollback. Estos tests corren contra el esquema base, sin migrar, que es
exactamente el caso que fallo.
"""
from __future__ import annotations

import sqlite3

import pytest

import api
from ontimeai.live import SCHEMA


@pytest.fixture
def con_migrada() -> sqlite3.Connection:
    """El esquema vigente, que ya incluye `threshold_used`."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


@pytest.fixture
def con_sin_migrar(con_migrada: sqlite3.Connection) -> sqlite3.Connection:
    """La forma legacy de `predictions`, la que el ALTER TABLE viene a arreglar.

    El SCHEMA de hoy ya trae las columnas; las migraciones existen para las
    bases creadas antes. Se reconstruye la tabla sin ellas porque esa —y no el
    esquema actual— es la que la API puede encontrarse sirviendo.
    """
    con_migrada.executescript(
        """
        DROP TABLE predictions;
        CREATE TABLE predictions (
            fa_flight_id TEXT NOT NULL,
            stable_id TEXT,
            predicted_at_utc TEXT NOT NULL,
            proba_delay REAL NOT NULL,
            predicted_delay INTEGER NOT NULL,
            PRIMARY KEY (fa_flight_id, predicted_at_utc)
        );
        """
    )
    return con_migrada


class TestOptionalPredictionColumn:
    def test_una_columna_ausente_se_reemplaza_por_null(self, con_sin_migrar) -> None:
        assert (
            api._optional_prediction_column(con_sin_migrar, "threshold_used")
            == "NULL AS threshold_used"
        )

    def test_una_columna_presente_se_referencia(self, con_migrada) -> None:
        assert (
            api._optional_prediction_column(con_migrada, "threshold_used")
            == "p.threshold_used"
        )

    def test_una_columna_que_siempre_existio_se_referencia(self, con_sin_migrar) -> None:
        assert api._optional_prediction_column(con_sin_migrar, "proba_delay") == "p.proba_delay"


class TestConsultasContraBaseSinMigrar:
    """El caso del incidente: las consultas no pueden explotar."""

    def test_latest_predictions_active_no_explota(self, con_sin_migrar) -> None:
        assert api._latest_predictions_active(con_sin_migrar) == []

    def test_historical_flight_no_explota(self, con_sin_migrar) -> None:
        assert api._get_historical_flight(con_sin_migrar, "FA123") is None

    def test_tambien_funcionan_con_la_base_migrada(self, con_migrada) -> None:
        assert api._latest_predictions_active(con_migrada) == []
        assert api._get_historical_flight(con_migrada, "FA123") is None


class TestFilaConDatos:
    def _sembrar(self, con: sqlite3.Connection, *, con_umbral: bool) -> None:
        from datetime import datetime, timedelta, timezone

        pronto = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        con.execute(
            "INSERT INTO flights (fa_flight_id, ident_iata, op_carrier, flight_number,"
            " origin, dest, scheduled_out_utc, scheduled_in_utc, cancelled,"
            " first_seen_utc, last_updated_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("FA1", "DL100", "DL", "100", "ATL", "MIA", pronto, pronto, 0, pronto, pronto),
        )
        if con_umbral:
            con.execute(
                "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
                " proba_delay, predicted_delay, threshold_used) VALUES (?,?,?,?,?,?)",
                ("FA1", "DL100", pronto, 0.12, 1, 0.09),
            )
        else:
            con.execute(
                "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
                " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
                ("FA1", "DL100", pronto, 0.12, 1),
            )
        con.commit()

    def test_sin_la_columna_el_riesgo_cae_al_umbral_del_artefacto(
        self, con_sin_migrar
    ) -> None:
        self._sembrar(con_sin_migrar, con_umbral=False)
        filas = api._latest_predictions_active(con_sin_migrar)
        assert len(filas) == 1
        assert filas[0]["threshold_used"] is None
        # 0.12 contra el fallback 0.32 -> bajo, sin reventar.
        assert api.risk_level(0.12, api._row_threshold(filas[0])) == "low"

    def test_con_la_columna_el_riesgo_usa_el_umbral_real(self, con_migrada) -> None:
        self._sembrar(con_migrada, con_umbral=True)
        filas = api._latest_predictions_active(con_migrada)
        assert len(filas) == 1
        assert filas[0]["threshold_used"] == pytest.approx(0.09)
        assert api.risk_level(0.12, api._row_threshold(filas[0])) == "medium"
