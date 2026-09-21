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
    # La crea el scrapper, no el SCHEMA del backend, pero en produccion vive en
    # la misma base y la API la lee.
    c.execute(
        "CREATE TABLE IF NOT EXISTS aircraft_position ("
        " icao24 TEXT NOT NULL, captured_at_utc TEXT NOT NULL, registration TEXT)"
    )
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


class TestFrescuraPorFuente:
    """
    Una fuente que deja de escribir tiene que ser visible.

    ADS-B dejo de escribir `aircraft_position` el 12/08 y estuvo 25 dias
    inerte: las corridas figuraban exitosas, no habia errores, y dos de las
    cuatro etapas de ajuste recibian None en cada ciclo. Ver #9.
    """

    def test_una_tabla_vacia_cuenta_como_caida(self, con_migrada) -> None:
        # Es el estado exacto en que quedo aircraft_position: la purga de 30
        # dias se llevo lo ultimo que habia y nada volvio a escribir.
        fuentes = api._source_freshness(con_migrada)
        assert fuentes["aircraft_position"]["stale"] is True
        assert fuentes["aircraft_position"]["age_minutes"] is None

    def test_una_escritura_reciente_no_esta_caida(self, con_migrada) -> None:
        from datetime import datetime, timezone

        ahora = datetime.now(timezone.utc).isoformat()
        con_migrada.execute(
            "INSERT INTO actuals (fa_flight_id, stable_id, settled_at_utc)"
            " VALUES (?,?,?)", ("X", "X", ahora),
        )
        fuentes = api._source_freshness(con_migrada)
        assert fuentes["actuals"]["stale"] is False

    def test_una_escritura_vieja_esta_caida(self, con_migrada) -> None:
        from datetime import datetime, timedelta, timezone

        viejo = (datetime.now(timezone.utc) - timedelta(minutes=300)).isoformat()
        con_migrada.execute(
            "INSERT INTO actuals (fa_flight_id, stable_id, settled_at_utc)"
            " VALUES (?,?,?)", ("X", "X", viejo),
        )
        fuentes = api._source_freshness(con_migrada)
        assert fuentes["actuals"]["stale"] is True
        assert fuentes["actuals"]["age_minutes"] == pytest.approx(300, abs=2)

    def test_una_fecha_sin_zona_se_lee_como_utc(self, con_migrada) -> None:
        # weather_obs guarda `valid_utc` sin offset. Leerla como hora local
        # daria 3 horas de mas en Argentina y marcaria caida una fuente sana.
        from datetime import datetime, timezone

        ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        con_migrada.execute(
            "INSERT INTO weather_obs (station, valid_utc) VALUES (?,?)", ("ATL", ahora)
        )
        fuentes = api._source_freshness(con_migrada)
        assert fuentes["weather_obs"]["stale"] is False

    def test_una_tabla_que_no_existe_se_omite(self, con_migrada) -> None:
        """
        Distinto de una tabla vacia. Que no exista significa que esa fuente
        nunca se instalo en esta base —el backend puede leer una creada solo
        por su propio SCHEMA—, no que dejo de escribir. Reportarla como caida
        seria ruido permanente.
        """
        con_migrada.execute("DROP TABLE aircraft_position")
        fuentes = api._source_freshness(con_migrada)
        assert "aircraft_position" not in fuentes
        assert "actuals" in fuentes


class TestDuracionDeCiclos:
    """
    El job baja la base entera de GCS y la vuelve a subir en cada ciclo, asi que
    su duracion crece con el archivo. Los predictores corren cada 15 min: cuando
    el ciclo se acerca a esa ventana, dos jobs terminan escribiendo la misma
    base y pisandose las subidas. Ver issue #4.
    """

    def _run(self, con, started, finished) -> None:
        con.execute(
            "INSERT INTO runs (started_utc, finished_utc) VALUES (?,?)",
            (started, finished),
        )

    def test_calcula_mediana_y_maximo(self, con_migrada) -> None:
        self._run(con_migrada, "2026-09-15T01:00:00+00:00", "2026-09-15T01:05:00+00:00")
        self._run(con_migrada, "2026-09-15T02:00:00+00:00", "2026-09-15T02:07:00+00:00")
        self._run(con_migrada, "2026-09-15T03:00:00+00:00", "2026-09-15T03:12:00+00:00")
        con_migrada.commit()

        c = api._recent_cycles(con_migrada)
        assert c["median_minutes"] == pytest.approx(7.0)
        assert c["max_minutes"] == pytest.approx(12.0)

    def test_un_ciclo_lento_aislado_no_dispara(self, con_migrada) -> None:
        """Por eso se compara la mediana y no el maximo."""
        for h in ("01", "02", "03", "04"):
            self._run(con_migrada, f"2026-09-15T{h}:00:00+00:00", f"2026-09-15T{h}:05:00+00:00")
        self._run(con_migrada, "2026-09-15T05:00:00+00:00", "2026-09-15T05:14:00+00:00")
        con_migrada.commit()

        c = api._recent_cycles(con_migrada)
        assert c["slow"] is False
        assert c["max_minutes"] == pytest.approx(14.0)

    def test_marca_lento_cuando_la_mediana_pasa_dos_tercios(self, con_migrada) -> None:
        # Dos tercios de 15 min son 10.
        for h in ("01", "02", "03"):
            self._run(con_migrada, f"2026-09-15T{h}:00:00+00:00", f"2026-09-15T{h}:11:00+00:00")
        con_migrada.commit()

        c = api._recent_cycles(con_migrada)
        assert c["slow"] is True
        assert c["tolerated_minutes"] == pytest.approx(10.0)

    def test_ignora_las_corridas_sin_terminar(self, con_migrada) -> None:
        self._run(con_migrada, "2026-09-15T01:00:00+00:00", "2026-09-15T01:05:00+00:00")
        self._run(con_migrada, "2026-09-15T02:00:00+00:00", None)
        con_migrada.commit()

        assert api._recent_cycles(con_migrada)["recent_minutes"] == [5.0]

    def test_sin_corridas_devuelve_vacio(self, con_migrada) -> None:
        assert api._recent_cycles(con_migrada) == {}

    def test_una_fecha_ilegible_no_rompe_el_resto(self, con_migrada) -> None:
        self._run(con_migrada, "no-es-una-fecha", "tampoco")
        self._run(con_migrada, "2026-09-15T01:00:00+00:00", "2026-09-15T01:05:00+00:00")
        con_migrada.commit()

        assert api._recent_cycles(con_migrada)["recent_minutes"] == [5.0]


class TestDuracionRealDelCiclo:
    """
    `runs.started_utc`/`finished_utc` los escribe live_pull y miden solo el
    pipeline. El ciclo real incluye arranque del contenedor, descarga de GCS,
    purga, VACUUM y subida.

    Medido el 15/09 en produccion: 1,7-3,2 min segun `runs` contra 4,7-7,5 de
    la ejecucion de Cloud Run. Una alerta construida sobre el numero chico no
    se dispararia nunca, aunque el job estuviera por solaparse con el siguiente.
    `job_seconds` lo escribe live_job y cubre el ciclo entero.
    """

    def _run(self, con, started, finished, job_seconds=None) -> None:
        con.execute(
            "INSERT INTO runs (started_utc, finished_utc, job_seconds) VALUES (?,?,?)",
            (started, finished, job_seconds),
        )

    @pytest.fixture
    def con_con_job_seconds(self, con_migrada):
        con_migrada.execute("ALTER TABLE runs ADD COLUMN job_seconds REAL")
        return con_migrada

    def test_prefiere_la_duracion_completa(self, con_con_job_seconds) -> None:
        con = con_con_job_seconds
        # El pipeline dice 2 min; el ciclo completo, 6.
        self._run(con, "2026-09-15T01:00:00+00:00", "2026-09-15T01:02:00+00:00", 360.0)
        con.commit()

        c = api._recent_cycles(con)
        assert c["median_minutes"] == pytest.approx(6.0), "uso el numero del pipeline"
        assert c["full_cycle_samples"] == 1

    def test_cae_al_pipeline_cuando_no_hay_dato_completo(self, con_con_job_seconds) -> None:
        con = con_con_job_seconds
        self._run(con, "2026-09-15T01:00:00+00:00", "2026-09-15T01:02:00+00:00", None)
        con.commit()

        c = api._recent_cycles(con)
        assert c["median_minutes"] == pytest.approx(2.0)
        assert c["full_cycle_samples"] == 0

    def test_reporta_cuantas_muestras_son_completas(self, con_con_job_seconds) -> None:
        """Mientras no sean todas, la mediana esta sesgada hacia abajo."""
        con = con_con_job_seconds
        self._run(con, "2026-09-15T01:00:00+00:00", "2026-09-15T01:02:00+00:00", 360.0)
        self._run(con, "2026-09-15T02:00:00+00:00", "2026-09-15T02:02:00+00:00", None)
        con.commit()

        c = api._recent_cycles(con)
        assert c["full_cycle_samples"] == 1
        assert c["total_samples"] == 2

    def test_una_base_sin_la_columna_sigue_funcionando(self, con_migrada) -> None:
        # Sin migrar todavia: se usa el pipeline y se avisa que no hay completas.
        con_migrada.execute(
            "INSERT INTO runs (started_utc, finished_utc) VALUES (?,?)",
            ("2026-09-15T01:00:00+00:00", "2026-09-15T01:03:00+00:00"),
        )
        con_migrada.commit()

        c = api._recent_cycles(con_migrada)
        assert c["median_minutes"] == pytest.approx(3.0)
        assert c["full_cycle_samples"] == 0


class TestArranqueSinGcs:
    """
    La imagen del job ya no trae `live_data.db`.

    Eran 81 MB que solo se usaban con GCS_BUCKET vacio, y la imagen se baja
    entera en cada ciclo: 96 veces por dia. Medido, el arranque en frio es el
    92% del tiempo de un job que no hace nada —111 s de 120—. Ver issue #4.

    En produccion GCS_BUCKET siempre esta; sin el es desarrollo local, y el
    fallo tiene que decir que falta en vez de reventar.
    """

    def test_sin_bucket_y_sin_db_bundleada_falla_con_mensaje(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        import live_job

        monkeypatch.setattr(live_job, "GCS_BUCKET", "")
        monkeypatch.setattr(live_job, "BUNDLED_DB", tmp_path / "no-existe.db")
        monkeypatch.setattr(live_job, "_validate_training_store_config", lambda: None)

        assert live_job.main() == 4
        salida = capsys.readouterr().out
        assert "GCS_BUCKET" in salida and "no-existe.db" in salida

    def test_sin_bucket_pero_con_db_local_sigue_andando(
        self, tmp_path, monkeypatch
    ) -> None:
        import live_job

        bundled = tmp_path / "live_data.db"
        bundled.write_bytes(b"SQLite format 3\x00")
        monkeypatch.setattr(live_job, "GCS_BUCKET", "")
        monkeypatch.setattr(live_job, "BUNDLED_DB", bundled)
        monkeypatch.setattr(live_job, "TMP_DB", tmp_path / "tmp.db")
        monkeypatch.setattr(live_job, "_validate_training_store_config", lambda: None)
        monkeypatch.setattr(live_job, "_handle_training_publish", lambda **kw: True)
        monkeypatch.setattr(live_job, "_run_pipeline_attempt", lambda args: 0)

        assert live_job.main() == 0
        assert (tmp_path / "tmp.db").exists()
