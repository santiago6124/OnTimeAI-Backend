"""Caracterización del ciclo del Cloud Run Job (`live_job.py`).

CP-COB-04 del trabajo de Calidad (OnTimeAI-Calidad).
`tests/test_live_job_generation.py` cubre la concurrencia optimista sobre la
generación de GCS, el VACUUM y la configuración del training store. Quedaban
sin caracterizar: la traducción de variables de entorno a flags de
`live_pull`, el intento de pipeline con sus pasos posteriores (agregados
diarios, calibrador post-cadena, purga) y su tolerancia a fallos, la
publicación del outbox de entrenamiento, el registro de la duración del
ciclo, los caminos de error de descarga/subida (objeto inexistente, base
corrupta) y las salidas de `main` que no son el camino feliz.

Nada toca GCS ni corre el pipeline real: `live_pull` se reemplaza por un
módulo falso en `sys.modules`, el blob por un doble en memoria y las bases
son SQLite temporales.
"""
from __future__ import annotations

import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

import live_job
from ontimeai.training_store import PublishSummary

# ── Dobles y bases ─────────────────────────────────────────────────────────


class _BlobFalso:
    def __init__(self, generation: int = 42) -> None:
        self.generation = str(generation)
        self.falla_reload: Exception | None = None
        self.falla_descarga: Exception | None = None
        self.subidas: list[int] = []

    def reload(self) -> None:
        if self.falla_reload is not None:
            raise self.falla_reload

    def download_to_filename(self, filename: str, *, if_generation_match: int) -> None:
        if self.falla_descarga is not None:
            raise self.falla_descarga
        Path(filename).write_bytes(b"snapshot")

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        self.subidas.append(if_generation_match)


def _db_valida(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE ok(id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()


def _db_con_runs(path: Path, *, con_job_seconds: bool = False) -> None:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, started_utc TEXT NOT NULL,"
        " finished_utc TEXT" + (", job_seconds REAL" if con_job_seconds else "") + ")"
    )
    con.execute("INSERT INTO runs (started_utc, finished_utc) VALUES ('2026-10-23T12:00:00', NULL)")
    con.execute("INSERT INTO runs (started_utc, finished_utc) VALUES ('2026-10-23T12:15:00', NULL)")
    con.commit()
    con.close()


class _ConexionQueFalla:
    """Envuelve una conexión real y hace fallar las sentencias elegidas."""

    def __init__(self, real: sqlite3.Connection, fallas: dict[str, object]) -> None:
        self._real = real
        self._fallas = fallas

    def execute(self, sql: str, *args):
        for fragmento, resultado in self._fallas.items():
            if fragmento in sql:
                if isinstance(resultado, Exception):
                    raise resultado
                return resultado
        return self._real.execute(sql, *args)

    def __getattr__(self, nombre: str):
        return getattr(self._real, nombre)


class _CursorFijo:
    def __init__(self, filas: list) -> None:
        self._filas = filas

    def fetchone(self):
        return self._filas[0] if self._filas else None

    def fetchall(self):
        return list(self._filas)


@pytest.fixture
def tmp_db(tmp_path: Path, monkeypatch) -> Path:
    target = tmp_path / "live.db"
    monkeypatch.setattr(live_job, "TMP_DB", target)
    return target


@pytest.fixture
def argv(monkeypatch) -> list[str]:
    """`_run_pipeline_attempt` pisa sys.argv; se restaura solo al salir."""
    original = ["live_job.py"]
    monkeypatch.setattr(sys, "argv", list(original))
    return original


def _live_pull_falso(monkeypatch, *, codigo: int) -> SimpleNamespace:
    registro = SimpleNamespace(llamadas=0, argv_visto=None)

    def _main() -> int:
        registro.llamadas += 1
        registro.argv_visto = list(sys.argv)
        return codigo

    modulo = types.ModuleType("live_pull")
    modulo.main = _main
    monkeypatch.setitem(sys.modules, "live_pull", modulo)
    return registro


# ── Configuración ──────────────────────────────────────────────────────────


class TestConfiguracion:
    def test_un_prefijo_vacio_del_training_store_se_rechaza(self, monkeypatch) -> None:
        # Arrange: bucket configurado pero prefijo en blanco.
        monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
        monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
        monkeypatch.setattr(live_job, "TRAINING_DATA_PREFIX", "")
        monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
        monkeypatch.delenv("TRAINING_STORE_ENABLED", raising=False)
        # Act / Assert
        with pytest.raises(RuntimeError, match="PREFIX"):
            live_job._validate_training_store_config()

    def test_las_variables_de_live_pull_se_traducen_a_flags_en_orden(self, monkeypatch) -> None:
        # Arrange: dos variables presentes, el resto ausente o vacío.
        for nombre in ("ACTUALS_HOURS", "ACTUALS_OFFSET_HOURS", "SCHEDULE_HOURS",
                       "CHAIN_WALK_MAX", "TARGET_POS_RATE", "ABS_THRESHOLD", "MAX_PAGES"):
            monkeypatch.delenv(nombre, raising=False)
        monkeypatch.setenv("MAX_PAGES", "3")
        monkeypatch.setenv("ACTUALS_HOURS", "6")
        monkeypatch.setenv("SCHEDULE_HOURS", "")
        # Act
        flags = live_job._live_pull_args_from_env()
        # Assert: el orden es el de la tabla del módulo, no el del entorno.
        assert flags == ["--actuals-hours", "6", "--max-pages", "3"]

    def test_sin_variables_no_hay_flags(self, monkeypatch) -> None:
        for nombre in ("ACTUALS_HOURS", "ACTUALS_OFFSET_HOURS", "SCHEDULE_HOURS",
                       "CHAIN_WALK_MAX", "TARGET_POS_RATE", "ABS_THRESHOLD", "MAX_PAGES"):
            monkeypatch.delenv(nombre, raising=False)
        assert live_job._live_pull_args_from_env() == []


# ── GCS ────────────────────────────────────────────────────────────────────


class TestGcs:
    def test_el_blob_se_pide_al_bucket_y_objeto_configurados(self, monkeypatch) -> None:
        # Arrange
        from google.cloud import storage

        pedidos: list[tuple[str, str]] = []

        class _Cliente:
            def bucket(self, nombre):
                return SimpleNamespace(blob=lambda objeto: pedidos.append((nombre, objeto)) or "blob")

        monkeypatch.setattr(storage, "Client", _Cliente)
        monkeypatch.setattr(live_job, "GCS_BUCKET", "mi-bucket")
        # Act
        blob = live_job._gcs_blob()
        # Assert
        assert blob == "blob"
        assert pedidos == [("mi-bucket", "live_data.db")]

    def test_sin_objeto_en_gcs_se_arranca_de_la_base_bundleada(self, tmp_path, tmp_db, monkeypatch, capsys) -> None:
        # Arrange: la primera corrida de un bucket vacío.
        blob = _BlobFalso()
        blob.falla_reload = NotFound("no existe")
        bundleada = tmp_path / "bundled.db"
        bundleada.write_bytes(b"base de la imagen")
        monkeypatch.setattr(live_job, "BUNDLED_DB", bundleada)
        monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)
        # Act
        generacion = live_job._gcs_download()
        # Assert: generación 0 (la subida va a crear el objeto) y copia local.
        assert generacion == 0
        assert tmp_db.read_bytes() == b"base de la imagen"
        assert "No hay DB en GCS" in capsys.readouterr().out

    def test_si_la_base_cambia_durante_la_descarga_es_un_conflicto(self, tmp_db, monkeypatch) -> None:
        blob = _BlobFalso(generation=42)
        blob.falla_descarga = PreconditionFailed("cambió")
        monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)
        with pytest.raises(live_job.GCSGenerationConflict, match="42"):
            live_job._gcs_download()

    def test_un_archivo_que_no_es_sqlite_no_se_sube(self, tmp_db, monkeypatch, capsys) -> None:
        # Arrange: basura en lugar de la base.
        tmp_db.write_bytes(b"esto no es una base sqlite" * 100)
        blob = _BlobFalso()
        monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)
        # Act / Assert: la verificación explota y se propaga; nada se subió.
        with pytest.raises(sqlite3.DatabaseError):
            live_job._gcs_upload(expected_generation=42)
        assert blob.subidas == []
        assert "ABORT upload" in capsys.readouterr().out

    def test_una_base_que_no_pasa_quick_check_no_se_sube(self, tmp_db, monkeypatch, capsys) -> None:
        # Arrange: la base existe, pero el motor reporta páginas rotas.
        _db_valida(tmp_db)
        real_connect = sqlite3.connect

        def _connect(path, *a, **k):
            return _ConexionQueFalla(real_connect(path, *a, **k), {
                "quick_check": _CursorFijo([("*** in database main ***\nPage 4: rota",)]),
                "integrity_check": _CursorFijo([("Page 4: btreeInitPage() returns error code 11",),
                                                ("Page 7: free list corrupt",)]),
            })

        monkeypatch.setattr(sqlite3, "connect", _connect)
        blob = _BlobFalso()
        monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)
        # Act / Assert
        with pytest.raises(RuntimeError, match="integrity check failed"):
            live_job._gcs_upload(expected_generation=42)
        salida = capsys.readouterr().out
        assert blob.subidas == []
        assert "DB corrupted" in salida
        assert "integrity_check: Page 4" in salida and "integrity_check: Page 7" in salida


class TestLimpieza:
    def test_un_vacuum_que_falla_no_frena_el_ciclo(self, tmp_db, monkeypatch, capsys) -> None:
        # Arrange: hay espacio por recuperar pero el VACUUM revienta.
        con = sqlite3.connect(tmp_db)
        con.execute("CREATE TABLE prediction_shap(predicted_at_utc TEXT)")
        con.execute("CREATE TABLE weather_obs(valid_utc TEXT)")
        con.commit()
        con.close()
        monkeypatch.setattr(live_job, "PRUNE_VACUUM_MIN_FREE_MB", 0)
        real_connect = sqlite3.connect
        monkeypatch.setattr(
            sqlite3, "connect",
            lambda path, *a, **k: _ConexionQueFalla(
                real_connect(path, *a, **k), {"VACUUM": sqlite3.OperationalError("database is locked")},
            ),
        )
        # Act
        live_job._cleanup_old_data()
        # Assert
        salida = capsys.readouterr().out
        assert "Running VACUUM" in salida
        assert "VACUUM skipped: database is locked" in salida


# ── Outbox de entrenamiento ────────────────────────────────────────────────


class TestOutbox:
    @pytest.fixture
    def publicador(self, monkeypatch):
        """Doble de `publish_pending_outbox`: registra los argumentos."""
        import ontimeai.training_store as ts

        registro = SimpleNamespace(llamadas=[], eventos=0)

        def _publish(con, *, bucket_name, prefix, mark_delivered):
            registro.llamadas.append({"bucket": bucket_name, "prefix": prefix,
                                      "mark_delivered": mark_delivered})
            return PublishSummary(events=registro.eventos, objects=1 if registro.eventos else 0,
                                  object_names=("a.parquet",) if registro.eventos else ())

        monkeypatch.setattr(ts, "publish_pending_outbox", _publish)
        monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
        monkeypatch.setattr(live_job, "TRAINING_DATA_PREFIX", "live-training")
        monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
        monkeypatch.delenv("TRAINING_OUTBOX_WARN_EVENTS", raising=False)
        return registro

    def _outbox(self, path: Path, filas: int) -> None:
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE training_export_outbox(created_at_utc TEXT)")
        con.executemany(
            "INSERT INTO training_export_outbox VALUES (?)",
            [(f"2026-10-23T1{i}:00:00+00:00",) for i in range(filas)],
        )
        con.commit()
        con.close()

    def test_sin_bucket_no_se_toca_la_base(self, tmp_db, monkeypatch) -> None:
        monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "")
        monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
        # TMP_DB no existe: si se intentara abrir, se crearía.
        assert live_job._publish_training_outbox(mark_delivered=True) == 0
        assert not tmp_db.exists()

    def test_una_base_sin_outbox_publica_igual_y_no_informa_backlog(self, tmp_db, publicador, capsys) -> None:
        # Arrange: base vieja, anterior a la tabla.
        _db_valida(tmp_db)
        # Act
        eventos = live_job._publish_training_outbox(mark_delivered=True)
        # Assert
        assert eventos == 0
        assert publicador.llamadas == [{"bucket": "training-bucket", "prefix": "live-training",
                                        "mark_delivered": True}]
        assert "backlog" not in capsys.readouterr().out

    def test_el_backlog_pendiente_se_informa_con_su_evento_mas_viejo(self, tmp_db, publicador, capsys) -> None:
        # Arrange
        self._outbox(tmp_db, filas=3)
        publicador.eventos = 3
        # Act
        eventos = live_job._publish_training_outbox(mark_delivered=True)
        # Assert
        salida = capsys.readouterr().out
        assert eventos == 3
        assert "backlog pending: 3 events; oldest=2026-10-23T10:00:00+00:00" in salida
        assert "published+acknowledged: 3 events in 1 immutable parquet object(s)" in salida

    def test_un_backlog_grande_sube_a_warning(self, tmp_db, publicador, monkeypatch, capsys) -> None:
        self._outbox(tmp_db, filas=3)
        monkeypatch.setenv("TRAINING_OUTBOX_WARN_EVENTS", "2")
        publicador.eventos = 3
        live_job._publish_training_outbox(mark_delivered=False)
        salida = capsys.readouterr().out
        assert "backlog WARNING: 3 events" in salida
        # Sin acknowledge, el estado que se informa es sólo "published".
        assert "[training-store] published: 3 events" in salida
        assert publicador.llamadas[0]["mark_delivered"] is False

    def test_un_fallo_de_publicacion_no_corta_si_el_store_es_opcional(self, monkeypatch, capsys) -> None:
        def _explota(*, mark_delivered):
            raise ConnectionError("bucket inalcanzable")

        monkeypatch.setattr(live_job, "_publish_training_outbox", _explota)
        monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
        assert live_job._handle_training_publish(mark_delivered=True) is True
        salida = capsys.readouterr().out
        assert "publish failed (mark_delivered=True): ConnectionError: bucket inalcanzable" in salida


# ── Intento de pipeline ────────────────────────────────────────────────────


class TestIntentoDePipeline:
    @pytest.fixture
    def pasos(self, monkeypatch):
        """Dobles de los pasos posteriores: rollup, calibrador y purga."""
        import ontimeai.chain_calibration as cc
        import scripts.prune_db as prune
        from ontimeai import rollup

        registro = SimpleNamespace(rollup=[], calibrador=[], purga=[],
                                   ajuste=None, falla_rollup=None, falla_purga=None)

        def _rollup(con, *, model_version=""):
            if registro.falla_rollup is not None:
                raise registro.falla_rollup
            registro.rollup.append(model_version)
            return 7

        def _fit(con):
            registro.calibrador.append(True)
            return registro.ajuste

        def _prune(db_path, *, days, dry_run):
            if registro.falla_purga is not None:
                raise registro.falla_purga
            registro.purga.append({"db": db_path, "days": days, "dry_run": dry_run})
            return 0

        monkeypatch.setattr(rollup, "rollup_daily_metrics", _rollup)
        monkeypatch.setattr(cc, "fit_chain_calibrator", _fit)
        monkeypatch.setattr(prune, "prune_db", _prune)
        return registro

    def test_si_live_pull_falla_se_devuelve_su_codigo_y_no_se_sigue(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        # Arrange
        _db_valida(tmp_db)
        pull = _live_pull_falso(monkeypatch, codigo=2)
        # Act
        codigo = live_job._run_pipeline_attempt([])
        # Assert
        assert codigo == 2 and pull.llamadas == 1
        assert pasos.rollup == [] and pasos.purga == []
        assert "live_pull failed with exit_code=2; upload skipped" in capsys.readouterr().out

    def test_los_flags_llegan_a_live_pull_por_argv(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        pull = _live_pull_falso(monkeypatch, codigo=0)
        live_job._run_pipeline_attempt(["--max-pages", "3"])
        assert pull.argv_visto == ["live_job.py", "--max-pages", "3"]
        assert "live_pull args from env: --max-pages 3" in capsys.readouterr().out

    def test_tras_el_pipeline_corren_agregados_calibrador_y_purga(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        # Arrange: sin muestra para calibrar.
        _db_valida(tmp_db)
        _live_pull_falso(monkeypatch, codigo=0)
        monkeypatch.setenv("ACTIVE_MODEL", "4year_v9")
        pasos.ajuste = None
        # Act
        codigo = live_job._run_pipeline_attempt([])
        # Assert
        salida = capsys.readouterr().out
        assert codigo == 0
        assert pasos.rollup == ["4year_v9"]
        assert pasos.calibrador == [True]
        assert pasos.purga == [{"db": tmp_db, "days": live_job.RETENTION_DAYS, "dry_run": False}]
        assert "Agregados diarios: 7 filas" in salida
        assert "sin muestra suficiente" in salida

    def test_el_calibrador_ajustado_informa_su_muestra(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        _db_valida(tmp_db)
        _live_pull_falso(monkeypatch, codigo=0)
        pasos.ajuste = {"n_samples": 1825, "window_days": 30, "base_rate": 0.21}
        live_job._run_pipeline_attempt([])
        assert "ajustado con 1,825 vuelos de 30 dias (tasa base 21.0%)" in capsys.readouterr().out

    def test_un_rollup_roto_no_frena_la_purga(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        # Perder un rollup cuesta un día de histórico; frenar el pipeline
        # cuesta la predicción de todos los vuelos.
        _db_valida(tmp_db)
        _live_pull_falso(monkeypatch, codigo=0)
        pasos.falla_rollup = sqlite3.OperationalError("no such table: actuals")
        codigo = live_job._run_pipeline_attempt([])
        salida = capsys.readouterr().out
        assert codigo == 0
        assert "Error computing daily rollup: no such table: actuals" in salida
        assert len(pasos.purga) == 1
        assert pasos.calibrador == []

    def test_una_purga_rota_tampoco_frena_el_ciclo(self, tmp_db, argv, pasos, monkeypatch, capsys) -> None:
        _db_valida(tmp_db)
        _live_pull_falso(monkeypatch, codigo=0)
        pasos.falla_purga = RuntimeError("disco lleno")
        assert live_job._run_pipeline_attempt([]) == 0
        assert "Error running database pruning: disco lleno" in capsys.readouterr().out

    def test_sin_base_local_no_hay_pasos_posteriores(self, tmp_db, argv, pasos, monkeypatch) -> None:
        _live_pull_falso(monkeypatch, codigo=0)
        assert not tmp_db.exists()
        assert live_job._run_pipeline_attempt([]) == 0
        assert pasos.rollup == [] and pasos.purga == []


# ── Duración del ciclo ─────────────────────────────────────────────────────


class TestDuracionDelCiclo:
    def test_agrega_la_columna_y_la_escribe_en_la_ultima_corrida(self, tmp_db, capsys) -> None:
        # Arrange: tabla `runs` anterior a la columna.
        _db_con_runs(tmp_db)
        # Act
        live_job._record_job_duration()
        # Assert
        con = sqlite3.connect(tmp_db)
        filas = con.execute("SELECT run_id, job_seconds FROM runs ORDER BY run_id").fetchall()
        con.close()
        assert filas[0][1] is None
        assert filas[1][1] is not None and filas[1][1] >= 0
        assert "ciclo completo" in capsys.readouterr().out

    def test_si_la_columna_ya_existe_la_reutiliza(self, tmp_db) -> None:
        _db_con_runs(tmp_db, con_job_seconds=True)
        live_job._record_job_duration()
        con = sqlite3.connect(tmp_db)
        columnas = [r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()]
        valor = con.execute("SELECT job_seconds FROM runs WHERE run_id = 2").fetchone()[0]
        con.close()
        assert columnas.count("job_seconds") == 1
        assert valor is not None

    def test_sin_tabla_runs_lo_deja_en_el_log_y_sigue(self, tmp_db, capsys) -> None:
        _db_valida(tmp_db)
        live_job._record_job_duration()
        assert "no se pudo registrar la duracion" in capsys.readouterr().out


# ── main ───────────────────────────────────────────────────────────────────


class _Publicaciones:
    """Secuencia de respuestas para `_handle_training_publish`."""

    def __init__(self, *respuestas: bool) -> None:
        self._respuestas = list(respuestas)
        self.llamadas: list[bool] = []

    def __call__(self, *, mark_delivered: bool) -> bool:
        self.llamadas.append(mark_delivered)
        return self._respuestas.pop(0) if self._respuestas else True


@pytest.fixture
def main_aislado(monkeypatch, tmp_db):
    """Un `main` sin GCS real ni pipeline: cada paso es un doble registrado."""
    registro = SimpleNamespace(pipeline=[], subidas=[], generaciones=iter([10, 11, 12]),
                               codigo_pipeline=0, falla_subida=None)
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
    monkeypatch.delenv("TRAINING_STORE_ENABLED", raising=False)
    monkeypatch.setattr(live_job, "_live_pull_args_from_env", list)
    monkeypatch.setattr(live_job, "_cleanup_old_data", lambda: None)
    monkeypatch.setattr(live_job, "_record_job_duration", lambda: None)
    monkeypatch.setattr(live_job, "_gcs_download", lambda: next(registro.generaciones))

    def _pipeline(args):
        registro.pipeline.append(list(args))
        return registro.codigo_pipeline

    def _subida(expected_generation):
        registro.subidas.append(expected_generation)
        if registro.falla_subida is not None:
            raise registro.falla_subida
        return expected_generation + 1

    monkeypatch.setattr(live_job, "_run_pipeline_attempt", _pipeline)
    monkeypatch.setattr(live_job, "_gcs_upload", _subida)
    return registro


class TestMainSinGcs:
    @pytest.fixture
    def bundleada(self, tmp_path, monkeypatch) -> Path:
        base = tmp_path / "bundled.db"
        base.write_bytes(b"base de la imagen")
        monkeypatch.setattr(live_job, "BUNDLED_DB", base)
        monkeypatch.setattr(live_job, "GCS_BUCKET", "")
        return base

    def test_corre_sobre_la_copia_local_y_marca_la_generacion(self, main_aislado, bundleada, tmp_db, monkeypatch) -> None:
        # Arrange
        publicaciones = _Publicaciones(True, True)
        monkeypatch.setattr(live_job, "_handle_training_publish", publicaciones)
        # Act
        codigo = live_job.main()
        # Assert
        assert codigo == 0
        assert tmp_db.read_bytes() == b"base de la imagen"
        assert live_job.os.environ["LIVE_DB_BASE_GENERATION"] == "local"
        assert main_aislado.pipeline == [[]]
        assert publicaciones.llamadas == [True, True]
        assert main_aislado.subidas == []

    def test_si_el_outbox_previo_no_se_puede_entregar_no_arranca(self, main_aislado, bundleada, monkeypatch) -> None:
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(False))
        assert live_job.main() == 4
        assert main_aislado.pipeline == []

    def test_si_la_entrega_posterior_falla_el_ciclo_se_marca_fallido(self, main_aislado, bundleada, monkeypatch) -> None:
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(True, False))
        assert live_job.main() == 4
        assert main_aislado.pipeline == [[]]

    def test_devuelve_el_codigo_de_live_pull(self, main_aislado, bundleada, monkeypatch) -> None:
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(True, True))
        main_aislado.codigo_pipeline = 2
        assert live_job.main() == 2

    def test_sin_bucket_ni_base_bundleada_explica_que_falta(self, main_aislado, tmp_path, monkeypatch, capsys) -> None:
        monkeypatch.setattr(live_job, "GCS_BUCKET", "")
        monkeypatch.setattr(live_job, "BUNDLED_DB", tmp_path / "no-existe.db")
        assert live_job.main() == 4
        assert "GCS_BUCKET no configurado y no hay" in capsys.readouterr().out


class TestMainConGcs:
    @pytest.fixture(autouse=True)
    def _bucket(self, monkeypatch) -> None:
        monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
        monkeypatch.setattr(live_job, "GCS_GENERATION_RETRIES", 1)

    def test_si_el_outbox_previo_no_se_entrega_no_corre_el_pipeline(self, main_aislado, tmp_db, monkeypatch) -> None:
        _db_valida(tmp_db)
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(False))
        assert live_job.main() == 4
        assert main_aislado.pipeline == [] and main_aislado.subidas == []

    def test_si_live_pull_falla_no_se_sube_nada(self, main_aislado, tmp_db, monkeypatch) -> None:
        _db_valida(tmp_db)
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(True))
        main_aislado.codigo_pipeline = 2
        assert live_job.main() == 2
        assert main_aislado.subidas == []

    def test_un_pipeline_que_no_deja_base_es_un_error_que_se_propaga(self, main_aislado, tmp_db, monkeypatch) -> None:
        # No es un conflicto de generación: no hay reintento que lo arregle.
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones(True))
        assert not tmp_db.exists()
        with pytest.raises(FileNotFoundError, match="pipeline did not produce"):
            live_job.main()
        assert main_aislado.subidas == []

    def test_si_la_publicacion_posterior_falla_el_ciclo_devuelve_4(self, main_aislado, tmp_db, monkeypatch) -> None:
        _db_valida(tmp_db)
        publicaciones = _Publicaciones(True, False)
        monkeypatch.setattr(live_job, "_handle_training_publish", publicaciones)
        assert live_job.main() == 4
        # La base ya se subió: lo que falló es la entrega del outbox nuevo.
        assert main_aislado.subidas == [10]
        assert publicaciones.llamadas == [True, False]

    def test_agotados_los_reintentos_se_niega_a_pisar_la_base(self, main_aislado, tmp_db, monkeypatch, capsys) -> None:
        # Arrange: otro escritor gana todas las veces.
        _db_valida(tmp_db)
        monkeypatch.setattr(live_job, "_handle_training_publish", _Publicaciones())
        main_aislado.falla_subida = live_job.GCSGenerationConflict("otro escritor")
        # Act
        codigo = live_job.main()
        # Assert: dos intentos (1 + 1 reintento), cada uno desde la generación
        # ganadora, y salida 3 sin sobrescribir.
        assert codigo == 3
        assert main_aislado.subidas == [10, 11]
        assert len(main_aislado.pipeline) == 2
        salida = capsys.readouterr().out
        assert "generation conflict: otro escritor" in salida
        assert "after 2 attempts; refusing stale overwrite" in salida
