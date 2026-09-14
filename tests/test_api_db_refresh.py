"""Regression tests for atomic API refreshes of the shared SQLite DB."""
from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import api


def _create_snapshot(path: Path, marker: str) -> None:
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE flights(fa_flight_id TEXT PRIMARY KEY);
            CREATE TABLE snapshot_marker(value TEXT NOT NULL);
            """
        )
        con.execute("INSERT INTO flights VALUES ('TEST-1')")
        con.execute("INSERT INTO snapshot_marker VALUES (?)", (marker,))


def _read_marker(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT value FROM snapshot_marker").fetchone()
    assert row is not None
    return str(row[0])


@pytest.fixture
def refresh_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "live_data.db"
    _create_snapshot(target, "old")
    monkeypatch.setattr(api, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(api, "_TMP_DB", target)
    monkeypatch.setattr(api, "DB_PATH", target)
    monkeypatch.setattr(api, "_DB_REFRESH_LOCK", threading.Lock())
    # No alcanza con 0.0: el disparador es
    # `time.monotonic() - _db_last_refresh < _DB_REFRESH_INTERVAL`, y
    # `time.monotonic()` cuenta desde el arranque de la máquina. En un runner
    # de CI recién booteado vale unos pocos cientos de segundos, por debajo del
    # intervalo, así que con 0.0 el refresh no se dispara y los tests que
    # dependen del camino automático esperan un evento que nunca llega.
    # Restar el intervalo al reloj actual deja el refresh vencido con
    # independencia del uptime.
    monkeypatch.setattr(
        api, "_db_last_refresh", time.monotonic() - api._DB_REFRESH_INTERVAL - 1
    )
    monkeypatch.setattr(api, "_db_last_health_check", time.monotonic())
    return target


def test_refresh_atomically_installs_verified_snapshot(
    refresh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "new.db"
    _create_snapshot(source, "new")
    downloads: list[Path] = []

    def download(destination: Path) -> int:
        downloads.append(destination)
        assert destination != refresh_env
        with sqlite3.connect(refresh_env) as active:
            assert _read_marker(active) == "old"
        shutil.copyfile(source, destination)
        return 42

    monkeypatch.setattr(api, "_download_db_snapshot", download)

    assert api._refresh_db_from_gcs(force=True) is True
    with api.get_db() as installed:
        assert _read_marker(installed) == "new"
    assert len(downloads) == 1

    # Startup marked the snapshot fresh, so the first request does not fetch it again.
    assert api._refresh_db_from_gcs() is False
    assert len(downloads) == 1
    assert not list(tmp_path.glob(".live_data.db.*.tmp"))


def test_invalid_download_never_replaces_active_snapshot(
    refresh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(destination: Path) -> int:
        destination.write_bytes(b"not a sqlite database")
        return 43

    monkeypatch.setattr(api, "_download_db_snapshot", download)

    assert api._refresh_db_from_gcs(force=True) is False
    with sqlite3.connect(refresh_env) as active:
        assert _read_marker(active) == "old"
    assert not list(tmp_path.glob(".live_data.db.*.tmp"))


def test_refresh_is_synchronous_so_it_gets_cpu(
    refresh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    El refresh ocurre dentro del pedido, no en un hilo de fondo.

    Antes se delegaba a un hilo para no demorar la respuesta. Esa decision es
    razonable en general, pero no sobrevive a Cloud Run: con throttling —el
    default— el contenedor recibe CPU solo mientras procesa un pedido. El hilo
    quedaba sin CPU apenas se enviaba la respuesta y la descarga de ~700 MB caia
    a ~2 MB/s hasta morir en el deadline. El backend servia datos de horas
    atras respondiendo 200, sin ninguna senal.

    El precio es que un pedido cada ~16 minutos espera la descarga. En la
    practica lo paga el vigia, que es quien mas consulta.
    """
    source = tmp_path / "new.db"
    _create_snapshot(source, "new")
    downloads = 0

    def download(destination: Path) -> int:
        nonlocal downloads
        downloads += 1
        shutil.copyfile(source, destination)
        return 44

    monkeypatch.setattr(api, "_download_db_snapshot", download)

    # El pedido no vuelve hasta tener el snapshot nuevo.
    with api.get_db() as con:
        assert _read_marker(con) == "new"
    assert downloads == 1


def test_concurrent_requests_do_not_pile_up_on_the_download(
    refresh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Solo el primer pedido vencido paga la espera.

    El lock es no bloqueante: mientras uno descarga, los demas siguen sirviendo
    el snapshot anterior en vez de encolarse. Sin eso, una descarga lenta
    convertiria cada pedido concurrente en una espera de minutos.
    """
    source = tmp_path / "new.db"
    _create_snapshot(source, "new")
    in_download = threading.Event()
    release = threading.Event()
    downloads = 0

    def slow_download(destination: Path) -> int:
        nonlocal downloads
        downloads += 1
        in_download.set()
        assert release.wait(timeout=5)
        shutil.copyfile(source, destination)
        return 44

    monkeypatch.setattr(api, "_download_db_snapshot", slow_download)

    first = threading.Thread(target=lambda: api.get_db().close(), daemon=True)
    first.start()
    assert in_download.wait(timeout=5), "el primer pedido deberia estar descargando"

    # Mientras tanto, otro pedido responde ya con el snapshot viejo.
    with api.get_db() as con:
        assert _read_marker(con) == "old"

    release.set()
    first.join(timeout=5)
    assert not first.is_alive()
    assert downloads == 1, "no deberia descargarse dos veces en paralelo"

    with api.get_db() as con:
        assert _read_marker(con) == "new"


def test_download_retries_when_a_job_replaces_the_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Tres jobs reescriben el objeto en GCS y no hay versionado, asi que la
    generacion leida puede desaparecer antes de terminar la descarga de
    ~685 MB. El lector tiene que reintentar, como ya hacen los escritores.
    """
    import google.cloud.storage as gcs_module
    from google.cloud.exceptions import NotFound

    source = tmp_path / "new.db"
    _create_snapshot(source, "new")
    generations = iter([111, 222])
    attempts: list[int] = []

    class _Blob:
        generation = 0

        def reload(self):
            type(self).generation = next(generations)

        def download_to_filename(self, destination, **kwargs):
            requested = kwargs["if_generation_match"]
            attempts.append(requested)
            if requested == 111:
                # Un job la reemplazo mientras se descargaba.
                raise NotFound("No such object")
            shutil.copyfile(source, Path(destination))

    blob = _Blob()

    class _Client:
        def bucket(self, _name):
            return type("B", (), {"blob": lambda _s, _n: blob})()

    monkeypatch.setattr(api, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(gcs_module, "Client", _Client)

    destination = tmp_path / "downloaded.db"
    assert api._download_db_snapshot(destination) == 222
    assert attempts == [111, 222], "deberia reintentar con la generacion nueva"
    with sqlite3.connect(destination) as con:
        assert _read_marker(con) == "new"


def test_download_gives_up_after_repeated_generation_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si el objeto se reemplaza en cada intento, el error tiene que salir."""
    import itertools

    import google.cloud.storage as gcs_module
    from google.cloud.exceptions import NotFound

    counter = itertools.count(1)

    class _Blob:
        generation = 0

        def reload(self):
            type(self).generation = next(counter)

        def download_to_filename(self, destination, **kwargs):
            raise NotFound("No such object")

    blob = _Blob()

    class _Client:
        def bucket(self, _name):
            return type("B", (), {"blob": lambda _s, _n: blob})()

    monkeypatch.setattr(api, "GCS_BUCKET", "test-bucket")
    monkeypatch.setattr(gcs_module, "Client", _Client)
    monkeypatch.setattr(api, "_DB_DOWNLOAD_GENERATION_RETRIES", 2)

    with pytest.raises(NotFound):
        api._download_db_snapshot(tmp_path / "x.db")
