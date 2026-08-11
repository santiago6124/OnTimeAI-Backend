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
    monkeypatch.setattr(api, "_DB_REFRESH_THREAD_LOCK", threading.Lock())
    monkeypatch.setattr(api, "_db_refresh_thread", None)
    monkeypatch.setattr(api, "_db_last_refresh", 0.0)
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


def test_concurrent_request_keeps_reading_previous_complete_snapshot(
    refresh_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "new.db"
    _create_snapshot(source, "new")
    download_started = threading.Event()
    finish_download = threading.Event()
    download_count = 0

    def slow_download(destination: Path) -> int:
        nonlocal download_count
        download_count += 1
        destination.write_bytes(b"partial download")
        download_started.set()
        assert finish_download.wait(timeout=5)
        shutil.copyfile(source, destination)
        return 44

    monkeypatch.setattr(api, "_download_db_snapshot", slow_download)

    # The request starts the refresh in the background and immediately opens
    # the previous snapshot instead of waiting for the large GCS download.
    with api.get_db() as active:
        assert _read_marker(active) == "old"
    assert download_started.wait(timeout=5)

    # Concurrent requests keep reading the untouched active DB.
    with api.get_db() as active:
        assert _read_marker(active) == "old"

    finish_download.set()
    refresh_thread = api._db_refresh_thread
    assert refresh_thread is not None
    refresh_thread.join(timeout=5)
    assert not refresh_thread.is_alive()
    assert download_count == 1
    with api.get_db() as installed:
        assert _read_marker(installed) == "new"
