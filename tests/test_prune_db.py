"""Tests for bounded database pruning maintenance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from scripts import prune_db as prune_module


def _prunable_db(path: Path, *, old_predictions: int) -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE predictions(predicted_at_utc TEXT);
        CREATE TABLE prediction_shap(predicted_at_utc TEXT);
        CREATE TABLE flights(fa_flight_id TEXT, scheduled_out_utc TEXT);
        CREATE TABLE actuals(fa_flight_id TEXT);
        CREATE TABLE weather_obs(valid_utc TEXT);
        CREATE TABLE runs(started_utc TEXT);
        CREATE TABLE harvester_runs(run_at_utc TEXT);
        CREATE TABLE nas_status(captured_at_utc TEXT);
        CREATE TABLE aircraft_position(captured_at_utc TEXT);
        """
    )
    con.executemany(
        "INSERT INTO predictions(predicted_at_utc) VALUES (?)",
        [(old,)] * old_predictions,
    )
    con.commit()
    con.close()


def test_prune_skips_vacuum_below_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "skip.db"
    _prunable_db(db_path, old_predictions=2)
    vacuum_calls: list[bool] = []
    monkeypatch.setattr(
        prune_module,
        "_vacuum",
        lambda _con: vacuum_calls.append(True),
    )

    result = prune_module.prune_db(
        db_path,
        days=30,
        vacuum_min_deleted=2,
    )

    assert result == 0
    assert vacuum_calls == []


def test_prune_runs_vacuum_above_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "vacuum.db"
    _prunable_db(db_path, old_predictions=3)
    vacuum_calls: list[bool] = []
    monkeypatch.setattr(
        prune_module,
        "_vacuum",
        lambda _con: vacuum_calls.append(True),
    )

    result = prune_module.prune_db(
        db_path,
        days=30,
        vacuum_min_deleted=2,
    )

    assert result == 0
    assert vacuum_calls == [True]
