"""Tests for the live SQLite schema, migrations, and chain-walk logic."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from ontimeai import live as live_mod
from ontimeai.live import (
    aeroapi_to_flight_row,
    chain_walk_inbound,
    open_db,
    upsert_flights,
)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_open_db_fresh_includes_threshold_columns(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "fresh.db")
    cols = _columns(conn, "predictions")
    assert {"threshold_used", "threshold_strategy"}.issubset(cols)


def test_open_db_migrates_legacy_predictions_table(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """CREATE TABLE predictions (
            fa_flight_id TEXT NOT NULL,
            predicted_at_utc TEXT NOT NULL,
            proba_delay REAL NOT NULL,
            predicted_delay INTEGER NOT NULL,
            PRIMARY KEY (fa_flight_id, predicted_at_utc)
        )"""
    )
    legacy.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?)",
        ("FA1", "2026-04-27T00:00:00Z", 0.42, 1),
    )
    legacy.commit()
    legacy.close()

    conn = open_db(db_path)
    cols = _columns(conn, "predictions")
    assert "threshold_used" in cols
    assert "threshold_strategy" in cols

    row = conn.execute(
        "SELECT proba_delay, threshold_used, threshold_strategy FROM predictions"
    ).fetchone()
    assert row[0] == pytest.approx(0.42)
    assert row[1] is None
    assert row[2] is None


def test_open_db_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "idem.db"
    open_db(db_path).close()
    conn = open_db(db_path)  # second open must not raise
    cols = _columns(conn, "predictions")
    assert {"threshold_used", "threshold_strategy"}.issubset(cols)


# ----------------------------- chain-walk tests ---------------------------


def _ar(rec: dict) -> dict:
    """Minimal AeroAPI-like flight record for tests."""
    base = {
        "fa_flight_id": "FA_TARGET_1",
        "ident_iata": "DL100",
        "operator_iata": "DL",
        "flight_number": "100",
        "registration": "N100DL",
        "origin": {"code_iata": "ATL"},
        "destination": {"code_iata": "MCO"},
        "scheduled_out": "2026-04-27T18:00:00Z",
        "scheduled_off": "2026-04-27T18:15:00Z",
        "scheduled_on": "2026-04-27T19:55:00Z",
        "scheduled_in": "2026-04-27T20:05:00Z",
        "route_distance": 400,
        "aircraft_type": "B738",
        "cancelled": False,
        "diverted": False,
        "inbound_fa_flight_id": None,
    }
    base.update(rec)
    return base


def test_aeroapi_to_flight_row_relaxed_filter_accepts_unknown_origin() -> None:
    # Use a fake IATA guaranteed not in any universe — international Buenos
    # Aires (EZE) is not in the US-only BTS universe.
    rec = _ar({"origin": {"code_iata": "EZE"}, "destination": {"code_iata": "ATL"}})
    assert aeroapi_to_flight_row(rec, strict_airport_filter=True) is None
    row = aeroapi_to_flight_row(rec, strict_airport_filter=False)
    assert row is not None
    assert row["origin"] == "EZE"
    assert row["dest"] == "ATL"


def test_chain_walk_skips_already_settled_inbounds(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "cw.db")
    # Pre-populate actuals so chain-walk should skip
    conn.execute(
        """INSERT INTO actuals (fa_flight_id, actual_in_utc, settled_at_utc)
           VALUES (?, ?, ?)""",
        ("FA_INBOUND_1", "2026-04-27T17:30:00Z", "2026-04-27T17:35:00Z"),
    )
    conn.commit()

    targets = [
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_TARGET_1",
            "inbound_fa_flight_id": "FA_INBOUND_1",
        }))
    ]

    with patch.object(live_mod, "fetch_flight_by_id") as fetch_mock:
        calls, written = chain_walk_inbound(conn, targets, key="dummy", max_calls=10)

    assert calls == 0
    assert written == 0
    fetch_mock.assert_not_called()


def test_chain_walk_hydrates_unsettled_inbound(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "cw2.db")

    targets = [
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_TARGET_2",
            "inbound_fa_flight_id": "FA_INBOUND_2",
        }))
    ]

    inbound_payload = [_ar({
        "fa_flight_id": "FA_INBOUND_2",
        "ident_iata": "DL99",
        "registration": "N100DL",
        "origin": {"code_iata": "SFO"},  # outside tracked network — relaxed filter
        "destination": {"code_iata": "ATL"},
        "scheduled_out": "2026-04-27T13:00:00Z",
        "scheduled_off": "2026-04-27T13:15:00Z",
        "scheduled_on": "2026-04-27T16:45:00Z",
        "scheduled_in": "2026-04-27T17:00:00Z",
        "actual_in": "2026-04-27T17:18:00Z",
        "actual_off": "2026-04-27T13:30:00Z",
        "actual_on": "2026-04-27T17:05:00Z",
        "actual_out": "2026-04-27T13:08:00Z",
        "arrival_delay": 18 * 60,  # 18 min, AeroAPI returns seconds
        "departure_delay": 5 * 60,
    })]

    with patch.object(live_mod, "fetch_flight_by_id", return_value=inbound_payload) as fetch_mock:
        with patch.object(live_mod.time, "sleep"):  # don't actually sleep in tests
            calls, written = chain_walk_inbound(conn, targets, key="dummy", max_calls=10)

    assert calls == 1
    assert written == 1
    fetch_mock.assert_called_once_with("FA_INBOUND_2", key="dummy")

    # Inbound flight should be in flights table (relaxed filter accepted SFO origin)
    flights_row = conn.execute(
        "SELECT origin, dest, tail_num FROM flights WHERE fa_flight_id = ?",
        ("FA_INBOUND_2",),
    ).fetchone()
    assert flights_row == ("SFO", "ATL", "N100DL")

    # Inbound actuals should be in actuals table
    actuals_row = conn.execute(
        "SELECT arr_delay_min FROM actuals WHERE fa_flight_id = ?",
        ("FA_INBOUND_2",),
    ).fetchone()
    assert actuals_row[0] == pytest.approx(18.0, abs=0.01)


def test_chain_walk_respects_max_calls_budget(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "cw3.db")

    targets = [
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": f"FA_TARGET_{i}",
            "inbound_fa_flight_id": f"FA_INBOUND_{i}",
        }))
        for i in range(10)
    ]

    with patch.object(live_mod, "fetch_flight_by_id", return_value=[]) as fetch_mock:
        with patch.object(live_mod.time, "sleep"):
            calls, written = chain_walk_inbound(conn, targets, key="dummy", max_calls=3)

    assert calls == 3  # capped at budget
    assert written == 0
    assert fetch_mock.call_count == 3


def test_chain_walk_dedups_repeated_inbound_ids(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "cw4.db")

    # Two different targets share the same inbound (e.g., codeshare or duplicate listing)
    targets = [
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_TARGET_A",
            "inbound_fa_flight_id": "FA_SHARED_INBOUND",
        })),
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_TARGET_B",
            "inbound_fa_flight_id": "FA_SHARED_INBOUND",
        })),
    ]

    with patch.object(live_mod, "fetch_flight_by_id", return_value=[]) as fetch_mock:
        with patch.object(live_mod.time, "sleep"):
            calls, _ = chain_walk_inbound(conn, targets, key="dummy", max_calls=10)

    assert calls == 1
    assert fetch_mock.call_count == 1


def test_chain_walk_swallows_api_error_and_continues(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "cw5.db")

    targets = [
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_T_OK",
            "inbound_fa_flight_id": "FA_OK",
        })),
        aeroapi_to_flight_row(_ar({
            "fa_flight_id": "FA_T_BAD",
            "inbound_fa_flight_id": "FA_BAD",
        })),
    ]

    def side_effect(fa_id, key=None):
        if fa_id == "FA_BAD":
            raise RuntimeError("AeroAPI 500 simulated")
        return [_ar({
            "fa_flight_id": fa_id,
            "registration": "N100DL",
            "actual_in": "2026-04-27T17:00:00Z",
            "scheduled_in": "2026-04-27T16:50:00Z",
            "arrival_delay": 600,
        })]

    with patch.object(live_mod, "fetch_flight_by_id", side_effect=side_effect):
        with patch.object(live_mod.time, "sleep"):
            calls, written = chain_walk_inbound(conn, targets, key="dummy", max_calls=10)

    assert calls == 1  # only FA_OK counted; FA_BAD raised before increment
    assert written == 1
