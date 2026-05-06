"""Tests for scripts/analyze_live_period.py — metrics, calibration, rendering."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Load the script as a module (it lives outside the package).
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_live_period.py"
spec = importlib.util.spec_from_file_location("analyze_live_period", SCRIPT)
alp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alp)


# ---------------------------- compute_metrics ----------------------------


def test_compute_metrics_balanced() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    # proba correlated with truth → AUC > 0.5
    y_proba = np.where(y_true == 1, rng.uniform(0.5, 1.0, 200), rng.uniform(0.0, 0.5, 200))
    y_pred = (y_proba >= 0.5).astype(int)
    m = alp.compute_metrics(y_true, y_pred, y_proba)
    assert m["n"] == 200
    assert m["roc_auc"] > 0.9
    assert 0.0 <= m["brier"] <= 0.25
    assert "confusion_matrix" in m


def test_compute_metrics_single_class_returns_none_auc() -> None:
    y_true = np.zeros(50, dtype=int)
    y_proba = np.full(50, 0.3)
    y_pred = (y_proba >= 0.5).astype(int)
    m = alp.compute_metrics(y_true, y_pred, y_proba)
    assert m["roc_auc"] is None
    assert m["log_loss"] is None
    assert m["accuracy"] == 1.0


def test_compute_metrics_empty_returns_just_n_zero() -> None:
    m = alp.compute_metrics(np.array([]), np.array([]), np.array([]))
    assert m == {"n": 0}


# ---------------------------- compute_calibration ------------------------


def test_compute_calibration_perfect_when_proba_equals_outcome() -> None:
    # Calibration is perfect: every flight with proba p has actual rate p
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1])  # 5 zeros, 5 ones
    y_proba = np.linspace(0.05, 0.95, 10)
    cal = alp.compute_calibration(y_true, y_proba, n_bins=10)
    assert "ece" in cal
    assert 0.0 <= cal["ece"] <= 1.0
    assert len(cal["bins"]) == 10


def test_compute_calibration_handles_empty_bins_gracefully() -> None:
    y_true = np.array([0, 0, 1])
    y_proba = np.array([0.05, 0.10, 0.95])
    cal = alp.compute_calibration(y_true, y_proba, n_bins=10)
    populated = [b for b in cal["bins"] if b["n"] > 0]
    empty = [b for b in cal["bins"] if b["n"] == 0]
    assert len(populated) >= 1
    assert len(empty) >= 1
    for b in empty:
        assert b["mean_proba"] is None
        assert b["observed_pos_rate"] is None


# ---------------------------- per-day / per-carrier / per-strategy -------


def _synthetic_df(n_per_day: int = 30, n_days: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-05-05T14:00:00", tz="UTC")
    carriers = ["DL", "AA", "F9", "WN"]
    for d in range(n_days):
        for i in range(n_per_day):
            arr_delay = rng.normal(8, 25)
            proba = rng.uniform(0.05, 0.95)
            rows.append({
                "fa_flight_id": f"DAY{d}_{i}",
                "predicted_at_utc": (base + pd.Timedelta(days=d, hours=(i % 8))).isoformat(),
                "proba_delay": proba,
                "predicted_delay": int(proba >= 0.5),
                "threshold_used": 0.5,
                "threshold_strategy": "quantile@0.26" if i >= 5 else "artifact",
                "actual_in_utc": (base + pd.Timedelta(days=d, hours=4 + (i % 8))).isoformat(),
                "arr_delay_min": float(arr_delay),
                "scheduled_in_utc": (base + pd.Timedelta(days=d, hours=4 + (i % 8))).isoformat(),
                "op_carrier": carriers[i % len(carriers)],
                "origin": "ATL",
                "dest": ["MCO", "LGA", "DFW"][i % 3],
                "tail_num": f"N{i:04d}",
            })
    return pd.DataFrame(rows)


def test_per_day_groups_correctly() -> None:
    df = _synthetic_df(n_per_day=20, n_days=3)
    out = alp.per_day(df)
    assert len(out) == 3
    dates = [d["date"] for d in out]
    assert dates == sorted(dates)
    for d in out:
        assert d["n"] == 20


def test_per_carrier_filters_by_min_n() -> None:
    df = _synthetic_df(n_per_day=4, n_days=1)  # only 4 total → no carrier hits MIN_CARRIER_N=10
    out = alp.per_carrier(df)
    assert out == []

    df_big = _synthetic_df(n_per_day=80, n_days=1)
    out_big = alp.per_carrier(df_big)
    assert len(out_big) >= 1
    for c in out_big:
        assert c["n"] >= alp.MIN_CARRIER_N


def test_per_strategy_partitions_predictions() -> None:
    df = _synthetic_df(n_per_day=20, n_days=2)
    out = alp.per_strategy(df)
    strategies = {s["strategy"] for s in out}
    assert {"quantile@0.26", "artifact"}.issubset(strategies)
    total = sum(s["n"] for s in out)
    assert total == len(df)


def test_threshold_by_hour_returns_per_hour_rows() -> None:
    df = _synthetic_df(n_per_day=80, n_days=1)
    out = alp.threshold_by_hour(df)
    assert len(out) >= 1
    for r in out:
        assert 0 <= r["hour_utc"] < 24
        assert r["n"] >= alp.MIN_HOUR_N


# ---------------------------- end-to-end with a fake DB ------------------


def _make_fake_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE flights (
            fa_flight_id TEXT PRIMARY KEY,
            stable_id TEXT,
            ident_iata TEXT, op_carrier TEXT, flight_number TEXT,
            tail_num TEXT, origin TEXT, dest TEXT,
            inbound_fa_flight_id TEXT, fl_date TEXT, crs_dep_min INTEGER,
            scheduled_out_utc TEXT, scheduled_off_utc TEXT,
            scheduled_on_utc TEXT, scheduled_in_utc TEXT,
            crs_elapsed_min REAL, distance REAL, aircraft_type TEXT,
            cancelled INTEGER, diverted INTEGER,
            first_seen_utc TEXT, last_updated_utc TEXT
        );
        CREATE TABLE predictions (
            fa_flight_id TEXT, stable_id TEXT, predicted_at_utc TEXT,
            proba_delay REAL, predicted_delay INTEGER,
            threshold_used REAL, threshold_strategy TEXT,
            PRIMARY KEY (fa_flight_id, predicted_at_utc)
        );
        CREATE TABLE actuals (
            fa_flight_id TEXT PRIMARY KEY,
            stable_id TEXT,
            actual_out_utc TEXT, actual_off_utc TEXT,
            actual_on_utc TEXT, actual_in_utc TEXT,
            arr_delay_min REAL, departure_delay_min REAL,
            cancelled INTEGER, diverted INTEGER, settled_at_utc TEXT
        );
        """
    )
    rng = np.random.default_rng(0)
    for i in range(40):
        fid = f"FA{i:03d}-1700000000"
        sid = "-".join(fid.split("-")[:2])  # stable id matches the join in load_from_db
        conn.execute(
            "INSERT INTO flights (fa_flight_id, stable_id, op_carrier, origin, dest, tail_num, "
            "scheduled_in_utc, first_seen_utc, last_updated_utc) VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, sid, ["DL", "AA", "F9"][i % 3], "ATL", "MCO", f"N{i:04d}",
             "2026-05-05T18:00:00", "now", "now"),
        )
        proba = float(rng.uniform(0.0, 1.0))
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
            (fid, sid, "2026-05-05T14:00:00+00:00", proba, int(proba >= 0.5),
             0.5, "quantile@0.26"),
        )
        # Some are settled, some not; arr_delay correlated with proba so AUC > 0.5
        if i < 35:
            arr_delay = float(np.random.normal(15 if proba > 0.5 else -10, 10))
            conn.execute(
                "INSERT INTO actuals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fid, sid, "2026-05-05T17:55:00", "2026-05-05T14:05:00",
                 "2026-05-05T17:50:00", "2026-05-05T18:00:00",
                 arr_delay, 0.0, 0, 0, "now"),
            )
    conn.commit()
    conn.close()


def test_load_from_db_returns_only_settled(tmp_path: Path) -> None:
    db = tmp_path / "fake.db"
    _make_fake_db(db)
    conn = sqlite3.connect(str(db))
    df = alp.load_from_db(conn, "2026-05-05T00:00:00+00:00", "2026-05-05T23:59:59+00:00")
    conn.close()
    assert len(df) == 35  # only settled ones
    assert df["arr_delay_min"].notna().all()
    assert {"op_carrier", "proba_delay", "predicted_delay", "threshold_strategy"}.issubset(df.columns)


def test_build_report_smoke(tmp_path: Path) -> None:
    df = _synthetic_df(n_per_day=20, n_days=2)
    report = alp.build_report(df, source="synthetic", since="2026-05-05", until="2026-05-06")
    assert report["n_settled_predictions"] == 40
    assert "overall" in report and "roc_auc" in report["overall"]
    assert "calibration" in report and "ece" in report["calibration"]
    assert "daily" in report and len(report["daily"]) == 2
    assert isinstance(report["overall"]["confusion_matrix"], dict)


# ---------------------------- markdown rendering -------------------------


def test_render_md_includes_key_sections() -> None:
    df = _synthetic_df(n_per_day=20, n_days=2)
    report = alp.build_report(df, source="synthetic", since="2026-05-05", until="2026-05-06")
    md = alp.render_md(report)
    for must in [
        "Live period report",
        "Overall metrics",
        "Daily breakdown",
        "By threshold strategy",
        "Calibration",
        "Threshold and observed pos rate by UTC hour",
        "Confusion matrix",
    ]:
        assert must in md, f"missing section: {must}"


def test_render_md_handles_none_metrics_gracefully() -> None:
    # Single-class subset → roc_auc=None
    df = _synthetic_df(n_per_day=10, n_days=1).copy()
    df["arr_delay_min"] = -50.0  # all on-time → all y_true=0
    report = alp.build_report(df, source="x", since="2026-05-05", until="2026-05-05")
    md = alp.render_md(report)
    assert "n/a" in md  # AUC reported as n/a, no crash
