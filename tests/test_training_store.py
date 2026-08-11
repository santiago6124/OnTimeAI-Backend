from __future__ import annotations

from datetime import date
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from google.api_core.exceptions import PreconditionFailed
import numpy as np
import pandas as pd
import pytest

import ontimeai.training_store as training_store_module
from ontimeai.live import open_db
from ontimeai.training_store import (
    TRAINING_SCHEMA_VERSION,
    build_flight_keys,
    enqueue_prediction_snapshots,
    enqueue_recent_outcomes,
    ensure_training_store_schema,
    materialize_fixed_horizons,
    pending_outbox_count,
    prepare_silver_training_data,
    prune_outcome_state,
    publish_pending_outbox,
    temporal_split_silver,
    transformer_code_sha256,
    training_store_enabled,
)
from scripts import materialize_live_training as materialize_cli


def test_training_store_rejects_required_disabled_conflict(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_STORE_REQUIRED", " true ")
    monkeypatch.setenv("TRAINING_STORE_ENABLED", "false")

    with pytest.raises(ValueError, match="conflicts"):
        training_store_enabled()


def test_training_store_rejects_enabled_capture_without_bucket(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_STORE_REQUIRED", "false")
    monkeypatch.setenv("TRAINING_STORE_ENABLED", "true")
    monkeypatch.delenv("TRAINING_DATA_BUCKET", raising=False)

    with pytest.raises(ValueError, match="requires TRAINING_DATA_BUCKET"):
        training_store_enabled()


class FakeBlob:
    def __init__(
        self,
        objects: dict[str, bytes],
        metadata_by_name: dict[str, dict[str, str]],
        name: str,
    ) -> None:
        self.objects = objects
        self.metadata_by_name = metadata_by_name
        self.name = name
        self.preconditions: list[int] = []
        self.metadata: dict[str, str] | None = None
        self.generation: int | None = None

    def upload_from_string(
        self,
        body: bytes,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None:
        assert content_type in {
            "application/vnd.apache.parquet", "application/json",
        }
        self.preconditions.append(if_generation_match)
        if self.name in self.objects:
            raise PreconditionFailed("already created")
        self.objects[self.name] = body
        self.metadata_by_name[self.name] = dict(self.metadata or {})

    def reload(self) -> None:
        self.metadata = dict(self.metadata_by_name.get(self.name, {}))
        self.generation = 1 if self.name in self.objects else None

    def download_as_bytes(self, *, if_generation_match: int) -> bytes:
        assert self.generation is not None
        assert if_generation_match == self.generation
        return self.objects[self.name]


class FakeBucket:
    def __init__(
        self,
        objects: dict[str, bytes],
        metadata_by_name: dict[str, dict[str, str]],
    ) -> None:
        self.objects = objects
        self.metadata_by_name = metadata_by_name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.objects, self.metadata_by_name, name)


class FakeStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata_by_name: dict[str, dict[str, str]] = {}

    def bucket(self, _name: str) -> FakeBucket:
        return FakeBucket(self.objects, self.metadata_by_name)


def _flight_row(fa_flight_id: str = "DAL123-123-airline-a") -> dict:
    return {
        "fa_flight_id": fa_flight_id,
        "stable_id": "DAL123-123",
        "op_carrier": "dl",
        "flight_number": "123.0",
        "tail_num": "n123dl",
        "origin": "atl",
        "dest": "mco",
        "fl_date": "2026-08-11",
        "scheduled_out_utc": "2026-08-11T12:00:00+00:00",
        "scheduled_off_utc": "2026-08-11T12:10:00+00:00",
        "scheduled_on_utc": "2026-08-11T13:10:00+00:00",
        "scheduled_in_utc": "2026-08-11T13:20:00+00:00",
        "estimated_out_utc": "2026-08-11T12:05:00+00:00",
        "estimated_in_utc": "2026-08-11T13:25:00+00:00",
    }


def test_flight_key_is_source_independent_when_schedule_identity_matches() -> None:
    aero = _flight_row("DAL123-123-airline-a")
    fr24 = {**aero, "fa_flight_id": "fr24-abcdef", "stable_id": "fr24-abcdef"}

    aero_key, aero_rotation = build_flight_keys(aero)
    fr24_key, fr24_rotation = build_flight_keys(fr24)

    assert aero_key == fr24_key
    assert aero_rotation == fr24_rotation
    assert aero_key.startswith("flight-key-v1:")


def test_flight_key_normalizes_service_date_representation() -> None:
    string_date = _flight_row()
    timestamp_date = {**string_date, "fl_date": pd.Timestamp("2026-08-11")}

    assert build_flight_keys(string_date) == build_flight_keys(timestamp_date)


def test_prediction_snapshot_preserves_exact_raw_and_model_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOURCE_COMMIT", "commit-test-123")
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "execution-must-not-be-the-version")
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()], index=[7])
    raw = pd.DataFrame(
        {
            "OP_CARRIER": pd.Categorical(["DL"], categories=["AA", "DL"]),
            "TAIL_DELAY_DECAY": [np.nan],
        },
        index=[7],
    )
    model = pd.DataFrame(
        {
            "OP_CARRIER": pd.Categorical(["DL"], categories=["AA", "DL"]),
            "TAIL_DELAY_DECAY": [0.42],
        },
        index=[7],
    )
    artifact = tmp_path / "4year_test"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")

    inserted = enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=raw,
        model_features=model,
        target_indices=[7],
        contexts={
            7: {
                "prediction_phase": "PRE_DEPARTURE",
                "booster_probability": 0.1,
                "calibrated_probability": 0.2,
                "final_probability": 0.3,
                "fallback_applied": True,
            }
        },
        predicted_at_utc="2026-08-11T10:00:00+00:00",
        run_id=99,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["OP_CARRIER", "TAIL_DELAY_DECAY"],
        cat_cols=["OP_CARRIER"],
        cat_mapping={"OP_CARRIER": ["AA", "DL"]},
    )

    assert inserted == 1
    payload = json.loads(
        conn.execute("SELECT payload_json FROM training_export_outbox").fetchone()[0]
    )
    assert payload["raw__OP_CARRIER"] == "DL"
    assert payload["model__OP_CARRIER"] == "DL"
    assert payload["raw__TAIL_DELAY_DECAY"] is None
    assert payload["model__TAIL_DELAY_DECAY"] == 0.42
    assert payload["raw_missing_count"] == 1
    assert payload["model_missing_count"] == 0
    assert payload["booster_probability"] == 0.1
    assert payload["calibrated_probability"] == 0.2
    assert payload["final_probability"] == 0.3
    assert payload["lead_to_scheduled_out_min"] == 120.0
    assert json.loads(payload["feature_order_json"]) == [
        "OP_CARRIER", "TAIL_DELAY_DECAY",
    ]
    assert json.loads(payload["categorical_features_json"]) == ["OP_CARRIER"]
    assert payload["imputed__TAIL_DELAY_DECAY"] is True
    assert payload["causal_validation_status"] == "pass_known_sources_partial"
    assert payload["transformer_version"] == (
        f"code-sha256:{transformer_code_sha256()}"
    )
    assert payload["source_commit"] == "commit-test-123"
    conn.close()


def test_prediction_and_snapshot_share_one_rollback_boundary(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    ensure_training_store_schema(conn)
    conn.commit()
    row = _flight_row()
    conn.execute(
        """INSERT INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay)
           VALUES (?,?,?,?,?)""",
        (row["fa_flight_id"], row["stable_id"], "2026-08-11T10:00:00Z", 0.2, 0),
    )
    flights = pd.DataFrame([row])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=features,
        model_features=features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00Z",
        run_id=1,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )

    conn.rollback()

    assert conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM training_export_outbox").fetchone()[0] == 0
    conn.close()


def test_snapshot_enqueue_batch_rolls_back_after_mid_batch_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_db(tmp_path / "live.db")
    ensure_training_store_schema(conn)
    conn.commit()
    row = _flight_row()
    conn.execute(
        """INSERT INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay)
           VALUES (?,?,?,?,?)""",
        (row["fa_flight_id"], row["stable_id"], "2026-08-11T10:00:00Z", 0.2, 0),
    )
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    original_insert = training_store_module._insert_outbox_event
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second-row failure")
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(training_store_module, "_insert_outbox_event", fail_on_second)

    with pytest.raises(RuntimeError, match="second-row"):
        enqueue_prediction_snapshots(
            conn,
            flight_frame=pd.DataFrame([row, row]),
            raw_features=pd.DataFrame({"A": [1.0, 2.0]}),
            model_features=pd.DataFrame({"A": [1.0, 2.0]}),
            target_indices=[0, 1],
            contexts={
                0: {"prediction_phase": "PRE_DEPARTURE"},
                1: {"prediction_phase": "PRE_DEPARTURE"},
            },
            predicted_at_utc="2026-08-11T10:00:00Z",
            run_id=1,
            data_source="harvester",
            artifact_dir=artifact,
            feature_cols=["A"],
            cat_cols=[],
            cat_mapping={},
        )

    assert conn.execute("SELECT count(*) FROM predictions").fetchone()[0] == 1
    assert conn.execute(
        "SELECT count(*) FROM training_export_outbox",
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM training_feature_contracts",
    ).fetchone()[0] == 0
    conn.close()


def test_outcome_revisions_are_enqueued_idempotently(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    row = _flight_row()
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], row["op_carrier"],
            row["flight_number"], row["tail_num"], row["origin"], row["dest"],
            row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
            row["scheduled_on_utc"], row["scheduled_in_utc"],
            "2026-08-11T09:00:00+00:00", "2026-08-11T14:00:00+00:00",
        ),
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, source_provider, actual_off_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], "aeroapi",
            "2026-08-11T12:15:00+00:00", "2026-08-11T13:40:00+00:00",
            20.0, 5.0, 0, 0, "2026-08-11T13:45:00+00:00",
        ),
    )
    conn.commit()

    first = enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:00:00+00:00", lookback_days=7,
    )
    conn.execute(
        "UPDATE actuals SET settled_at_utc=? WHERE fa_flight_id=?",
        ("2026-08-11T14:04:00+00:00", row["fa_flight_id"]),
    )
    second = enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:05:00+00:00", lookback_days=7,
    )

    assert first == 1
    assert second == 0
    payload = json.loads(
        conn.execute("SELECT payload_json FROM training_export_outbox").fetchone()[0]
    )
    assert payload["arr_delay_min"] == 20.0
    assert payload["source_provider"] == "AEROAPI"
    assert payload["is_final"] is True
    assert payload["source_settled_at_utc"] == "2026-08-11T13:45:00+00:00"

    conn.execute(
        "UPDATE actuals SET arr_delay_min=?, settled_at_utc=? WHERE fa_flight_id=?",
        (40.0, "2026-08-11T14:06:00+00:00", row["fa_flight_id"]),
    )
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:07:00+00:00", lookback_days=7,
    ) == 1
    conn.execute(
        "UPDATE actuals SET arr_delay_min=?, settled_at_utc=? WHERE fa_flight_id=?",
        (20.0, "2026-08-11T14:08:00+00:00", row["fa_flight_id"]),
    )
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:09:00+00:00", lookback_days=7,
    ) == 1
    revisions = [
        json.loads(value[0])
        for value in conn.execute(
            "SELECT payload_json FROM training_export_outbox "
            "WHERE event_type='flight_outcomes' ORDER BY created_at_utc",
        ).fetchall()
    ]
    assert [item["outcome_revision"] for item in revisions] == [1, 2, 3]
    assert [item["arr_delay_min"] for item in revisions] == [20.0, 40.0, 20.0]
    conn.close()


def test_outcome_partition_date_is_immutable_across_schedule_revision(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    row = _flight_row("schedule-revision")
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], row["op_carrier"],
            row["flight_number"], row["tail_num"], row["origin"], row["dest"],
            row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
            row["scheduled_on_utc"], row["scheduled_in_utc"],
            "2026-08-11T09:00:00Z", "2026-08-11T14:00:00Z",
        ),
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, source_provider, actual_off_utc,
            actual_in_utc, arr_delay_min, departure_delay_min, cancelled,
            diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], "aeroapi",
            "2026-08-11T12:15:00Z", "2026-08-11T13:40:00Z",
            20.0, 5.0, 0, 0, "2026-08-11T13:45:00Z",
        ),
    )
    conn.commit()
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:00:00Z",
    ) == 1

    conn.execute(
        """UPDATE flights
           SET fl_date='2026-08-20',
               scheduled_out_utc='2026-08-20T12:00:00Z',
               scheduled_off_utc='2026-08-20T12:10:00Z',
               scheduled_on_utc='2026-08-20T13:10:00Z',
               scheduled_in_utc='2026-08-20T13:20:00Z',
               last_updated_utc='2026-08-20T14:00:00Z'
           WHERE fa_flight_id=?""",
        (row["fa_flight_id"],),
    )
    conn.execute(
        """UPDATE actuals
           SET actual_in_utc=NULL, arr_delay_min=NULL, cancelled=1,
               settled_at_utc='2026-08-20T14:00:00Z'
           WHERE fa_flight_id=?""",
        (row["fa_flight_id"],),
    )
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-20T14:05:00Z",
    ) == 1

    payloads = [
        json.loads(value[0])
        for value in conn.execute(
            "SELECT payload_json FROM training_export_outbox "
            "WHERE event_type='flight_outcomes' ORDER BY created_at_utc",
        ).fetchall()
    ]
    assert [item["flight_date_local"] for item in payloads] == [
        "2026-08-11", "2026-08-20",
    ]
    assert [item["partition_flight_date_local"] for item in payloads] == [
        "2026-08-11", "2026-08-11",
    ]

    client = FakeStorageClient()
    summary = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=client,
    )
    assert summary.events == 2
    assert summary.objects == 2
    assert all(
        "/flight_date=2026-08-11/" in name
        for name in summary.object_names
    )
    assert all("/flight_date=2026-08-20/" not in name for name in summary.object_names)

    for name, body in client.objects.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _, loaded_outcomes, _, _ = materialize_cli.load_local_events(
        tmp_path,
        start=date(2026, 8, 11),
        end=date(2026, 8, 11),
        outcome_lag_days=0,
    )
    assert len(loaded_outcomes) == 2
    assert set(loaded_outcomes["outcome_revision"]) == {1, 2}
    conn.close()


def test_outcome_enqueue_batch_is_atomic_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = open_db(tmp_path / "live.db")
    ensure_training_store_schema(conn)
    for index in range(3):
        row = {
            **_flight_row(f"bounded-{index}"),
            "stable_id": f"stable-{index}",
            "flight_number": str(100 + index),
            "tail_num": f"N10{index}DL",
        }
        conn.execute(
            """INSERT INTO flights
               (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
                origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
                scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row["fa_flight_id"], row["stable_id"], row["op_carrier"],
                row["flight_number"], row["tail_num"], row["origin"], row["dest"],
                row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
                row["scheduled_on_utc"], row["scheduled_in_utc"],
                "2026-08-11T09:00:00Z", "2026-08-11T13:00:00Z",
            ),
        )
        conn.execute(
            """INSERT INTO actuals
               (fa_flight_id, stable_id, source_provider, actual_off_utc,
                actual_in_utc, arr_delay_min, departure_delay_min, cancelled,
                diverted, settled_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                row["fa_flight_id"], row["stable_id"], "aeroapi",
                "2026-08-11T12:15:00Z", "2026-08-11T13:40:00Z",
                20.0 + index, 5.0, 0, 0,
                f"2026-08-11T13:4{index}:00Z",
            ),
        )
    conn.commit()

    # A bounded bootstrap makes forward progress instead of repeatedly
    # selecting already-watermarked recent rows.
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:00:00Z", max_events=2,
    ) == 2
    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:01:00Z", max_events=2,
    ) == 1
    assert conn.execute(
        "SELECT count(*) FROM training_outcome_state",
    ).fetchone()[0] == 3
    conn.execute("DELETE FROM training_export_outbox")
    conn.execute("DELETE FROM training_outcome_state")
    conn.commit()

    original_insert = training_store_module._insert_outbox_event
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic outcome failure")
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(training_store_module, "_insert_outbox_event", fail_on_second)
    with pytest.raises(RuntimeError, match="outcome failure"):
        enqueue_recent_outcomes(
            conn, observed_at_utc="2026-08-11T14:02:00Z", max_events=3,
        )

    assert conn.execute(
        "SELECT count(*) FROM training_export_outbox",
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM training_outcome_state",
    ).fetchone()[0] == 0
    conn.close()


def test_outcome_state_schema_migrates_source_fa_identity(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    conn.execute(
        """CREATE TABLE training_outcome_state (
           source_record_id TEXT PRIMARY KEY,
           state_sha256 TEXT NOT NULL,
           outcome_revision INTEGER NOT NULL,
           last_observed_at_utc TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE training_export_delivery (
           event_id TEXT PRIMARY KEY,
           delivered_at_utc TEXT NOT NULL,
           object_name TEXT NOT NULL,
           content_sha256 TEXT NOT NULL
           )"""
    )

    ensure_training_store_schema(conn)

    columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(training_outcome_state)",
        ).fetchall()
    }
    indexes = {
        row[1] for row in conn.execute(
            "PRAGMA index_list(training_outcome_state)",
        ).fetchall()
    }
    assert "source_fa_flight_id" in columns
    assert "partition_flight_date" in columns
    assert "idx_training_outcome_state_source_fa" in indexes
    delivery_columns = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(training_export_delivery)",
        ).fetchall()
    }
    assert "payload_sha256" in delivery_columns
    conn.close()


def test_outcome_partition_migration_prefers_oldest_pending_revision(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    conn.execute(
        """CREATE TABLE training_outcome_state (
           source_record_id TEXT PRIMARY KEY,
           source_fa_flight_id TEXT,
           state_sha256 TEXT NOT NULL,
           outcome_revision INTEGER NOT NULL,
           last_observed_at_utc TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE training_export_outbox (
           event_id TEXT PRIMARY KEY,
           event_type TEXT NOT NULL,
           event_date TEXT NOT NULL,
           schema_version TEXT NOT NULL,
           created_at_utc TEXT NOT NULL,
           payload_json TEXT NOT NULL
           )"""
    )
    conn.execute(
        """INSERT INTO training_outcome_state
           (source_record_id, source_fa_flight_id, state_sha256,
            outcome_revision, last_observed_at_utc)
           VALUES ('source-old', 'same-flight', 'hash', 1,
                   '2026-08-11T14:00:00Z')"""
    )
    conn.execute(
        """INSERT INTO training_export_outbox
           (event_id, event_type, event_date, schema_version,
            created_at_utc, payload_json)
           VALUES (?,?,?,?,?,?)""",
        (
            "old-event", "flight_outcomes", "2026-08-11",
            TRAINING_SCHEMA_VERSION, "2026-08-11T14:00:00Z",
            json.dumps(
                {
                    "source_record_id": "source-old",
                    "flight_date_local": "2026-08-11",
                }
            ),
        ),
    )
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, fl_date, first_seen_utc, last_updated_utc)
           VALUES ('same-flight', 'same-stable', '2026-08-20',
                   '2026-08-20T09:00:00Z', '2026-08-20T10:00:00Z')"""
    )

    ensure_training_store_schema(conn)

    assert conn.execute(
        "SELECT partition_flight_date FROM training_outcome_state "
        "WHERE source_record_id='source-old'",
    ).fetchone() == ("2026-08-11",)
    conn.execute("DELETE FROM training_export_outbox")
    assert publish_pending_outbox(
        conn,
        bucket_name="unused-for-empty-outbox",
        mark_delivered=True,
        storage_client=FakeStorageClient(),
    ).events == 0
    conn.close()

    reopened = open_db(tmp_path / "live.db")
    assert reopened.execute(
        "SELECT partition_flight_date FROM training_outcome_state "
        "WHERE source_record_id='source-old'",
    ).fetchone() == ("2026-08-11",)
    reopened.close()


def test_unseen_outcome_is_recovered_after_gap_longer_than_lookback(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    row = _flight_row("old-unseen-flight")
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], row["op_carrier"],
            row["flight_number"], row["tail_num"], row["origin"], row["dest"],
            row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
            row["scheduled_on_utc"], row["scheduled_in_utc"],
            "2026-07-15T09:00:00Z", "2026-07-15T14:00:00Z",
        ),
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, source_provider, actual_off_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], "fr24",
            "2026-07-15T12:15:00Z", "2026-07-15T13:40:00Z",
            20.0, 5.0, 0, 0, "2026-07-15T13:45:00Z",
        ),
    )
    conn.commit()

    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-08-11T14:00:00Z",
        lookback_days=7,
    ) == 1
    state = conn.execute(
        "SELECT source_fa_flight_id, outcome_revision "
        "FROM training_outcome_state",
    ).fetchone()
    assert state == (row["fa_flight_id"], 1)

    client = FakeStorageClient()
    publish_pending_outbox(
        conn,
        bucket_name="training-test",
        mark_delivered=True,
        storage_client=client,
    )
    conn.execute("DELETE FROM training_outcome_state")
    conn.commit()

    # Restore a missing watermark from the matching delivered event.  The event
    # itself must not be emitted again.
    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-08-11T14:00:00Z",
        lookback_days=7,
    ) == 0
    restored = conn.execute(
        "SELECT source_fa_flight_id, outcome_revision "
        "FROM training_outcome_state",
    ).fetchone()
    assert restored == (row["fa_flight_id"], 1)

    # It is no longer recent, but the explicit source identity now records that
    # it was exported, so later recovery scans remain idempotent.
    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-08-11T14:05:00Z",
        lookback_days=7,
    ) == 0
    assert conn.execute(
        "SELECT count(*) FROM training_export_outbox "
        "WHERE event_type='flight_outcomes'",
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM training_export_delivery",
    ).fetchone()[0] == 1
    conn.close()


def test_known_outcome_revision_is_recovered_after_long_gap(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    row = _flight_row("known-before-gap")
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], row["op_carrier"],
            row["flight_number"], row["tail_num"], row["origin"], row["dest"],
            row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
            row["scheduled_on_utc"], row["scheduled_in_utc"],
            "2026-07-15T09:00:00Z", "2026-07-15T14:00:00Z",
        ),
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, source_provider, actual_off_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"], "aeroapi",
            "2026-07-15T12:15:00Z", "2026-07-15T13:40:00Z",
            10.0, 5.0, 0, 0, "2026-07-15T13:45:00Z",
        ),
    )
    conn.commit()
    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-07-16T14:00:00Z",
        lookback_days=7,
    ) == 1

    # The correction settled during an outage and is older than the normal
    # seven-day lookback when the job resumes.  settled > last_observed makes it
    # recoverable without scanning it as a brand-new source.
    conn.execute(
        "UPDATE actuals SET arr_delay_min=?, settled_at_utc=? WHERE fa_flight_id=?",
        (40.0, "2026-07-20T14:00:00Z", row["fa_flight_id"]),
    )
    conn.commit()
    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-08-11T14:00:00Z",
        lookback_days=7,
    ) == 1

    revisions = [
        json.loads(value[0])
        for value in conn.execute(
            "SELECT payload_json FROM training_export_outbox "
            "WHERE event_type='flight_outcomes' ORDER BY created_at_utc",
        ).fetchall()
    ]
    assert [item["outcome_revision"] for item in revisions] == [1, 2]
    assert [item["arr_delay_min"] for item in revisions] == [10.0, 40.0]
    conn.close()


def test_outcome_state_pruning_waits_for_an_empty_outbox(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    ensure_training_store_schema(conn)
    state_row = (
        "old-source", "gone-flight", "state-hash", 1,
        "2026-01-01T00:00:00Z", "2026-01-01",
    )
    conn.execute(
        """INSERT INTO training_outcome_state
           (source_record_id, source_fa_flight_id, state_sha256,
            outcome_revision, last_observed_at_utc, partition_flight_date)
           VALUES (?,?,?,?,?,?)""",
        state_row,
    )
    conn.execute(
        """INSERT INTO training_export_outbox
           (event_id, event_type, event_date, schema_version,
            created_at_utc, payload_json)
           VALUES (?,?,?,?,?,?)""",
        (
            "pending", "prediction_snapshots", "2026-08-11",
            TRAINING_SCHEMA_VERSION, "2026-08-11T10:00:00Z", "{}",
        ),
    )

    assert prune_outcome_state(conn, keep_days=45) == 0
    conn.execute("DELETE FROM training_export_outbox")
    assert prune_outcome_state(conn, keep_days=45) == 1
    assert conn.execute(
        "SELECT count(*) FROM training_outcome_state",
    ).fetchone()[0] == 0
    conn.close()


def test_ambiguous_stable_id_does_not_borrow_flight_identity(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    base = _flight_row("physical-one")
    rows = [
        {**base, "fa_flight_id": "physical-one", "stable_id": "SAME-STABLE"},
        {
            **base,
            "fa_flight_id": "physical-two",
            "stable_id": "SAME-STABLE",
            "tail_num": "N222AA",
            "dest": "BOS",
            "scheduled_off_utc": "2026-08-11T16:10:00Z",
        },
    ]
    conn.executemany(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                row["fa_flight_id"], row["stable_id"], row["op_carrier"],
                row["flight_number"], row["tail_num"], row["origin"], row["dest"],
                row["fl_date"], row["scheduled_out_utc"], row["scheduled_off_utc"],
                row["scheduled_on_utc"], row["scheduled_in_utc"],
                "2026-08-11T09:00:00Z", "2026-08-11T10:00:00Z",
            )
            for row in rows
        ],
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, actual_off_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            "actual-without-exact-flight", "SAME-STABLE", "2026-08-11T12:15:00Z",
            "2026-08-11T13:40:00Z", 20.0, 5.0, 0, 0, "2026-08-11T14:00:00Z",
        ),
    )
    conn.commit()

    assert enqueue_recent_outcomes(
        conn, observed_at_utc="2026-08-11T14:05:00Z",
    ) == 1
    payload = json.loads(
        conn.execute(
            "SELECT payload_json FROM training_export_outbox "
            "WHERE event_type='flight_outcomes'",
        ).fetchone()[0]
    )

    assert payload["tail_num"] is None
    assert payload["rotation_key"] is None
    assert payload["destination"] is None
    conn.close()


def test_publish_is_create_only_and_acknowledged_on_next_cycle(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=features,
        model_features=features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00+00:00",
        run_id=1,
        data_source="aeroapi",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )
    conn.commit()
    client = FakeStorageClient()

    first = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=client,
    )
    assert first.events == 1
    assert pending_outbox_count(conn) == 1
    assert len(client.objects) == 2  # feature contract + Parquet shard

    second = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=True,
        storage_client=client,
    )
    assert second.events == 1
    assert pending_outbox_count(conn) == 0
    assert conn.execute("SELECT count(*) FROM training_export_delivery").fetchone()[0] == 1
    assert len(client.objects) == 2
    parquet_body = next(
        body for name, body in client.objects.items() if name.endswith(".parquet")
    )
    parquet = pd.read_parquet(BytesIO(parquet_body))
    assert parquet.loc[0, "model__A"] == 1.0
    conn.close()


def test_publisher_keeps_unacknowledged_batches_in_separate_shards(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")

    def enqueue(predicted_at_utc: str, run_id: int) -> None:
        enqueue_prediction_snapshots(
            conn,
            flight_frame=flights,
            raw_features=features,
            model_features=features,
            target_indices=[0],
            contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
            predicted_at_utc=predicted_at_utc,
            run_id=run_id,
            data_source="aeroapi",
            artifact_dir=artifact,
            feature_cols=["A"],
            cat_cols=[],
            cat_mapping={},
        )
        conn.commit()

    client = FakeStorageClient()
    enqueue("2026-08-11T10:00:00Z", 1)
    first = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=client,
    )
    first_event_id = pd.read_parquet(
        BytesIO(client.objects[first.object_names[0]]),
    ).loc[0, "event_id"]

    # Model a published A whose acknowledgement did not reach the durable DB,
    # followed by a new cycle B entering the same outbox.
    enqueue("2026-08-11T10:15:00Z", 2)
    second = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=client,
    )

    parquet_frames = [
        pd.read_parquet(BytesIO(body))
        for name, body in client.objects.items()
        if name.endswith(".parquet")
    ]
    all_event_ids = [
        event_id
        for frame in parquet_frames
        for event_id in frame["event_id"].tolist()
    ]
    assert second.events == 2
    assert len(parquet_frames) == 2
    assert all_event_ids.count(first_event_id) == 1
    assert len(set(all_event_ids)) == 2
    assert pending_outbox_count(conn) == 2
    conn.close()


def test_publisher_drains_only_bounded_oldest_created_at_groups(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    for run_id, minute in enumerate((0, 15, 30), start=1):
        enqueue_prediction_snapshots(
            conn,
            flight_frame=flights,
            raw_features=features,
            model_features=features,
            target_indices=[0],
            contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
            predicted_at_utc=f"2026-08-11T10:{minute:02d}:00Z",
            run_id=run_id,
            data_source="harvester",
            artifact_dir=artifact,
            feature_cols=["A"],
            cat_cols=[],
            cat_mapping={},
        )
    conn.commit()
    client = FakeStorageClient()

    summaries = [
        publish_pending_outbox(
            conn,
            bucket_name="training-test",
            prefix="ledger",
            mark_delivered=True,
            storage_client=client,
            max_created_at_groups=1,
        )
        for _ in range(3)
    ]

    assert [summary.events for summary in summaries] == [1, 1, 1]
    assert pending_outbox_count(conn) == 0
    parquet_names = sorted(
        name for name in client.objects if name.endswith(".parquet")
    )
    assert len(parquet_names) == 3
    delivered_ids = {
        event_id
        for name in parquet_names
        for event_id in pd.read_parquet(BytesIO(client.objects[name]))["event_id"]
    }
    assert len(delivered_ids) == 3
    conn.close()


def test_publisher_splits_incompatible_feature_contracts(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    first_artifact = tmp_path / "first"
    second_artifact = tmp_path / "second"
    for artifact, model_bytes in (
        (first_artifact, b"model-one"), (second_artifact, b"model-two"),
    ):
        artifact.mkdir()
        (artifact / "model.lgb").write_bytes(model_bytes)
        (artifact / "meta.joblib").write_bytes(b"meta")

    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=pd.DataFrame({"A": [1.0]}),
        model_features=pd.DataFrame({"A": [1.0]}),
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00Z",
        run_id=1,
        data_source="harvester",
        artifact_dir=first_artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )
    second_features = pd.DataFrame(
        {"A": [1.0], "NEW_CAT": pd.Categorical(["DL"], categories=["DL", "AA"])}
    )
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=second_features,
        model_features=second_features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:01:00Z",
        run_id=2,
        data_source="harvester",
        artifact_dir=second_artifact,
        feature_cols=["A", "NEW_CAT"],
        cat_cols=["NEW_CAT"],
        cat_mapping={"NEW_CAT": ["DL", "AA"]},
    )
    conn.commit()
    client = FakeStorageClient()

    summary = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=client,
    )

    assert summary.events == 2
    assert summary.objects == 2
    frames = [
        pd.read_parquet(BytesIO(body))
        for name, body in client.objects.items()
        if name.endswith(".parquet")
    ]
    categorical_frame = next(frame for frame in frames if "model__NEW_CAT" in frame)
    assert categorical_frame.loc[0, "model__NEW_CAT"] == "DL"
    conn.close()


def test_publisher_does_not_ack_metadata_collision(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=features,
        model_features=features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00Z",
        run_id=1,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )
    conn.commit()
    client = FakeStorageClient()
    first = publish_pending_outbox(
        conn, bucket_name="training-test", mark_delivered=False,
        storage_client=client,
    )
    client.metadata_by_name[first.object_names[0]]["event_count"] = "999"

    try:
        publish_pending_outbox(
            conn, bucket_name="training-test", mark_delivered=True,
            storage_client=client,
        )
    except RuntimeError as exc:
        assert "immutable training object collision" in str(exc)
    else:
        raise AssertionError("metadata collision must fail closed")

    assert pending_outbox_count(conn) == 1
    conn.close()


def test_publisher_does_not_ack_matching_metadata_with_corrupt_body(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=features,
        model_features=features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00Z",
        run_id=1,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )
    conn.commit()
    client = FakeStorageClient()
    first = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        mark_delivered=False,
        storage_client=client,
    )
    object_name = first.object_names[0]
    client.objects[object_name] = b"corrupt-but-metadata-is-unchanged"

    try:
        publish_pending_outbox(
            conn,
            bucket_name="training-test",
            mark_delivered=True,
            storage_client=client,
        )
    except RuntimeError as exc:
        assert "content SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("corrupt immutable object must fail closed")

    assert pending_outbox_count(conn) == 1
    assert conn.execute(
        "SELECT count(*) FROM training_export_delivery",
    ).fetchone()[0] == 0
    conn.close()


def test_publisher_accepts_semantically_identical_parquet_after_encoder_change(
    tmp_path: Path,
) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=features,
        model_features=features,
        target_indices=[0],
        contexts={0: {"prediction_phase": "PRE_DEPARTURE"}},
        predicted_at_utc="2026-08-11T10:00:00Z",
        run_id=1,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    )
    conn.commit()
    client = FakeStorageClient()
    first = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        mark_delivered=False,
        storage_client=client,
    )
    object_name = first.object_names[0]
    original = client.objects[object_name]
    frame = pd.read_parquet(BytesIO(original))
    alternate_buffer = BytesIO()
    frame.to_parquet(alternate_buffer, index=False, compression=None)
    alternate = alternate_buffer.getvalue()
    assert alternate != original
    client.objects[object_name] = alternate

    publish_pending_outbox(
        conn,
        bucket_name="training-test",
        mark_delivered=True,
        storage_client=client,
    )

    assert pending_outbox_count(conn) == 0
    conn.close()


def test_delivered_event_id_rejects_a_different_payload(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "live.db")
    flights = pd.DataFrame([_flight_row()])
    first_features = pd.DataFrame({"A": [1.0]})
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    common = {
        "flight_frame": flights,
        "target_indices": [0],
        "contexts": {0: {"prediction_phase": "PRE_DEPARTURE"}},
        "predicted_at_utc": "2026-08-11T10:00:00Z",
        "run_id": 1,
        "data_source": "harvester",
        "artifact_dir": artifact,
        "feature_cols": ["A"],
        "cat_cols": [],
        "cat_mapping": {},
    }
    enqueue_prediction_snapshots(
        conn,
        raw_features=first_features,
        model_features=first_features,
        **common,
    )
    conn.commit()
    publish_pending_outbox(
        conn,
        bucket_name="training-test",
        mark_delivered=True,
        storage_client=FakeStorageClient(),
    )
    delivered_hash = conn.execute(
        "SELECT payload_sha256 FROM training_export_delivery",
    ).fetchone()[0]
    assert delivered_hash

    changed_features = pd.DataFrame({"A": [2.0]})
    try:
        enqueue_prediction_snapshots(
            conn,
            raw_features=changed_features,
            model_features=changed_features,
            **common,
        )
    except ValueError as exc:
        assert "event_id collision with different payload" in str(exc)
    else:
        raise AssertionError("delivered event_id payload collision must fail closed")

    assert pending_outbox_count(conn) == 0
    conn.close()


def _seal_event(payload: dict) -> dict:
    payload.pop("payload_sha256", None)
    payload.pop("payload_keys_json", None)
    keys = sorted(payload)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    payload["payload_keys_json"] = json.dumps(keys, separators=(",", ":"))
    payload["payload_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _snapshot_event(
    event_id: str,
    key: str,
    predicted_at: str,
    scheduled_out: str,
    feature_value: float,
    probability: float,
) -> dict:
    category_mapping_json = "{}"
    return _seal_event({
        "event_id": event_id,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "canonical_flight_key": key,
        "rotation_key": None,
        "fa_flight_id": key,
        "stable_id": key,
        "prediction_phase": "PRE_DEPARTURE",
        "predicted_at_utc": predicted_at,
        "scheduled_out_utc": scheduled_out,
        "model_version": "v-test",
        "artifact_model_sha256": "model-hash",
        "artifact_meta_sha256": "meta-hash",
        "feature_schema_hash": "schema-hash",
        "category_mapping_sha256": sha256(
            category_mapping_json.encode("utf-8"),
        ).hexdigest(),
        "category_mapping_json": category_mapping_json,
        "feature_order_json": '["Z","A"]',
        "categorical_features_json": "[]",
        "transformer_version": "commit-test",
        "target_definition": "arr_delay_min > 15",
        "model_missing_count": 0,
        "causal_validation_status": "pass_known_sources_partial",
        "provenance_completeness": "partial-v1",
        "final_probability": probability,
        "booster_probability": probability,
        "calibrated_probability": probability,
        "threshold_used": 0.5,
        "threshold_strategy": "test",
        "model__Z": feature_value + 10.0,
        "model__A": feature_value,
        "raw__Z": feature_value + 10.0,
        "raw__A": feature_value,
        "imputed__Z": False,
        "imputed__A": False,
    })


def _outcome_event(
    event_id: str,
    key: str,
    *,
    observed_at: str = "2026-08-11T14:00:00Z",
    delay: float | None = 20.0,
    cancelled: int = 0,
    diverted: int = 0,
    rotation_key: str | None = None,
    source_provider: str | None = "AEROAPI",
) -> dict:
    return _seal_event({
        "event_id": event_id,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "canonical_flight_key": key,
        "rotation_key": rotation_key,
        "fa_flight_id": key,
        "stable_id": key,
        "source_provider": source_provider,
        "source_record_id": f"source-{event_id}",
        "outcome_revision": 1,
        "outcome_state_sha256": f"state-{event_id}",
        "is_final": True,
        "arr_delay_min": delay,
        "cancelled": cancelled,
        "diverted": diverted,
        "observed_at_utc": observed_at,
        "actual_off_utc": "2026-08-11T12:05:00Z",
        "actual_in_utc": "2026-08-11T13:20:00Z",
    })


def test_materializer_selects_latest_snapshot_before_scheduled_cutoff() -> None:
    snapshots = pd.DataFrame(
        [
            _snapshot_event("old", "flight-1", "2026-08-11T09:00:00Z", "2026-08-11T12:00:00Z", 1.0, 0.1),
            _snapshot_event("valid", "flight-1", "2026-08-11T09:45:00Z", "2026-08-11T12:00:00Z", 2.0, 0.2),
            # Higher score, but it is after the T-2h cutoff and must never win.
            _snapshot_event("future", "flight-1", "2026-08-11T10:05:00Z", "2026-08-11T12:00:00Z", 99.0, 0.99),
            _snapshot_event("edge", "flight-2", "2026-08-11T09:50:00Z", "2026-08-11T12:00:00Z", 3.0, 0.3),
        ]
    )
    outcomes = pd.DataFrame(
        [
            _outcome_event("out-1", "flight-1", delay=15.0),
            _outcome_event("out-2", "flight-2", delay=15.1),
        ]
    )

    result, report = materialize_fixed_horizons(
        snapshots,
        outcomes,
        horizons_hours=[2.0],
        max_snapshot_age_minutes=60,
    )

    assert report["rows"] == 2
    by_key = result.set_index("canonical_flight_key")
    assert by_key.loc["flight-1", "event_id"] == "valid"
    assert by_key.loc["flight-1", "A"] == 2.0
    assert by_key.loc["flight-1", "TARGET"] == 0
    assert by_key.loc["flight-1", "outcome_source_provider"] == "AEROAPI"
    assert by_key.loc["flight-2", "TARGET"] == 1
    assert [col for col in result.columns if col in {"Z", "A"}] == ["Z", "A"]

    X, y, contract = prepare_silver_training_data(
        result, feature_contract=report["feature_contract"],
    )
    assert list(X.columns) == ["Z", "A"]
    assert X.loc[0, "Z"] == result.loc[0, "Z"]
    assert y.tolist() == result["TARGET"].tolist()
    assert contract["feature_order"] == ["Z", "A"]
    raw_X, _, raw_contract = prepare_silver_training_data(
        result,
        representation="raw_challenger",
        feature_contract=report["feature_contract"],
    )
    assert list(raw_X.columns) == ["Z", "A"]
    assert raw_contract["requires_train_only_category_fit"] is True


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("model_version", "v-other"),
        ("artifact_model_sha256", "other-model-hash"),
        ("artifact_meta_sha256", "other-meta-hash"),
        ("transformer_version", "code-sha256:other-transformer"),
    ],
)
def test_materializer_rejects_mixed_model_or_transformer_deployments(
    field: str,
    different_value: str,
) -> None:
    first = _snapshot_event(
        "snap-a", "flight-a", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    second = _snapshot_event(
        "snap-b", "flight-b", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 2.0, 0.3,
    )
    second[field] = different_value
    _seal_event(second)
    outcomes = pd.DataFrame(
        [
            _outcome_event("out-a", "flight-a"),
            _outcome_event("out-b", "flight-b"),
        ]
    )

    with pytest.raises(ValueError, match="mixes 2 model/transformer deployments"):
        materialize_fixed_horizons(
            pd.DataFrame([first, second]),
            outcomes,
        )


def test_silver_adapter_rejects_mixed_deployment_identity() -> None:
    silver, report = materialize_fixed_horizons(
        pd.DataFrame(
            [
                _snapshot_event(
                    "snap", "flight-1", "2026-08-11T09:45:00Z",
                    "2026-08-11T12:00:00Z", 1.0, 0.2,
                )
            ]
        ),
        pd.DataFrame([_outcome_event("out", "flight-1")]),
    )
    other = silver.copy()
    other["artifact_meta_sha256"] = "other-meta-hash"
    mixed = pd.concat([silver, other], ignore_index=True)

    with pytest.raises(ValueError, match="mixes 2 model/transformer deployments"):
        prepare_silver_training_data(
            mixed,
            feature_contract=report["feature_contract"],
        )


def test_materializer_uses_latest_outcome_before_eligibility_filter() -> None:
    snapshots = pd.DataFrame(
        [_snapshot_event("snap", "flight-1", "2026-08-11T09:45:00Z", "2026-08-11T12:00:00Z", 1.0, 0.2)]
    )
    outcomes = pd.DataFrame(
        [
            _outcome_event("normal", "flight-1", delay=30.0),
            _outcome_event(
                "cancelled", "flight-1", observed_at="2026-08-11T15:00:00Z",
                delay=None, cancelled=1,
            ),
        ]
    )

    result, report = materialize_fixed_horizons(snapshots, outcomes)

    assert result.empty
    assert report["eligible_outcomes"] == 0


def test_materializer_returns_empty_when_no_snapshot_is_near_cutoff() -> None:
    snapshots = pd.DataFrame(
        [_snapshot_event("old", "flight-1", "2026-08-11T07:00:00Z", "2026-08-11T12:00:00Z", 1.0, 0.2)]
    )
    outcomes = pd.DataFrame([_outcome_event("out", "flight-1")])

    result, report = materialize_fixed_horizons(
        snapshots, outcomes, horizons_hours=[2.0], max_snapshot_age_minutes=60,
    )

    assert result.empty
    assert report["horizon_counts"] == {"2.0": 0}


def test_materializer_prefers_canonical_identity_over_changed_tail() -> None:
    snapshot = _snapshot_event(
        "snap", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot["rotation_key"] = "old-tail"
    outcome = _outcome_event("out", "flight-1", rotation_key="new-tail")
    _seal_event(snapshot)

    result, _ = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([outcome]),
    )

    assert len(result) == 1
    assert result.loc[0, "canonical_flight_key"] == "flight-1"


def test_materializer_uses_exact_source_id_across_schedule_revision() -> None:
    snapshot = _snapshot_event(
        "snap", "old-schedule-key", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot.update(
        fa_flight_id="same-source-flight", stable_id="same-stable",
        rotation_key="old-rotation",
    )
    outcome = _outcome_event(
        "out", "new-schedule-key", rotation_key="new-rotation",
    )
    outcome.update(fa_flight_id="same-source-flight", stable_id="same-stable")
    _seal_event(snapshot)
    _seal_event(outcome)

    result, _ = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([outcome]),
    )

    assert len(result) == 1
    assert result.loc[0, "_training_flight_key"] == "new-schedule-key"


def test_materializer_tie_break_is_deterministic_and_prefers_fewer_missing() -> None:
    better = _snapshot_event(
        "a-better", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    worse = _snapshot_event(
        "z-worse", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 9.0, 0.9,
    )
    worse["model_missing_count"] = 1
    _seal_event(worse)
    outcome = pd.DataFrame([_outcome_event("out", "flight-1")])

    first, _ = materialize_fixed_horizons(
        pd.DataFrame([better, worse]), outcome,
    )
    second, _ = materialize_fixed_horizons(
        pd.DataFrame([worse, better]), outcome,
    )

    assert first.loc[0, "event_id"] == "a-better"
    assert second.loc[0, "event_id"] == "a-better"


def test_materializer_excludes_known_future_source() -> None:
    snapshot = _snapshot_event(
        "future-source", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot["causal_validation_status"] = "failed_future_source"
    _seal_event(snapshot)

    result, _ = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([_outcome_event("out", "flight-1")]),
    )

    assert result.empty


def test_materializer_quarantines_conflicting_canonicals_for_same_rotation() -> None:
    snapshot = _snapshot_event(
        "snap", "provider-a-canonical", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.8,
    )
    snapshot["rotation_key"] = "same-tail-scheduled-off-minute"
    _seal_event(snapshot)
    first = _outcome_event(
        "out-a", "provider-a-canonical", delay=30.0,
        rotation_key="same-tail-scheduled-off-minute", source_provider="AEROAPI",
    )
    second = _outcome_event(
        "out-b", "provider-b-canonical", delay=5.0,
        observed_at="2026-08-12T14:00:00Z",
        rotation_key="same-tail-scheduled-off-minute", source_provider="FR24",
    )

    result, report = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([first, second]),
    )

    assert result.empty
    assert report["conflicting_outcomes"] == 1
    assert report["conflicting_outcome_provider_counts"] == {
        "AEROAPI": 1,
        "FR24": 1,
    }


def test_materializer_never_uses_ambiguous_stable_id_as_identity() -> None:
    snapshot = _snapshot_event(
        "snap", "snapshot-canonical", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot.update(fa_flight_id="snapshot-source", stable_id="reused-stable")
    _seal_event(snapshot)
    first = _outcome_event("out-a", "outcome-a", delay=5.0)
    first.update(fa_flight_id="outcome-source-a", stable_id="reused-stable")
    second = _outcome_event("out-b", "outcome-b", delay=5.0)
    second.update(fa_flight_id="outcome-source-b", stable_id="reused-stable")
    _seal_event(first)
    _seal_event(second)

    result, report = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([first, second]),
    )

    assert result.empty
    assert report["joined_cycles"] == 0


def test_materializer_latest_source_revision_can_change_identity_and_cancel() -> None:
    snapshot = _snapshot_event(
        "snap", "old-canonical", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.8,
    )
    snapshot.update(fa_flight_id="same-source-flight", rotation_key="old-rotation")
    _seal_event(snapshot)
    normal = _outcome_event(
        "normal", "old-canonical", delay=30.0, rotation_key="old-rotation",
    )
    normal.update(
        fa_flight_id="same-source-flight",
        source_record_id="same-source-record",
        outcome_revision=1,
    )
    cancelled = _outcome_event(
        "cancelled", "new-canonical", observed_at="2026-08-11T15:00:00Z",
        delay=None, cancelled=1, rotation_key="new-rotation",
    )
    cancelled.update(
        fa_flight_id="same-source-flight",
        source_record_id="same-source-record",
        outcome_revision=2,
    )
    _seal_event(normal)
    _seal_event(cancelled)

    result, report = materialize_fixed_horizons(
        pd.DataFrame([snapshot]), pd.DataFrame([normal, cancelled]),
    )

    assert result.empty
    assert report["eligible_outcomes"] == 0


def test_materializer_rejects_payload_mutation_even_if_declared_hash_is_reused() -> None:
    snapshot = _snapshot_event(
        "snap", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot["model__A"] = 999.0

    with pytest.raises(ValueError, match="payload integrity failure"):
        materialize_fixed_horizons(
            pd.DataFrame([snapshot]),
            pd.DataFrame([_outcome_event("out", "flight-1")]),
        )


@pytest.mark.parametrize("status", [None, "PASS", "unknown-status"])
def test_materializer_fails_closed_for_unknown_causal_status(
    status: str | None,
) -> None:
    snapshot = _snapshot_event(
        "snap", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot["causal_validation_status"] = status
    _seal_event(snapshot)

    result, _ = materialize_fixed_horizons(
        pd.DataFrame([snapshot]),
        pd.DataFrame([_outcome_event("out", "flight-1")]),
    )

    assert result.empty


def test_materializer_rejects_unsupported_target_definition() -> None:
    snapshot = _snapshot_event(
        "snap", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    snapshot["target_definition"] = "arr_delay_min > 30"
    _seal_event(snapshot)

    with pytest.raises(ValueError, match="unsupported or mixed target_definition"):
        materialize_fixed_horizons(
            pd.DataFrame([snapshot]),
            pd.DataFrame([_outcome_event("out", "flight-1")]),
        )


def test_silver_loader_requires_one_horizon_and_split_keeps_flight_whole() -> None:
    snapshot = _snapshot_event(
        "snap", "flight-1", "2026-08-11T09:45:00Z",
        "2026-08-11T12:00:00Z", 1.0, 0.2,
    )
    silver, report = materialize_fixed_horizons(
        pd.DataFrame([snapshot]),
        pd.DataFrame([_outcome_event("out", "flight-1")]),
    )
    second_horizon = silver.copy()
    second_horizon["_horizon_hours"] = 6.0
    mixed = pd.concat([silver, second_horizon], ignore_index=True)

    with pytest.raises(ValueError, match="select exactly one horizon"):
        prepare_silver_training_data(
            mixed,
            feature_contract=report["feature_contract"],
        )

    crossing = mixed.copy()
    crossing.loc[1, "scheduled_out_utc"] = "2026-09-15T12:00:00Z"
    with pytest.raises(ValueError, match="physical flight crosses temporal splits"):
        temporal_split_silver(
            crossing,
            train_end_utc="2026-09-01T00:00:00Z",
            validation_end_utc="2026-10-01T00:00:00Z",
        )
