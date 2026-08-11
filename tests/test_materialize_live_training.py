from __future__ import annotations

from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest

from ontimeai.live import open_db
from ontimeai.training_store import (
    MATERIALIZER_VERSION,
    enqueue_prediction_snapshots,
    enqueue_recent_outcomes,
    materialize_fixed_horizons,
    prepare_silver_training_data,
    publish_pending_outbox,
)
from scripts import materialize_live_training as materialize_cli


class _MemoryBlob:
    def __init__(self, client: "_MemoryStorageClient", name: str) -> None:
        self.client = client
        self.name = name
        self.metadata = dict(client.metadata.get(name, {}))
        self.generation = client.generations.get(name, 1)

    def upload_from_string(
        self,
        body: bytes,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None:
        assert content_type in {"application/json", "application/vnd.apache.parquet"}
        assert if_generation_match == 0
        if self.name in self.client.objects:
            from google.api_core.exceptions import PreconditionFailed

            raise PreconditionFailed("already exists")
        self.client.objects[self.name] = body
        self.client.metadata[self.name] = dict(self.metadata)
        self.client.generations[self.name] = 1
        self.generation = 1

    def download_as_bytes(self, *, if_generation_match: int) -> bytes:
        assert if_generation_match == self.client.generations[self.name]
        return self.client.objects[self.name]

    def reload(self) -> None:
        self.metadata = dict(self.client.metadata.get(self.name, {}))
        self.generation = self.client.generations[self.name]


class _MemoryBucket:
    def __init__(self, client: "_MemoryStorageClient") -> None:
        self.client = client

    def blob(self, name: str) -> _MemoryBlob:
        return _MemoryBlob(self.client, name)


class _MemoryStorageClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.generations: dict[str, int] = {}

    def bucket(self, _name: str) -> _MemoryBucket:
        return _MemoryBucket(self)

    def list_blobs(self, _bucket_name: str, *, prefix: str) -> list[_MemoryBlob]:
        return [
            _MemoryBlob(self, name)
            for name in sorted(self.objects)
            if name.startswith(prefix)
        ]


def _flight_row() -> dict[str, object]:
    return {
        "fa_flight_id": "DAL123-1786449600-airline-test",
        "stable_id": "DAL123-1786449600",
        "op_carrier": "DL",
        "flight_number": "123",
        "tail_num": "N123DL",
        "origin": "ATL",
        "dest": "MCO",
        "fl_date": "2026-08-11",
        "scheduled_out_utc": "2026-08-11T12:00:00+00:00",
        "scheduled_off_utc": "2026-08-11T12:10:00+00:00",
        "scheduled_on_utc": "2026-08-11T13:10:00+00:00",
        "scheduled_in_utc": "2026-08-11T13:20:00+00:00",
        "estimated_out_utc": "2026-08-11T12:05:00+00:00",
        "estimated_in_utc": "2026-08-11T13:25:00+00:00",
        "first_seen_utc": "2026-08-11T08:00:00+00:00",
        "last_updated_utc": "2026-08-11T09:40:00+00:00",
    }


@pytest.fixture
def published_ledger(tmp_path: Path) -> _MemoryStorageClient:
    conn = open_db(tmp_path / "live.db")
    row = _flight_row()
    conn.execute(
        """INSERT INTO flights
           (fa_flight_id, stable_id, op_carrier, flight_number, tail_num,
            origin, dest, fl_date, scheduled_out_utc, scheduled_off_utc,
            scheduled_on_utc, scheduled_in_utc, estimated_out_utc,
            estimated_in_utc, first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        tuple(row[column] for column in (
            "fa_flight_id", "stable_id", "op_carrier", "flight_number",
            "tail_num", "origin", "dest", "fl_date", "scheduled_out_utc",
            "scheduled_off_utc", "scheduled_on_utc", "scheduled_in_utc",
            "estimated_out_utc", "estimated_in_utc", "first_seen_utc",
            "last_updated_utc",
        )),
    )
    conn.execute(
        """INSERT INTO actuals
           (fa_flight_id, stable_id, actual_off_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            row["fa_flight_id"], row["stable_id"],
            "2026-08-11T12:15:00+00:00", "2026-08-11T13:40:00+00:00",
            20.0, 5.0, 0, 0, "2026-08-11T13:45:00+00:00",
        ),
    )
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "model.lgb").write_bytes(b"model")
    (artifact / "meta.joblib").write_bytes(b"meta")
    flights = pd.DataFrame([row])
    raw = pd.DataFrame({"A": [1.5]})
    model = pd.DataFrame({"A": [2.5]})
    assert enqueue_prediction_snapshots(
        conn,
        flight_frame=flights,
        raw_features=raw,
        model_features=model,
        target_indices=[0],
        contexts={
            0: {
                "prediction_phase": "PRE_DEPARTURE",
                "booster_probability": 0.2,
                "calibrated_probability": 0.3,
                "final_probability": 0.4,
                "threshold_used": 0.5,
                "threshold_strategy": "artifact",
                "fallback_applied": True,
            }
        },
        predicted_at_utc="2026-08-11T09:45:00+00:00",
        run_id=1,
        data_source="harvester",
        artifact_dir=artifact,
        feature_cols=["A"],
        cat_cols=[],
        cat_mapping={},
    ) == 1
    assert enqueue_recent_outcomes(
        conn,
        observed_at_utc="2026-08-11T14:00:00+00:00",
        lookback_days=7,
    ) == 1
    conn.commit()

    storage = _MemoryStorageClient()
    summary = publish_pending_outbox(
        conn,
        bucket_name="training-test",
        prefix="ledger",
        mark_delivered=False,
        storage_client=storage,
    )
    conn.close()
    assert summary.events == 2
    assert len(storage.objects) == 3  # one contract + two event shards
    return storage


def _write_ledger(root: Path, storage: _MemoryStorageClient) -> None:
    for name, body in storage.objects.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)


def test_local_sidecar_round_trip_and_manifest(
    tmp_path: Path,
    published_ledger: _MemoryStorageClient,
) -> None:
    _write_ledger(tmp_path, published_ledger)

    snapshots, outcomes, contracts, sources = materialize_cli.load_local_events(
        tmp_path, start=None, end=None,
    )

    assert len(snapshots) == 1
    assert len(outcomes) == 1
    assert "category_mapping_json" not in snapshots.columns
    assert len(contracts) == 1
    assert {source["kind"] for source in sources} == {
        "event_shard", "feature_contract",
    }
    result, report = materialize_fixed_horizons(
        snapshots,
        outcomes,
        feature_contracts=contracts,
    )
    assert len(result) == 1
    assert result.loc[0, "A"] == 2.5
    assert result.loc[0, "raw__A"] == 1.5
    assert result.loc[0, "TARGET"] == 1
    contract = report["feature_contract"]
    X_model, _, _ = prepare_silver_training_data(
        result, feature_contract=contract,
    )
    X_raw, _, raw_contract = prepare_silver_training_data(
        result, representation="raw_challenger", feature_contract=contract,
    )
    assert X_model.loc[0, "A"] == 2.5
    assert X_raw.loc[0, "A"] == 1.5
    assert raw_contract["requires_train_only_category_fit"] is True

    output = tmp_path / "silver.parquet"
    assert materialize_cli.main([
        "--input-dir", str(tmp_path),
        "--horizons-hours", "2",
        "--output", str(output),
    ]) == 0
    output_body = output.read_bytes()
    manifest = json.loads(
        output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["materializer_version"] == MATERIALIZER_VERSION
    assert manifest["output_sha256"] == sha256(output_body).hexdigest()
    assert manifest["dataset_identity"]["output_sha256"] == manifest["output_sha256"]
    assert manifest["dataset_id"] == materialize_cli._dataset_id(
        manifest["dataset_identity"]
    )
    assert manifest["dataset_id"].startswith(MATERIALIZER_VERSION + "-")
    assert len(manifest["feature_contracts"]) == 1
    code_hash = manifest["materializer_code_sha256"]
    assert len(code_hash) == 64
    assert int(code_hash, 16) >= 0
    assert code_hash == materialize_cli._materializer_code_sha256()
    assert manifest["dataset_identity"]["materializer_code_sha256"] == code_hash
    assert manifest["runtime_versions"] == materialize_cli._runtime_versions()
    assert manifest["dataset_identity"]["runtime_versions"] == (
        manifest["runtime_versions"]
    )
    assert set(manifest["runtime_versions"]) == {
        "python", "pandas", "pyarrow", "numpy",
    }
    assert all(manifest["runtime_versions"].values())
    changed_identity = {
        **manifest["dataset_identity"],
        "output_sha256": "0" * 64,
    }
    assert materialize_cli._dataset_id(changed_identity) != manifest["dataset_id"]
    changed_code_identity = {
        **manifest["dataset_identity"],
        "materializer_code_sha256": "0" * 64,
    }
    assert (
        materialize_cli._dataset_id(changed_code_identity)
        != manifest["dataset_id"]
    )


def test_gcs_loader_fetches_contract_sidecar_with_generation_guard(
    monkeypatch: pytest.MonkeyPatch,
    published_ledger: _MemoryStorageClient,
) -> None:
    monkeypatch.setattr(
        "google.cloud.storage.Client",
        lambda: published_ledger,
    )

    snapshots, outcomes, contracts, sources = materialize_cli.load_gcs_events(
        "training-test",
        "ledger",
        start=None,
        end=None,
    )

    assert len(snapshots) == 1
    assert len(outcomes) == 1
    assert len(contracts) == 1
    assert len(sources) == 3
    assert all(source["generation"] == 1 for source in sources)
    result, _ = materialize_fixed_horizons(
        snapshots,
        outcomes,
        feature_contracts=contracts,
    )
    assert result.loc[0, "A"] == 2.5


def test_late_outcome_partition_is_selected_by_physical_flight_date(
    tmp_path: Path,
    published_ledger: _MemoryStorageClient,
) -> None:
    for name, body in published_ledger.objects.items():
        target_name = name
        if "/flight_outcomes/" in name:
            assert "flight_date=2026-08-11/" in name
            target_name = name.replace(
                "observed_date=2026-08-11",
                "observed_date=2026-09-20",
            )
        path = tmp_path / target_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    snapshots, outcomes, _, _ = materialize_cli.load_local_events(
        tmp_path,
        start=date(2026, 8, 11),
        end=date(2026, 8, 11),
        outcome_lag_days=0,
    )

    assert len(snapshots) == 1
    assert len(outcomes) == 1
    assert outcomes.loc[0, "flight_date_local"] == "2026-08-11"


def test_local_loader_fails_closed_when_referenced_sidecar_is_missing(
    tmp_path: Path,
    published_ledger: _MemoryStorageClient,
) -> None:
    for name, body in published_ledger.objects.items():
        if not name.endswith(".parquet"):
            continue
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    with pytest.raises(ValueError, match="missing feature-contract sidecars"):
        materialize_cli.load_local_events(tmp_path, start=None, end=None)


def test_contract_parser_rejects_mapping_hash_mismatch() -> None:
    bad_contract = {
        "feature_schema_hash": "not-used-because-mapping-fails-first",
        "category_mapping_sha256": "0" * 64,
        "feature_order_json": '["A"]',
        "categorical_features_json": "[]",
        "category_mapping_json": "{}",
    }

    with pytest.raises(ValueError, match="category mapping SHA-256 mismatch"):
        materialize_cli._feature_contract_from_bytes(
            json.dumps(bad_contract).encode("utf-8"),
            source_name="bad-contract.json",
        )


def test_cli_loads_causal_snapshot_partition_before_requested_flight_date(
    tmp_path: Path,
    published_ledger: _MemoryStorageClient,
) -> None:
    for name, body in published_ledger.objects.items():
        target_name = name
        if "/prediction_snapshots/" in name:
            target_name = name.replace(
                "event_date=2026-08-11",
                "event_date=2026-08-10",
            )
        path = tmp_path / target_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    output = tmp_path / "boundary-silver.parquet"
    assert materialize_cli.main(
        [
            "--input-dir", str(tmp_path),
            "--start-date", "2026-08-11",
            "--end-date", "2026-08-11",
            "--horizons-hours", "2",
            "--output", str(output),
        ]
    ) == 0

    result = pd.read_parquet(output)
    assert len(result) == 1
    manifest = json.loads(
        output.with_suffix(".parquet.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["report"]["flight_date_filter"]["boundary_lookback_days"] == 1
