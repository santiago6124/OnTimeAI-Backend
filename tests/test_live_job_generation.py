"""Optimistic-concurrency tests for the shared GCS SQLite object."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from google.api_core.exceptions import PreconditionFailed

import live_job


class FakeBlob:
    def __init__(self, generation: int = 10) -> None:
        self.generation = str(generation)
        self.download_precondition: int | None = None
        self.upload_precondition: int | None = None
        self.fail_upload = False

    def reload(self) -> None:
        return None

    def download_to_filename(self, filename: str, *, if_generation_match: int) -> None:
        self.download_precondition = if_generation_match
        Path(filename).write_bytes(b"immutable generation snapshot")

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        self.upload_precondition = if_generation_match
        if self.fail_upload:
            raise PreconditionFailed("simulated concurrent writer")
        assert Path(filename).exists()
        self.generation = str(if_generation_match + 1)


def _valid_sqlite(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE audit_ok(id INTEGER PRIMARY KEY)")
    con.commit()
    con.close()


def _cleanup_sqlite(path: Path, *, old_rows: int) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE prediction_shap(predicted_at_utc TEXT)")
    con.execute("CREATE TABLE weather_obs(valid_utc TEXT)")
    con.executemany(
        "INSERT INTO prediction_shap(predicted_at_utc) VALUES (?)",
        [("2000-01-01T00:00:00+00:00",)] * old_rows,
    )
    con.commit()
    con.close()


def test_download_is_pinned_to_reloaded_generation(tmp_path: Path, monkeypatch) -> None:
    blob = FakeBlob(generation=42)
    target = tmp_path / "live.db"
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)

    generation = live_job._gcs_download()

    assert generation == 42
    assert blob.download_precondition == 42
    assert target.exists()


def test_upload_uses_base_generation(tmp_path: Path, monkeypatch) -> None:
    blob = FakeBlob(generation=42)
    target = tmp_path / "live.db"
    _valid_sqlite(target)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)

    uploaded_generation = live_job._gcs_upload(expected_generation=42)

    assert blob.upload_precondition == 42
    assert uploaded_generation == 43


def test_upload_conflict_never_falls_back_to_unconditional_write(
    tmp_path: Path, monkeypatch
) -> None:
    blob = FakeBlob(generation=42)
    blob.fail_upload = True
    target = tmp_path / "live.db"
    _valid_sqlite(target)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "_gcs_blob", lambda: blob)

    with pytest.raises(live_job.GCSGenerationConflict):
        live_job._gcs_upload(expected_generation=42)

    assert blob.upload_precondition == 42


def test_cleanup_skips_vacuum_at_threshold(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    target = tmp_path / "cleanup-skip.db"
    _cleanup_sqlite(target, old_rows=2)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "PRUNE_VACUUM_MIN_DELETED", 2)

    live_job._cleanup_old_data()

    output = capsys.readouterr().out
    assert "VACUUM skipped: 2 deleted rows" in output
    assert "Running VACUUM" not in output


def test_cleanup_runs_vacuum_above_threshold(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    target = tmp_path / "cleanup-vacuum.db"
    _cleanup_sqlite(target, old_rows=3)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "PRUNE_VACUUM_MIN_DELETED", 2)

    live_job._cleanup_old_data()

    output = capsys.readouterr().out
    assert "Running VACUUM: 3 deleted rows" in output
    assert "post-VACUUM" in output


def test_main_reloads_winning_generation_and_retries(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "live.db"
    target.write_bytes(b"placeholder")
    generations = iter([10, 11])
    uploaded_from: list[int] = []
    pipeline_calls: list[int] = []

    monkeypatch.setattr(live_job, "GCS_BUCKET", "audit-bucket")
    monkeypatch.setattr(live_job, "GCS_GENERATION_RETRIES", 1)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "_gcs_download", lambda: next(generations))
    monkeypatch.setattr(live_job, "_cleanup_old_data", lambda: None)
    monkeypatch.setattr(live_job, "_live_pull_args_from_env", lambda: [])
    monkeypatch.setattr(
        live_job,
        "_run_pipeline_attempt",
        lambda _args: pipeline_calls.append(1) or 0,
    )

    def guarded_upload(expected_generation: int) -> int:
        uploaded_from.append(expected_generation)
        if expected_generation == 10:
            raise live_job.GCSGenerationConflict("simulated winner")
        return 12

    monkeypatch.setattr(live_job, "_gcs_upload", guarded_upload)

    assert live_job.main() == 0
    assert uploaded_from == [10, 11]
    assert len(pipeline_calls) == 2


def test_training_outbox_is_drained_before_pipeline_and_published_after_win(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "live.db"
    _valid_sqlite(target)
    order: list[str] = []

    monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", True)
    monkeypatch.setattr(live_job, "GCS_GENERATION_RETRIES", 0)
    monkeypatch.setattr(live_job, "TMP_DB", target)
    monkeypatch.setattr(live_job, "_gcs_download", lambda: order.append("download") or 20)
    monkeypatch.setattr(live_job, "_cleanup_old_data", lambda: order.append("cleanup"))
    monkeypatch.setattr(live_job, "_live_pull_args_from_env", lambda: [])
    monkeypatch.setattr(
        live_job,
        "_publish_training_outbox",
        lambda *, mark_delivered: order.append(
            "ack-old" if mark_delivered else "publish-new"
        ) or 0,
    )
    monkeypatch.setattr(
        live_job,
        "_run_pipeline_attempt",
        lambda _args: order.append("pipeline") or 0,
    )
    monkeypatch.setattr(
        live_job,
        "_gcs_upload",
        lambda expected_generation: order.append(f"upload-{expected_generation}") or 21,
    )

    assert live_job.main() == 0
    assert order == [
        "download", "ack-old", "pipeline", "cleanup", "upload-20", "publish-new",
    ]


def test_required_training_store_rejects_missing_bucket(monkeypatch) -> None:
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", True)

    assert live_job._handle_training_publish(mark_delivered=True) is False


@pytest.mark.parametrize("value", ["1", " true ", "YES", "on"])
def test_required_boolean_parser_accepts_documented_true_values(
    monkeypatch, value: str,
) -> None:
    monkeypatch.setenv("TRAINING_STORE_REQUIRED", value)

    assert live_job.read_bool_env(
        "TRAINING_STORE_REQUIRED", default=False,
    ) is True


def test_main_rejects_capture_without_training_bucket(monkeypatch) -> None:
    monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
    monkeypatch.setenv("TRAINING_STORE_ENABLED", "true")

    assert live_job.main() == 4


def test_main_rejects_training_bucket_with_ephemeral_local_db(monkeypatch) -> None:
    monkeypatch.setattr(live_job, "GCS_BUCKET", "")
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
    monkeypatch.delenv("TRAINING_STORE_ENABLED", raising=False)

    assert live_job.main() == 4


def test_main_rejects_required_disabled_conflict(monkeypatch) -> None:
    monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", True)
    monkeypatch.setenv("TRAINING_STORE_ENABLED", "false")

    assert live_job.main() == 4


def test_main_rejects_invalid_enabled_boolean(monkeypatch) -> None:
    monkeypatch.setattr(live_job, "GCS_BUCKET", "live-bucket")
    monkeypatch.setattr(live_job, "TRAINING_DATA_BUCKET", "training-bucket")
    monkeypatch.setattr(live_job, "TRAINING_STORE_REQUIRED", False)
    monkeypatch.setenv("TRAINING_STORE_ENABLED", "truthy-ish")

    assert live_job.main() == 4
