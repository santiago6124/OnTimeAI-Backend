"""Cloud Run Job entrypoint: descarga DB de GCS, corre live_pull, sube el DB actualizado."""
from __future__ import annotations

import os
import resource
import shutil
import sys
from pathlib import Path

from ontimeai.training_store import read_bool_env


def _log_mem(label: str) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[mem] {label}: peak RSS {rss_mb:.0f} MB")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_OBJECT = "live_data.db"
TMP_DB = Path("/tmp/live_data.db")
BUNDLED_DB = Path(__file__).parent / "live_data.db"
GCS_GENERATION_RETRIES = max(0, int(os.environ.get("GCS_GENERATION_RETRIES", "2")))
PRUNE_VACUUM_MIN_DELETED = max(
    0,
    int(os.environ.get("PRUNE_VACUUM_MIN_DELETED", "50000")),
)
TRAINING_DATA_BUCKET = os.environ.get("TRAINING_DATA_BUCKET", "").strip()
TRAINING_DATA_PREFIX = os.environ.get("TRAINING_DATA_PREFIX", "live-training").strip("/")
TRAINING_STORE_REQUIRED = bool(
    read_bool_env("TRAINING_STORE_REQUIRED", default=False),
)

# Apuntar DB_PATH al /tmp antes de que ontimeai.live se importe
os.environ["DB_PATH"] = str(TMP_DB)


class GCSGenerationConflict(RuntimeError):
    """The shared DB changed after this process selected its base generation."""


def _validate_training_store_config() -> None:
    """Reject modes that cannot durably deliver the transactional outbox."""
    enabled_override = read_bool_env("TRAINING_STORE_ENABLED", default=None)
    if TRAINING_STORE_REQUIRED and enabled_override is False:
        raise RuntimeError(
            "TRAINING_STORE_REQUIRED=true conflicts with TRAINING_STORE_ENABLED=false"
        )
    capture_enabled = (
        TRAINING_STORE_REQUIRED
        or enabled_override is True
        or (enabled_override is None and bool(TRAINING_DATA_BUCKET))
    )
    if capture_enabled and not TRAINING_DATA_BUCKET:
        raise RuntimeError(
            "training capture requires a non-empty TRAINING_DATA_BUCKET"
        )
    if TRAINING_DATA_BUCKET and not TRAINING_DATA_PREFIX:
        raise RuntimeError("TRAINING_DATA_PREFIX must not be empty")
    if TRAINING_DATA_BUCKET and not GCS_BUCKET:
        raise RuntimeError(
            "training-store delivery requires GCS_BUCKET so the SQLite outbox "
            "survives process restarts"
        )


def _gcs_blob():
    from google.cloud import storage as gcs

    client = gcs.Client()
    return client.bucket(GCS_BUCKET).blob(GCS_OBJECT)


def _gcs_download() -> int:
    """Download one immutable GCS generation and return its generation number.

    Generation ``0`` means the object did not exist when the snapshot was
    selected.  The matching upload then uses ``if_generation_match=0`` so it
    can create the object but cannot overwrite a concurrently-created one.
    """
    from google.api_core.exceptions import NotFound, PreconditionFailed

    blob = _gcs_blob()
    try:
        blob.reload()
    except NotFound:
        print("[job] No hay DB en GCS todavía, usando la bundleada como base.")
        shutil.copy(BUNDLED_DB, TMP_DB)
        return 0

    generation = int(blob.generation)
    try:
        blob.download_to_filename(
            str(TMP_DB),
            if_generation_match=generation,
        )
    except PreconditionFailed as exc:
        raise GCSGenerationConflict(
            f"la DB cambió durante la descarga de la generación {generation}"
        ) from exc

    print(
        f"[job] Descargado {TMP_DB.stat().st_size / 1e6:.1f} MB desde "
        f"gs://{GCS_BUCKET}/{GCS_OBJECT} (generation={generation})"
    )
    return generation


def _cleanup_old_data() -> None:
    """Trim stale data to keep DB small. prediction_shap is not needed for retraining."""
    import sqlite3 as _sqlite3
    con = _sqlite3.connect(str(TMP_DB))
    # SHAP values: keep 7 days only (UI only, never used for retraining)
    con.execute("DELETE FROM prediction_shap WHERE predicted_at_utc < datetime('now', '-7 days')")
    shap_deleted = con.total_changes
    # Weather observations: keep 30 days (can be re-pulled from IEM if needed)
    con.execute("DELETE FROM weather_obs WHERE valid_utc < datetime('now', '-30 days')")
    weather_deleted = con.total_changes - shap_deleted
    con.commit()
    total_deleted = shap_deleted + weather_deleted
    db_mb = TMP_DB.stat().st_size / 1e6
    print(f"[job] cleanup: -{shap_deleted} SHAP rows, -{weather_deleted} weather rows (DB: {db_mb:.0f} MB)")
    if total_deleted > PRUNE_VACUUM_MIN_DELETED:
        try:
            print(
                "[job] Running VACUUM: "
                f"{total_deleted} deleted rows exceed the "
                f"{PRUNE_VACUUM_MIN_DELETED} threshold..."
            )
            con.execute("VACUUM")
            con.commit()
            db_mb_after = TMP_DB.stat().st_size / 1e6
            print(f"[job] post-VACUUM: {db_mb_after:.0f} MB (saved {db_mb - db_mb_after:.0f} MB)")
        except Exception as e:
            print(f"[job] VACUUM skipped: {e}")
    else:
        print(
            "[job] VACUUM skipped: "
            f"{total_deleted} deleted rows do not exceed the "
            f"{PRUNE_VACUUM_MIN_DELETED} threshold."
        )
    con.close()


def _gcs_upload(expected_generation: int) -> int:
    """Upload only if GCS still contains ``expected_generation``.

    A mismatch is a normal optimistic-concurrency conflict: the caller must
    restart from the new winning generation instead of overwriting it.
    """
    from google.api_core.exceptions import PreconditionFailed

    import sqlite3 as _sqlite3
    _log_mem("pre-upload")
    try:
        chk = _sqlite3.connect(str(TMP_DB))
        quick = chk.execute("PRAGMA quick_check(1)").fetchone()
        if not quick or quick[0] != "ok":
            # Full integrity check to log exactly what's broken
            errors = chk.execute("PRAGMA integrity_check(20)").fetchall()
            chk.close()
            print(f"[job] ABORT upload — DB corrupted. quick_check: {quick}")
            for row in errors:
                print(f"[job]   integrity_check: {row[0]}")
            raise RuntimeError("DB integrity check failed; refusing GCS upload")
        chk.close()
    except Exception as e:
        print(f"[job] ABORT upload — DB integrity check error: {e}")
        raise

    blob = _gcs_blob()
    try:
        blob.upload_from_filename(
            str(TMP_DB),
            if_generation_match=expected_generation,
        )
    except PreconditionFailed as exc:
        raise GCSGenerationConflict(
            f"GCS ya no está en generation={expected_generation}"
        ) from exc

    uploaded_generation = int(blob.generation)
    print(
        f"[job] Subido {TMP_DB.stat().st_size / 1e6:.1f} MB a "
        f"gs://{GCS_BUCKET}/{GCS_OBJECT} (generation={uploaded_generation}, "
        f"base={expected_generation})"
    )
    return uploaded_generation


def _publish_training_outbox(*, mark_delivered: bool) -> int:
    """Publish exact feature/outcome events without mutating the training bucket.

    With ``mark_delivered=False`` this runs after the winning SQLite generation
    has been uploaded.  The outbox remains in that durable generation and is
    acknowledged at the start of the next winning attempt, avoiding a second
    468-MB SQLite upload per cycle.
    """
    if not TRAINING_DATA_BUCKET:
        if TRAINING_STORE_REQUIRED:
            raise RuntimeError(
                "TRAINING_STORE_REQUIRED=true requires TRAINING_DATA_BUCKET"
            )
        return 0

    import sqlite3 as _sqlite3

    from ontimeai.training_store import publish_pending_outbox

    con = _sqlite3.connect(str(TMP_DB))
    try:
        try:
            backlog = con.execute(
                "SELECT count(*), min(created_at_utc) FROM training_export_outbox"
            ).fetchone()
        except _sqlite3.OperationalError:
            backlog = (0, None)
        backlog_count = int(backlog[0] or 0)
        if backlog_count:
            warn_at = max(
                1, int(os.environ.get("TRAINING_OUTBOX_WARN_EVENTS", "20000")),
            )
            level = "WARNING" if backlog_count >= warn_at else "pending"
            print(
                f"[training-store] backlog {level}: {backlog_count} events; "
                f"oldest={backlog[1]}"
            )
        summary = publish_pending_outbox(
            con,
            bucket_name=TRAINING_DATA_BUCKET,
            prefix=TRAINING_DATA_PREFIX,
            mark_delivered=mark_delivered,
        )
    finally:
        con.close()
    if summary.events:
        state = "published+acknowledged" if mark_delivered else "published"
        print(
            f"[training-store] {state}: {summary.events} events in "
            f"{summary.objects} immutable parquet object(s)"
        )
    return summary.events


def _handle_training_publish(*, mark_delivered: bool) -> bool:
    """Return False only when a required training-store delivery failed."""
    try:
        _publish_training_outbox(mark_delivered=mark_delivered)
        return True
    except Exception as exc:
        print(
            "[training-store] publish failed "
            f"(mark_delivered={mark_delivered}): {type(exc).__name__}: {exc}"
        )
        return not TRAINING_STORE_REQUIRED


def _live_pull_args_from_env() -> list[str]:
    extra_args: list[str] = []
    for env_name, flag in (
        ("ACTUALS_HOURS", "--actuals-hours"),
        ("ACTUALS_OFFSET_HOURS", "--actuals-offset-hours"),
        ("SCHEDULE_HOURS", "--schedule-hours"),
        ("CHAIN_WALK_MAX", "--chain-walk-max"),
        ("TARGET_POS_RATE", "--target-pos-rate"),
        ("ABS_THRESHOLD", "--abs-threshold"),
        ("MAX_PAGES", "--max-pages"),
    ):
        val = os.environ.get(env_name)
        if val:
            extra_args.extend([flag, val])
    return extra_args


def _run_pipeline_attempt(extra_args: list[str]) -> int:
    """Run one complete local mutation attempt against the current TMP_DB."""
    sys.argv = [sys.argv[0]] + extra_args
    if extra_args:
        print(f"[job] live_pull args from env: {' '.join(extra_args)}")
    sys.path.insert(0, str(Path(__file__).parent))
    _log_mem("pre-pipeline")
    import live_pull

    exit_code = live_pull.main()
    _log_mem("post-pipeline")
    if exit_code != 0:
        print(f"[job] live_pull failed with exit_code={exit_code}; upload skipped")
        return exit_code

    if TMP_DB.exists():
        print("[job] Running database pruning...")
        try:
            from scripts.prune_db import prune_db
            prune_db(
                TMP_DB,
                days=30,
                dry_run=False,
                vacuum_min_deleted=PRUNE_VACUUM_MIN_DELETED,
            )
        except Exception as e:
            print(f"[job] Error running database pruning: {e}")

    return 0


def main() -> int:
    try:
        _validate_training_store_config()
    except (RuntimeError, ValueError) as exc:
        print(f"[training-store] invalid configuration: {exc}")
        return 4
    extra_args = _live_pull_args_from_env()

    if not GCS_BUCKET:
        print("[job] GCS_BUCKET no configurado, usando DB local.")
        shutil.copy(BUNDLED_DB, TMP_DB)
        os.environ["LIVE_DB_BASE_GENERATION"] = "local"
        if not _handle_training_publish(mark_delivered=True):
            return 4
        exit_code = _run_pipeline_attempt(extra_args)
        if exit_code == 0 and not _handle_training_publish(mark_delivered=True):
            return 4
        _log_mem("end")
        return exit_code

    for attempt in range(GCS_GENERATION_RETRIES + 1):
        try:
            base_generation = _gcs_download()
            os.environ["LIVE_DB_BASE_GENERATION"] = str(base_generation)
            print(
                f"[job] mutation attempt {attempt + 1}/"
                f"{GCS_GENERATION_RETRIES + 1} from generation={base_generation}"
            )
            # Drain events from the previous winning generation before adding a
            # new transactional outbox batch.  Any later SQLite CAS conflict
            # simply retries against the winner; GCS objects are create-only.
            if not _handle_training_publish(mark_delivered=True):
                return 4
            exit_code = _run_pipeline_attempt(extra_args)
            if exit_code != 0:
                return exit_code
            if not TMP_DB.exists():
                raise FileNotFoundError(f"pipeline did not produce {TMP_DB}")

            _cleanup_old_data()
            _gcs_upload(base_generation)
            # Publish the newly durable outbox immediately, but acknowledge it
            # next cycle so no second full-DB upload is needed.
            if not _handle_training_publish(mark_delivered=False):
                return 4
            _log_mem("end")
            return 0
        except GCSGenerationConflict as exc:
            if attempt >= GCS_GENERATION_RETRIES:
                print(
                    f"[job] generation conflict after {attempt + 1} attempts; "
                    f"refusing stale overwrite: {exc}"
                )
                return 3
            print(
                f"[job] generation conflict: {exc}. Reloading the winning DB "
                f"and retrying ({attempt + 2}/{GCS_GENERATION_RETRIES + 1})."
            )

    return 3


if __name__ == "__main__":
    raise SystemExit(main())
