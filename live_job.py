"""Cloud Run Job entrypoint: descarga DB de GCS, corre live_pull, sube el DB actualizado."""
from __future__ import annotations

import os
import resource
import shutil
import sys
from pathlib import Path


def _log_mem(label: str) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[mem] {label}: peak RSS {rss_mb:.0f} MB")

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_OBJECT = "live_data.db"
TMP_DB = Path("/tmp/live_data.db")
BUNDLED_DB = Path(__file__).parent / "live_data.db"

# Apuntar DB_PATH al /tmp antes de que ontimeai.live se importe
os.environ["DB_PATH"] = str(TMP_DB)


def _gcs_download() -> None:
    from google.cloud import storage as gcs
    client = gcs.Client()
    blob = client.bucket(GCS_BUCKET).blob(GCS_OBJECT)
    if blob.exists():
        blob.download_to_filename(str(TMP_DB))
        print(f"[job] Descargado {TMP_DB.stat().st_size / 1e6:.1f} MB desde gs://{GCS_BUCKET}/{GCS_OBJECT}")
    else:
        print("[job] No hay DB en GCS todavía, usando la bundleada como base.")
        shutil.copy(BUNDLED_DB, TMP_DB)


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
    db_mb = TMP_DB.stat().st_size / 1e6
    print(f"[job] cleanup: -{shap_deleted} SHAP rows, -{weather_deleted} weather rows (DB: {db_mb:.0f} MB)")
    if db_mb > 800:
        # First time after cleanup: VACUUM reclaims freed space
        # (only runs until DB stabilizes below 800 MB)
        try:
            print("[job] Running VACUUM to reclaim space...")
            con.execute("VACUUM")
            con.commit()
            db_mb_after = TMP_DB.stat().st_size / 1e6
            print(f"[job] post-VACUUM: {db_mb_after:.0f} MB (saved {db_mb - db_mb_after:.0f} MB)")
        except Exception as e:
            print(f"[job] VACUUM skipped: {e}")
    con.close()


def _gcs_upload() -> None:
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
            return
        chk.close()
    except Exception as e:
        print(f"[job] ABORT upload — DB integrity check error: {e}")
        return
    from google.cloud import storage as gcs
    client = gcs.Client()
    blob = client.bucket(GCS_BUCKET).blob(GCS_OBJECT)
    blob.upload_from_filename(str(TMP_DB))
    print(f"[job] Subido {TMP_DB.stat().st_size / 1e6:.1f} MB a gs://{GCS_BUCKET}/{GCS_OBJECT}")


def main() -> int:
    if GCS_BUCKET:
        _gcs_download()
    else:
        print("[job] GCS_BUCKET no configurado, usando DB local.")
        shutil.copy(BUNDLED_DB, TMP_DB)

    # live_pull.main() usa parse_args() — sys.argv vacío usa defaults.
    # Permitimos override de args clave via env vars (útil para backfill o
    # diagnóstico). Si no están seteadas, mantenemos defaults.
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

    sys.argv = [sys.argv[0]] + extra_args
    if extra_args:
        print(f"[job] live_pull args from env: {' '.join(extra_args)}")
    sys.path.insert(0, str(Path(__file__).parent))
    _log_mem("pre-pipeline")
    import live_pull
    exit_code = live_pull.main()
    _log_mem("post-pipeline")

    if TMP_DB.exists():
        print("[job] Running database pruning...")
        try:
            from scripts.prune_db import prune_db
            prune_db(TMP_DB, days=30, dry_run=False)
        except Exception as e:
            print(f"[job] Error running database pruning: {e}")

    if GCS_BUCKET and TMP_DB.exists():
        _cleanup_old_data()
        _gcs_upload()

    _log_mem("end")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
