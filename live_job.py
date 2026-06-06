"""Cloud Run Job entrypoint: descarga DB de GCS, corre live_pull, sube el DB actualizado."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

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


def _gcs_upload() -> None:
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
    import live_pull
    exit_code = live_pull.main()

    if GCS_BUCKET and TMP_DB.exists():
        _gcs_upload()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
