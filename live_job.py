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

    # live_pull.main() usa parse_args() — sys.argv vacío usa todos los defaults
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(Path(__file__).parent))
    import live_pull
    exit_code = live_pull.main()

    if GCS_BUCKET and TMP_DB.exists():
        _gcs_upload()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
