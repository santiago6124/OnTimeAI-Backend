"""`prediction_shap` no puede volver a desbordar la base.

El 29/09 el harvester falló en cinco corridas seguidas: baja y sube la base
entera en cada pasada y a 815 MB no le entraba en el timeout. Medido sobre esa
base, `prediction_shap` era el **68%** del archivo:

    prediction_shap                        232,8 MB   31%
    sqlite_autoindex_prediction_shap_1     158,4 MB   21%
    idx_shap_pred                          120,8 MB   16%

Dos causas, las dos innecesarias:

  1. `idx_shap_pred` era un prefijo de la PRIMARY KEY, que SQLite ya respalda
     con su propio índice. No habilitaba ningún plan nuevo.
  2. Se guardaban 7 días de SHAP —345.000 filas diarias— cuando la pantalla
     solo muestra el de los vuelos del día y nada lo usa para reentrenar.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAIZ = Path(__file__).resolve().parents[1]
FUENTE_JOB = (RAIZ / "live_job.py").read_text("utf-8")


def test_no_se_crea_el_indice_redundante() -> None:
    from ontimeai.live import open_db

    con = open_db(":memory:")
    indices = {
        r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='prediction_shap'").fetchall()
    }
    assert "idx_shap_pred" not in indices, indices


def test_la_migracion_lo_borra_de_una_base_que_ya_lo_tiene() -> None:
    """Las bases en producción lo traen; hay que sacárselo, no solo dejar de crearlo."""
    from ontimeai.live import SCHEMA, _migrate_drop_idx_shap_pred

    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_shap_pred "
        "ON prediction_shap(fa_flight_id, predicted_at_utc)")
    assert con.execute(
        "SELECT 1 FROM sqlite_master WHERE name='idx_shap_pred'").fetchone()

    _migrate_drop_idx_shap_pred(con)

    assert not con.execute(
        "SELECT 1 FROM sqlite_master WHERE name='idx_shap_pred'").fetchone()


def test_la_clave_primaria_sigue_cubriendo_la_consulta() -> None:
    """
    Lo que justifica borrar el índice: su plan de consulta lo sirve igual el
    índice de la PRIMARY KEY, porque empieza por las mismas dos columnas.
    """
    from ontimeai.live import open_db

    con = open_db(":memory:")
    plan = con.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM prediction_shap "
        "WHERE fa_flight_id = ? AND predicted_at_utc = ?", ("x", "y")).fetchall()
    texto = " ".join(str(c) for fila in plan for c in fila)
    assert "SCAN" not in texto.upper() or "SEARCH" in texto.upper(), texto


def test_la_retencion_del_shap_es_corta() -> None:
    """Siete días eran 345.000 filas diarias que nadie abría."""
    m = re.search(r'SHAP_RETENTION_DAYS", "(\d+)"', FUENTE_JOB)
    assert m, "la retención del SHAP debería ser configurable por entorno"
    assert int(m.group(1)) <= 3, f"default de {m.group(1)} días es demasiado"


def test_la_retencion_se_puede_ajustar_por_entorno(monkeypatch) -> None:
    """Sin esto, corregir un desborde en producción exige desplegar."""
    assert 'os.environ.get("SHAP_RETENTION_DAYS"' in FUENTE_JOB
