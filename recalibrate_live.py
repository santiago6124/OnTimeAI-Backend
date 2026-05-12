"""Re-calibra el modelo v6_full con las predicciones live ya asentadas.

El calibrador original de v6_full fue entrenado sobre el set histórico 2022-2025.
En condiciones live (mayo 2026) hay drift sistemático:
  - Bin 0 (prob 0.0-0.1): modelo dice 5.7%  → observado 18.2%
  - Bin 1 (prob 0.1-0.2): modelo dice 14.5% → observado 28.4%
  - Bin 6 (prob 0.6-0.7): modelo dice 65%   → observado 21%

Este script:
  1. Lee 2,847 predicciones + actuals de live_data.db
  2. Ajusta un isotonic calibrador sobre esos datos (stacked sobre el original)
  3. Guarda artifacts/4year_v6_recal/ con calibrador compuesto + threshold actualizado
  4. Imprime ECE y Brier antes/después

Uso:
    python3 recalibrate_live.py
    python3 recalibrate_live.py --min-n 300 --out artifacts/4year_v6_recal
    python3 recalibrate_live.py --dry-run   # solo muestra métricas, no guarda
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from ontimeai.calibration import Calibrator, StackedCalibrator, fit_calibrator
from ontimeai.live import open_db
from ontimeai.model import load_artifact, save_artifact
from ontimeai.config import ARTIFACTS_DIR


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """ECE: weighted mean absolute difference entre prob predicha y observada."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    # calibration_curve puede tener menos bins si alguno está vacío; usar histograma uniforme
    bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(proba, bins=bins)
    total = counts.sum()
    ece = 0.0
    for i, (fp, mp) in enumerate(zip(frac_pos, mean_pred)):
        bin_lo = i / n_bins
        bin_hi = (i + 1) / n_bins
        n = ((proba >= bin_lo) & (proba < bin_hi)).sum()
        ece += (n / total) * abs(fp - mp)
    return float(ece)


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        n = mask.sum()
        if n == 0:
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": int(n),
            "mean_proba": float(proba[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
            "gap": float(abs(proba[mask].mean() - y_true[mask].mean())),
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ARTIFACTS_DIR / "4year_v6_recal"))
    p.add_argument("--source-artifact", default=str(ARTIFACTS_DIR / "4year_v6_full"))
    p.add_argument("--min-n", type=int, default=200,
                   help="Número mínimo de predicciones asentadas para recalibrar")
    p.add_argument("--delay-threshold-min", type=float, default=15.0,
                   help="Umbral de retraso en minutos para clasificación binaria")
    p.add_argument("--target-pos-rate", type=float, default=0.22,
                   help="Tasa positiva objetivo para recalcular threshold")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo muestra métricas, no guarda el artifact")
    p.add_argument("--strategy-filter", default=None,
                   help="Filtrar por threshold_strategy (e.g. 'artifact_batch_v7'). "
                        "Por defecto usa todas las predicciones.")
    args = p.parse_args()

    conn = open_db()
    now = datetime.now(timezone.utc)

    # ---- 1. cargar predicciones asentadas ----
    strategy_clause = (
        f"AND p.threshold_strategy = '{args.strategy_filter}'"
        if args.strategy_filter else ""
    )
    df = pd.read_sql_query(
        f"""
        SELECT p.fa_flight_id, p.proba_delay, p.predicted_delay,
               a.arr_delay_min, a.cancelled, a.diverted,
               f.origin, f.dest, f.op_carrier, f.fl_date
        FROM predictions p
        JOIN actuals   a ON a.fa_flight_id = p.fa_flight_id
        JOIN flights   f ON f.fa_flight_id = p.fa_flight_id
        WHERE a.cancelled = 0
          AND a.diverted  = 0
          AND a.arr_delay_min IS NOT NULL
          {strategy_clause}
        """,
        conn,
    )

    # Una predicción por vuelo: la más reciente
    df = df.sort_values("proba_delay").groupby("fa_flight_id", as_index=False).last()

    n_total = len(df)
    print(f"Predicciones asentadas disponibles: {n_total:,}")

    if n_total < args.min_n:
        print(f"Insuficientes datos (min={args.min_n}). Corré más ticks de live_pull primero.")
        return 1

    y_true = (df["arr_delay_min"] > args.delay_threshold_min).astype(int).values
    y_proba_live = df["proba_delay"].astype(float).values

    print(f"Tasa positiva real en live: {y_true.mean():.3f}")
    print(f"Tasa positiva predicha (pre-recal): {(y_proba_live >= 0.32).mean():.3f}")

    # ---- 2. métricas ANTES ----
    brier_before = brier_score_loss(y_true, y_proba_live)
    auc_before = roc_auc_score(y_true, y_proba_live) if len(np.unique(y_true)) > 1 else float("nan")
    ece_before = expected_calibration_error(y_true, y_proba_live)

    print(f"\n--- Antes de recalibrar ---")
    print(f"  AUC    : {auc_before:.4f}")
    print(f"  Brier  : {brier_before:.4f}")
    print(f"  ECE    : {ece_before:.4f}")

    cal_before = calibration_table(y_true, y_proba_live)
    print(cal_before.to_string(index=False))

    # ---- 3. ajustar calibrador sobre datos live ----
    # Los valores en DB ya son post-calibración del v6_full.
    # Ajustamos una segunda capa isotónica sobre esos valores → calibración apilada.
    live_cal = fit_calibrator(y_proba_live, y_true, method="isotonic")
    y_proba_recal = live_cal.transform(y_proba_live)

    # ---- 4. métricas DESPUÉS ----
    brier_after = brier_score_loss(y_true, y_proba_recal)
    auc_after = roc_auc_score(y_true, y_proba_recal) if len(np.unique(y_true)) > 1 else float("nan")
    ece_after = expected_calibration_error(y_true, y_proba_recal)

    print(f"\n--- Después de recalibrar ---")
    print(f"  AUC    : {auc_after:.4f}  (Δ {auc_after - auc_before:+.4f})")
    print(f"  Brier  : {brier_after:.4f}  (Δ {brier_after - brier_before:+.4f})")
    print(f"  ECE    : {ece_after:.4f}  (Δ {ece_after - ece_before:+.4f})")

    cal_after = calibration_table(y_true, y_proba_recal)
    print(cal_after.to_string(index=False))

    # ---- 5. nuevo threshold ----
    # Isotonic regression produce salidas discretas (función escalón): puede haber
    # muchas predicciones con exactamente el mismo valor, lo que rompe el percentil
    # simple. Buscamos el threshold mínimo tal que pos_rate sea <= target_pos_rate.
    sorted_unique = np.sort(np.unique(y_proba_recal))[::-1]
    new_threshold = float(sorted_unique[-1])  # fallback: mínimo valor
    for t in sorted_unique:
        if (y_proba_recal >= t).mean() <= args.target_pos_rate:
            new_threshold = float(t)
            break
    actual_pos_rate = (y_proba_recal >= new_threshold).mean()
    print(f"\nNuevo threshold (target_pos_rate={args.target_pos_rate}): {new_threshold:.4f}")
    print(f"Pos rate real con nuevo threshold: {actual_pos_rate:.3f}")

    if args.dry_run:
        print("\n[dry-run] No se guarda el artifact.")
        return 0

    # ---- 6. crear artifact 4year_v6_recal ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # El booster no cambia — creamos un hard link para evitar duplicar 23MB
    src_model = Path(args.source_artifact) / "model.lgb"
    dst_model = out_dir / "model.lgb"
    if dst_model.exists():
        dst_model.unlink()
    try:
        dst_model.hardlink_to(src_model)
    except OSError:
        shutil.copy2(src_model, dst_model)

    # Cargamos v6_full para extraer la metadata base
    meta_src = load_artifact(Path(args.source_artifact))

    # Calibrador compuesto: original v6_full + live isotonic
    stacked = StackedCalibrator(cal1=meta_src["calibrator"], cal2=live_cal)

    # Guardamos nuevo meta
    import joblib
    new_meta = {
        "threshold":    new_threshold,
        "feature_cols": meta_src["feature_cols"],
        "cat_cols":     meta_src["cat_cols"],
        "cat_mapping":  meta_src["cat_mapping"],
        "target":       meta_src["target"],
        "calibrator":   stacked,
    }
    joblib.dump(new_meta, out_dir / "meta.joblib")

    # Guardamos reporte de recalibración
    report = {
        "created_at_utc": now.isoformat(),
        "source_artifact": str(args.source_artifact),
        "n_live_samples": int(n_total),
        "delay_threshold_min": args.delay_threshold_min,
        "before": {
            "auc": round(auc_before, 4),
            "brier": round(brier_before, 4),
            "ece": round(ece_before, 4),
        },
        "after": {
            "auc": round(auc_after, 4),
            "brier": round(brier_after, 4),
            "ece": round(ece_after, 4),
        },
        "delta": {
            "auc":   round(auc_after - auc_before, 4),
            "brier": round(brier_after - brier_before, 4),
            "ece":   round(ece_after - ece_before, 4),
        },
        "new_threshold": new_threshold,
        "target_pos_rate": args.target_pos_rate,
    }
    (out_dir / "recal_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nArtifact guardado en: {out_dir}")
    print(f"  model.lgb  → hard link a {src_model}")
    print(f"  meta.joblib → StackedCalibrator + threshold={new_threshold:.4f}")
    print(f"  recal_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
