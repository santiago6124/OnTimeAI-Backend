"""Re-calibra un modelo con las predicciones live ya asentadas.

SUPERADO POR `ontimeai/chain_calibration.py`. No generes artefactos con esto
para ponerlos en producción.

Lo que sirve hoy es el calibrador post-cadena: vive en la base, se reajusta
cada ciclo con los últimos 30 días y se apaga si no puede hacerlo bien. Este
script produce un artefacto CONGELADO, y un corrector que no se actualiza
corrige en la dirección equivocada: medido, `4year_v9_recal` —del 06/08— deja
el ECE en 0,0490 cuando sin él es 0,0164.

Los artefactos `*_recal` del repo son residuos de este enfoque, no modelos
para activar vía `ACTIVE_MODEL`.

Queda por dos razones. La primera es que el script ya existía y alguien podía
correrlo y generar un artefacto malo sin enterarse: ahora hay un control fuera
de muestra que se niega a guardar si el recalibrado no mejora sobre datos no
vistos (ver el paso 5b). Fue exactamente esa puerta la que dejó pasar el
artefacto del 06/08, que declaraba AUC 0,9063 —inflado por el bug de selección
que elegía la predicción más optimista de cada vuelo—.

La segunda es que `ontimeai/validacion_calibrado.py`, que salió de acá, sirve
para juzgar cualquier recalibrado caminando cortes hacia adelante, incluido el
post-cadena.


Uso habitual:
    # v9 recalibration (default):
    python3 recalibrate_live.py
    python3 recalibrate_live.py --dry-run   # solo muestra métricas, no guarda

    # v6_full → v6_recal (uso original):
    python3 recalibrate_live.py \\
        --source-artifact artifacts/4year_v6_full \\
        --out artifacts/4year_v6_recal \\
        --since 2000-01-01

El script ajusta un isotonic calibrador de segunda capa sobre las probabilidades
ya calibradas que se almacenan en live_data.db, produciendo un StackedCalibrator.

OJO AL APILAR: `--source-artifact` tiene que ser el artefacto que PRODUJO las
predicciones de las que se aprende, o sea el `ACTIVE_MODEL` que estaba corriendo
en la ventana de `--since`.

El calibrador de segunda capa se ajusta sobre `proba_delay`, que es la salida
del calibrador del artefacto activo. Si se pasa como fuente un artefacto
distinto, el resultado aplica una capa que espera una entrada que nunca va a
recibir, y queda roto sin avisar.

No hay forma de detectarlo desde la base: no se registra qué artefacto generó
cada predicción —`predictions` no tiene columna de versión y `runs` tampoco— y
comparar `cal1(proba_raw)` contra `proba_delay` no sirve porque hay ajustes
posteriores a la calibración (estado del día, GDP, cadena): medido el 11/10 la
diferencia media es 0,032 y solo el 0,1% coincide. Hasta que exista esa
columna, la correspondencia es responsabilidad de quien corre el script.

Por qué se apila en vez de reemplazar: ajustar una capa única directo sobre
`proba_raw` es más simple y no tiene esta trampa, pero sale peor. Medido fuera
de muestra el 11/10 sobre los mismos 9 cortes:

    produccion (sin recalibrar)   Brier 0,0914  AUC 0,7289  gana   -
    apilado sobre proba_delay     Brier 0,0882  AUC 0,7240  gana 9/9
    desde cero sobre proba_raw    Brier 0,0897  AUC 0,7179  gana 3/9

El calibrador del artefacto hace trabajo que va más allá de mover el nivel, y
reemplazarlo pierde esa información.
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
from ontimeai.validacion_calibrado import (
    caminar_cortes, elegir_threshold, veredicto)
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
    p.add_argument("--out", default=str(ARTIFACTS_DIR / "4year_v9_recal"))
    p.add_argument("--source-artifact", default=str(ARTIFACTS_DIR / "4year_v9"))
    p.add_argument("--since", default="2026-05-15 01:43",
                   help="Only use predictions at or after this UTC timestamp (ISO format). "
                        "Default: v9 deploy time. Use '2000-01-01' for all predictions.")
    p.add_argument("--min-n", type=int, default=2000,
                   help="Número mínimo de predicciones asentadas para recalibrar")
    # 200 era demasiado bajo para isotonic. En el camino de cortes del 11/10,
    # ajustar con 305 vuelos empeoraba el Brier de 0,0506 a 0,0662 y con 1.253
    # de 0,0725 a 0,0742; el primer corte que mejora ajusta con 2.006. Con
    # muestra chica el isotónico sí sobreajusta, que era la advertencia del
    # comentario que eligió sigmoid —correcta en ese régimen, no en este—.
    p.add_argument("--delay-threshold-min", type=float, default=15.0,
                   help="Umbral de retraso en minutos para clasificación binaria")
    p.add_argument("--target-pos-rate", type=float, default=0.22,
                   help="Tasa positiva objetivo para recalcular threshold")
    p.add_argument("--dry-run", action="store_true",
                   help="Solo muestra métricas, no guarda el artifact")
    p.add_argument("--method", choices=("isotonic", "sigmoid"), default="isotonic",
                   help="Familia del calibrador de segunda capa. Isotonic por "
                        "medición fuera de muestra; ver el comentario en el paso 3.")
    p.add_argument("--horizonte-dias", type=int, default=2,
                   help="Días de vuelos que se evalúan después de cada corte")
    p.add_argument("--minimo-ganados", type=float, default=0.7,
                   help="Fracción de cortes que el recalibrado debe ganar para "
                        "que se guarde")
    p.add_argument("--sin-control", action="store_true",
                   help="Salta el control fuera de muestra (no recomendado)")
    p.add_argument("--force", action="store_true",
                   help="Guarda aunque el control fuera de muestra salga peor "
                        "que no recalibrar")
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
    since_clause = f"AND p.predicted_at_utc >= '{args.since}'"
    df = pd.read_sql_query(
        f"""
        SELECT p.fa_flight_id, p.proba_delay, p.predicted_delay,
               p.predicted_at_utc, f.scheduled_out_utc,
               a.arr_delay_min, a.cancelled, a.diverted
        FROM predictions p
        JOIN actuals a ON a.fa_flight_id = p.fa_flight_id
        JOIN flights  f ON f.fa_flight_id = p.fa_flight_id
        WHERE a.cancelled = 0
          AND a.diverted  = 0
          AND a.arr_delay_min IS NOT NULL
          AND f.scheduled_out_utc IS NOT NULL
          AND p.predicted_at_utc <= f.scheduled_out_utc
          {since_clause}
          {strategy_clause}
        """,
        conn,
    )

    # Una predicción por vuelo: la última anterior a la salida programada.
    #
    # Antes esto ordenaba por `proba_delay`, no por fecha: el comentario decía
    # "la más reciente" pero `.last()` se quedaba con la predicción de
    # PROBABILIDAD MÁS ALTA de cada vuelo. Y `predicted_at_utc` ni siquiera
    # estaba en el SELECT, así que ordenar por tiempo era imposible.
    #
    # No es un detalle. Medido el 27/09 sobre 8.372 vuelos de ATL, las dos
    # selecciones eligen distinto en el 86% de los casos, y la probabilidad
    # media pasa de 9,6% (la última) a 18,0% (la máxima). O sea que el
    # calibrador se ajustaba contra una distribución con casi el doble de
    # media que la que después ve en producción, y aprendía a encoger.
    #
    # Eso explica el artefacto `4year_v9_recal` que nadie activó: simulado
    # sobre los mismos datos, empuja la probabilidad media a 4,5% cuando la
    # tasa real es 12,7%, con Brier 0,1106 contra 0,1094 sin recalibrar. Era
    # peor que no hacer nada.
    #
    # La última predicción anterior a la salida es la que operaciones tiene en
    # la mano al decidir, así que es contra esa que hay que calibrar.
    df = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()

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
    # Isotonic, no sigmoid.
    #
    # Acá decía: "Usamos sigmoid (Platt scaling) en lugar de isotonic: solo 2
    # parámetros, no colapsa rankings ni destruye AUC con muestras limitadas
    # (<10k). Isotonic puede overfittear con pocas muestras". El razonamiento
    # es plausible y la medición dice lo contrario. `evaluar_recalibrado.py`,
    # caminando 9 cortes sobre 16.483 vuelos de prueba fuera de muestra:
    #
    #     estrategia    Brier     AUC     ECE   media   real
    #     cruda        0.0914  0.7289  0.0515    5.9%  10.9%
    #     isotonic     0.0882  0.7240  0.0152    9.6%  10.9%
    #     sigmoid      0.0948  0.6539  0.0208    9.2%  10.9%
    #
    # Isotonic le gana a la cruda en 9 de 9 cortes; sigmoid en 2 de 9. Y es
    # sigmoid el que destroza el AUC —0,654 contra 0,729—, no isotonic, que
    # pierde 0,005. Con 12.413 vuelos asentados no estamos en el régimen de
    # pocas muestras que motivaba la elección.
    #
    # Medido también: ajustar sobre ventana móvil de 3, 5 o 7 días empata con
    # usar todo el histórico (0,0882 contra 0,0882). No vale la complejidad;
    # la retención de `predictions` ya acota el histórico a 14 días.
    live_cal = fit_calibrator(y_proba_live, y_true, method=args.method)
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
    #
    # El cuantil directo no sirve con isotonic. El calibrador mapea tramos
    # enteros de score al mismo valor, así que el cuantil cae DENTRO de un
    # bloque de probabilidades idénticas y `>=` se lleva el bloque completo.
    # Medido: cuantil al 0,78 sobre 12.413 vuelos daba threshold 0,0886 y una
    # tasa positiva de 34,5% con objetivo 22%.
    #
    # Se elige el corte entre los valores efectivamente presentes que deje la
    # tasa positiva más cerca del objetivo.
    new_threshold = elegir_threshold(y_proba_recal, args.target_pos_rate)
    actual_pos_rate = (y_proba_recal >= new_threshold).mean()
    print(f"\nNuevo threshold (target_pos_rate={args.target_pos_rate}): {new_threshold:.4f}")
    print(f"Pos rate real con nuevo threshold: {actual_pos_rate:.3f}")

    # ---- 5b. control fuera de muestra ----
    #
    # Las métricas de arriba son EN MUESTRA: el calibrador vio las respuestas
    # con que se lo juzga, así que el Brier no puede empeorar y el ECE del
    # isotónico da exactamente 0,0000. Fue por esa puerta que pasó
    # `4year_v9_recal`, el artefacto que nadie activó porque en producción
    # empujaba la media a 4,5% contra una tasa real de 12,7%.
    #
    # Se camina sobre varios cortes en vez de reservar un holdout único: el
    # veredicto de un holdout cambia de signo según cuántos días se reserven,
    # porque el recalibrado ayuda los días con demoras y casi nada los calmos.
    # Ver `ontimeai/validacion_calibrado.py`.
    cortes: list = []
    resumen_control = "omitido (--sin-control)"
    if not args.sin_control:
        cortes = caminar_cortes(
            df["scheduled_out_utc"], y_true, y_proba_live,
            metodo=args.method, horizonte_dias=args.horizonte_dias,
            min_ajuste=args.min_n,
        )
        ok, resumen = veredicto(cortes, minimo_ganados=args.minimo_ganados)
        resumen_control = resumen
        print(f"\n--- Control fuera de muestra ({len(cortes)} cortes) ---")
        for c in cortes:
            marca = "mejora" if c.mejora else "PEOR"
            print(f"  {c.fecha:%d/%m}  ajuste {c.n_ajuste:>6,}  prueba {c.n_prueba:>5,}  "
                  f"Brier {c.brier_cruda:.4f} -> {c.brier_recal:.4f}  {marca}")
        print(f"  {resumen}")
        if not ok:
            print("  VEREDICTO: no se guarda.")
            if not args.force:
                print("  Usá --force si querés guardarlo igual.")
                return 1
            print("  --force: se guarda de todos modos.")
        else:
            print("  VEREDICTO: mejora sobre datos no vistos.")

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
        "method": args.method,
        # Lo de arriba es EN MUESTRA y por sí solo no dice nada: el isotónico
        # llega a ECE 0,0000 contra los datos con que se ajustó. Esta sección
        # es la que hay que mirar para decidir si activarlo.
        "fuera_de_muestra": {
            "resumen": resumen_control,
            "n_cortes": len(cortes),
            "cortes_ganados": sum(c.mejora for c in cortes),
            "brier_cruda": round(float(np.mean([c.brier_cruda for c in cortes])), 4)
                           if cortes else None,
            "brier_recal": round(float(np.mean([c.brier_recal for c in cortes])), 4)
                           if cortes else None,
            "detalle": [
                {"corte": f"{c.fecha:%Y-%m-%d}", "n_ajuste": c.n_ajuste,
                 "n_prueba": c.n_prueba, "brier_cruda": round(c.brier_cruda, 4),
                 "brier_recal": round(c.brier_recal, 4)}
                for c in cortes
            ],
        },
    }
    (out_dir / "recal_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nArtifact guardado en: {out_dir}")
    print(f"  model.lgb  → hard link a {src_model}")
    print(f"  meta.joblib → StackedCalibrator + threshold={new_threshold:.4f}")
    print(f"  recal_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
