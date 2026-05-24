"""One-shot local prediction script.

Usa la DB en `tmp/live_data.now.db` (descargada de GCS) para predecir los vuelos
KATL más próximos. Diagnóstico del bug LightGBM SIGSEGV — corre el predict en
dos etapas (sin fallback / con fallback) para identificar la fuente.

Uso:
    python scripts/predict_now.py
    python scripts/predict_now.py --hours 6 --limit 20
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "tmp" / "live_data.now.db"


def select_target_ids(conn: sqlite3.Connection, *, hours: int, limit: int) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT fa_flight_id
        FROM flights
        WHERE (origin='ATL' OR dest='ATL')
          AND scheduled_out_utc IS NOT NULL
          AND datetime(scheduled_out_utc) >= datetime('now')
          AND datetime(scheduled_out_utc) < datetime('now', '+{hours} hours')
          AND tail_num IS NOT NULL
        ORDER BY scheduled_out_utc
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--artifact", default="artifacts/4year_v9")
    parser.add_argument("--hours", type=int, default=6,
                        help="Buscar vuelos futuros con scheduled_out en próximas N horas")
    parser.add_argument("--limit", type=int, default=10, help="Max target flights")
    parser.add_argument("--with-fallback", action="store_true",
                        help="Usar lineage_fallback.joblib (puede ser fuente del SIGSEGV)")
    args = parser.parse_args()

    print(f"== predict_now.py (db={args.db}) ==")

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        return 2

    print("\n[1/6] imports...")
    from ontimeai.live import build_inference_frame
    from ontimeai.model import load_artifact, predict_proba, predict_label, quantile_threshold
    from predict import prepare_inference_frame
    print("       ok")

    print(f"\n[2/6] open db + pick targets (next {args.hours}h)...")
    conn = sqlite3.connect(str(db_path))
    target_ids = select_target_ids(conn, hours=args.hours, limit=args.limit)
    print(f"       selected {len(target_ids)} target flights: {target_ids}")
    if not target_ids:
        print("no target flights — try --hours 12 or check DB freshness", file=sys.stderr)
        return 0

    print("\n[3/6] load LightGBM artifact...")
    meta = load_artifact(Path(args.artifact))
    print(f"       target={meta['target']} threshold={meta['threshold']:.4f} "
          f"n_features={len(meta['feature_cols'])} cal={type(meta.get('calibrator')).__name__}")

    fallback = None
    if args.with_fallback:
        print("\n[3b] load lineage_fallback...")
        from ontimeai.lineage_fallback import load_lookups
        fb_path = Path("artifacts/lineage_fallback.joblib")
        try:
            fallback = load_lookups(fb_path)
            print(f"       fallback loaded from {fb_path}")
        except Exception as exc:
            print(f"       WARN: fallback failed to load: {exc}")
    else:
        print("\n[3b] SKIP fallback (set --with-fallback to enable)")

    print("\n[4/6] build_inference_frame (history_days=7)...")
    df = build_inference_frame(conn, target_ids, history_days=7)
    if df.empty:
        print("       inference frame empty", file=sys.stderr)
        return 1
    target_mask = df["fa_flight_id"].isin(target_ids) & df["ARR_DELAY"].isna()
    print(f"       {target_mask.sum()} target rows | {(~target_mask).sum()} history rows")

    print("\n[5/6] prepare_inference_frame + predict_proba...")
    try:
        X = prepare_inference_frame(
            df, meta["feature_cols"], meta["cat_mapping"],
            fallback_lookup=fallback,
        )
        print(f"       X shape: {X.shape}")
        # NaN check (acota >60% NaN filter — same as live_pull.py)
        cat_cols_set = set(meta.get("cat_cols", []))
        import pandas as pd
        X_check = X.copy()
        for c in X_check.columns:
            if c not in cat_cols_set and X_check[c].dtype == object:
                X_check[c] = pd.to_numeric(X_check[c], errors="coerce")
        for c in X.columns:
            if c in cat_cols_set:
                continue
            if X[c].dtype == object:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        nan_per_target = X_check.loc[df.index[target_mask]].isna().mean(axis=1)
        bad = nan_per_target[nan_per_target > 0.6]
        if not bad.empty:
            print(f"       WARN: {len(bad)} target rows con >60% NaN — filtered out")
            target_mask = target_mask & (~df.index.isin(bad.index))

        proba = predict_proba(meta["booster"], X)
        print(f"       proba shape: {proba.shape} mean={proba.mean():.4f}")
        if meta.get("calibrator") is not None and meta["target"] == "binary":
            proba = meta["calibrator"].transform(proba)
            print(f"       after calibrator mean={proba.mean():.4f}")
    except Exception:
        print("       *** PREDICT FAILED ***")
        traceback.print_exc()
        return 3

    print("\n[6/6] report target predictions:")
    import numpy as np
    target_proba = proba[target_mask.to_numpy()]
    if target_proba.size >= 5:
        threshold = quantile_threshold(target_proba, 0.22)
        strategy = "quantile@0.22"
    else:
        threshold = float(meta["threshold"])
        strategy = "artifact"
    print(f"       threshold={threshold:.4f} ({strategy})  "
          f"pred_pos_rate={(target_proba >= threshold).mean():.3f}")

    labels = predict_label(proba, threshold, "binary")
    print()
    print(f"{'fa_flight_id':<14} {'route':<10} {'carrier':<3} {'tail':<8} "
          f"{'sched_out_utc':<22} {'proba':>7} {'lbl':>4}")
    print("-" * 80)
    rows_out = []
    for idx in df.index[target_mask]:
        fid = df.loc[idx, "fa_flight_id"]
        rec = (fid,
               f"{df.loc[idx,'ORIGIN']}→{df.loc[idx,'DEST']}",
               str(df.loc[idx, "OP_CARRIER"]),
               str(df.loc[idx, "TAIL_NUM"]),
               str(df.loc[idx, "EVENT_ORIGIN_UTC"])[:19],
               float(proba[idx]),
               int(labels[idx]))
        rows_out.append(rec)
    rows_out.sort(key=lambda r: r[4])
    for r in rows_out:
        print(f"{r[0]:<14} {r[1]:<10} {r[2]:<3} {r[3]:<8} "
              f"{r[4]:<22} {r[5]:>7.4f} {r[6]:>4}")
    print()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
