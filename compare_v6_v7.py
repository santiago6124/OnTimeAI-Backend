"""Head-to-head live comparison: v6_recal vs v7 on the same historical flights.

Runs both model artifacts through the full inference pipeline on every flight
in the DB that has a settled actual, then reports side-by-side metrics.
No DB writes.

Usage:
    python3 compare_v6_v7.py
    python3 compare_v6_v7.py --days 7 --batch-size 500
    python3 compare_v6_v7.py --v6-artifact artifacts/4year_v6_recal
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score,
    accuracy_score, precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.live import open_db, build_inference_frame, stable_id
from ontimeai.model import load_artifact, predict_label, predict_proba, quantile_threshold
from ontimeai.lineage_fallback import load_lookups
from ontimeai.config import ARTIFACTS_DIR
from predict import prepare_inference_frame


def _fix_flow_atl(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["ORIGIN"].eq("ATL") & ~df["DEST"].eq("ATL")
    df.loc[mask, "FLOW_ATL"] = "NON_ATL"
    df.loc[mask, "PAR_AIRPORT"] = ""
    return df


def _coerce_numeric(X: pd.DataFrame, cat_cols_set: set) -> pd.DataFrame:
    for c in X.columns:
        if c not in cat_cols_set and X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _run_model(meta, fallback, df: pd.DataFrame, target_mask, target_pos_rate: float):
    X = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )
    X = _coerce_numeric(X, set(meta.get("cat_cols", [])))
    proba = predict_proba(meta["booster"], X)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)

    target_proba = proba[target_mask.to_numpy()]
    if target_pos_rate > 0 and target_proba.size >= 5:
        threshold = quantile_threshold(target_proba, target_pos_rate)
    else:
        threshold = float(meta["threshold"])

    labels = predict_label(proba, threshold, "binary")
    return proba[target_mask.to_numpy()], labels[target_mask.to_numpy()], threshold


def _metrics(proba, pred, true, label: str, n: int):
    auc   = roc_auc_score(true, proba)
    brier = brier_score_loss(true, proba)
    f1    = f1_score(true, pred, zero_division=0)
    acc   = accuracy_score(true, pred)
    prec  = precision_score(true, pred, zero_division=0)
    rec   = recall_score(true, pred, zero_division=0)
    pos_t = true.mean()
    pos_p = pred.mean()
    print(f"\n=== {label} (n={n}) ===")
    print(f"  AUC:           {auc:.4f}")
    print(f"  Brier:         {brier:.4f}")
    print(f"  F1:            {f1:.4f}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")
    print(f"  pos_rate_true: {pos_t:.4f}  pos_rate_pred: {pos_p:.4f}")
    return {"auc": auc, "brier": brier, "f1": f1, "accuracy": acc,
            "precision": prec, "recall": rec,
            "pos_rate_true": pos_t, "pos_rate_pred": pos_p, "n": n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v6-artifact", default=str(ARTIFACTS_DIR / "4year_v6_recal"))
    p.add_argument("--v7-artifact", default=str(ARTIFACTS_DIR / "4year_v7_full"))
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=400)
    p.add_argument("--target-pos-rate", type=float, default=0.22)
    args = p.parse_args()

    print(f"v6 artifact: {args.v6_artifact}")
    print(f"v7 artifact: {args.v7_artifact}")
    print(f"Window:      last {args.days} days")

    meta_v6  = load_artifact(args.v6_artifact)
    meta_v7  = load_artifact(args.v7_artifact)
    fallback = load_lookups(PROJECT_ROOT / "artifacts" / "lineage_fallback.joblib")

    conn = open_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    rows = conn.execute(
        """SELECT DISTINCT f.fa_flight_id
           FROM flights f
           JOIN actuals a ON a.stable_id = f.stable_id
           WHERE f.scheduled_off_utc >= ?
             AND a.arr_delay_min IS NOT NULL
           ORDER BY f.scheduled_off_utc""",
        (cutoff,),
    ).fetchall()

    all_ids = [r[0] for r in rows]
    print(f"\nFlights with settled actuals: {len(all_ids)}")

    # Collect per-flight results
    v6_proba_all, v6_pred_all, true_all = [], [], []
    v7_proba_all, v7_pred_all           = [], []

    conn2 = open_db()

    for i in range(0, len(all_ids), args.batch_size):
        batch = all_ids[i: i + args.batch_size]
        print(f"  Batch {i//args.batch_size + 1}/{-(-len(all_ids)//args.batch_size)}: {len(batch)} flights...", end=" ", flush=True)

        df = build_inference_frame(conn, batch, history_days=7)
        if df.empty:
            print("empty")
            continue

        df = _fix_flow_atl(df)
        target_mask = df["fa_flight_id"].isin(batch) & df["ARR_DELAY"].isna()
        if target_mask.sum() == 0:
            print("no targets")
            continue

        # Ground truth from actuals table
        target_ids_in_batch = df.loc[target_mask, "fa_flight_id"].tolist()
        sid_map = {r["fa_flight_id"]: r for r in (dict(zip(
            ["fa_flight_id","arr_delay_min"],
            conn2.execute(
                "SELECT f.fa_flight_id, a.arr_delay_min FROM flights f "
                "JOIN actuals a ON a.stable_id = f.stable_id "
                f"WHERE f.fa_flight_id IN ({','.join('?'*len(target_ids_in_batch))})",
                target_ids_in_batch
            ).fetchone() or [None, None]
        )) for _ in [None])}  # noqa: this is replaced below

        actual_rows = conn2.execute(
            f"SELECT f.fa_flight_id, a.arr_delay_min FROM flights f "
            f"JOIN actuals a ON a.stable_id = f.stable_id "
            f"WHERE f.fa_flight_id IN ({','.join('?'*len(target_ids_in_batch))})",
            target_ids_in_batch
        ).fetchall()
        delay_map = {r[0]: r[1] for r in actual_rows}

        try:
            p6, l6, _ = _run_model(meta_v6, fallback, df, target_mask, args.target_pos_rate)
            p7, l7, _ = _run_model(meta_v7, fallback, df, target_mask, args.target_pos_rate)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        for k, fid in enumerate(df.loc[target_mask, "fa_flight_id"]):
            delay = delay_map.get(fid)
            if delay is None:
                continue
            true_all.append(1 if delay >= 15 else 0)
            v6_proba_all.append(float(p6[k]))
            v6_pred_all.append(int(l6[k]))
            v7_proba_all.append(float(p7[k]))
            v7_pred_all.append(int(l7[k]))

        print(f"OK (+{target_mask.sum()})")

    if not true_all:
        print("No matched predictions — exiting.")
        return

    true  = np.array(true_all)
    n     = len(true)

    print("\n" + "="*55)
    print("HEAD-TO-HEAD: v6_recal  vs  v7  (same flights)")
    print("="*55)
    m6 = _metrics(np.array(v6_proba_all), np.array(v6_pred_all), true, "v6_recal", n)
    m7 = _metrics(np.array(v7_proba_all), np.array(v7_pred_all), true, "v7_full",  n)

    print("\n--- Delta (v7 − v6) ---")
    for k in ("auc", "brier", "f1", "accuracy"):
        delta = m7[k] - m6[k]
        sign  = "+" if delta >= 0 else ""
        better = ("↑" if (k != "brier" and delta > 0) else ("↓" if (k == "brier" and delta < 0) else "↔"))
        print(f"  {k:10s}: {sign}{delta:+.4f}  {better}")


if __name__ == "__main__":
    main()
