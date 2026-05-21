"""Feature parity audit: compare NaN rates between training data and live_data.db.

Loads the v9 model artifact to identify expected features, then:
  1. Samples the training parquet for NaN rates
  2. Queries live_data.db predictions for feature NaN rates (via inference frame rebuild)
  3. Flags features where live NaN rate > 3× training NaN rate

Usage:
    python3 scripts/feature_parity_audit.py
    python3 scripts/feature_parity_audit.py --out artifacts/feature_parity_report.json
    python3 scripts/feature_parity_audit.py --sample 100000  # subsample training data
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH
from ontimeai.model import load_artifact
from ontimeai.live import open_db, build_inference_frame


def training_nan_rates(feature_cols: list[str], sample_n: int = 100_000) -> dict[str, float]:
    """Compute NaN rate per feature from a sample of the training dataset.

    Uses the raw parquet (pre-feature-engineering) so we get a sense of what
    the pipeline sees before lineage/rolling features are computed.
    """
    print(f"Loading training data sample ({sample_n:,} rows) from {DATA_PATH.name}...")
    if DATA_PATH.suffix == ".parquet":
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)

    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    nan_rates = {}
    for col in feature_cols:
        if col in df.columns:
            nan_rates[col] = float(df[col].isna().mean())
        else:
            # Feature was engineered during pipeline — not in raw data
            nan_rates[col] = None  # will be computed during prepare_dataset
    return nan_rates


def live_nan_rates(feature_cols: list[str], meta: dict, max_flights: int = 2000) -> dict[str, float]:
    """Compute NaN rate per feature from the live inference pipeline.

    Rebuilds inference frames for a sample of recently predicted flights
    to see what the model actually receives in production.
    """
    conn = open_db()

    # Get a sample of recently predicted flight IDs
    rows = conn.execute(
        """SELECT DISTINCT p.fa_flight_id
           FROM predictions p
           JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           ORDER BY p.predicted_at_utc DESC
           LIMIT ?""",
        (max_flights,),
    ).fetchall()
    fa_ids = [r[0] for r in rows]
    print(f"Rebuilding inference frame for {len(fa_ids)} live flights...")

    if not fa_ids:
        return {col: None for col in feature_cols}

    df = build_inference_frame(conn, fa_ids, history_days=7)
    if df.empty:
        return {col: None for col in feature_cols}

    # Only look at target rows (not history)
    target_mask = df["fa_flight_id"].isin(fa_ids) & df["ARR_DELAY"].isna()
    df_target = df[target_mask]
    print(f"  target rows: {len(df_target)}")

    # Now run the prediction prep to get the actual feature matrix
    from predict import prepare_inference_frame
    from ontimeai.lineage_fallback import load_lookups

    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = None
    if fallback_path.exists():
        try:
            fallback = load_lookups(fallback_path)
        except Exception as e:
            print(f"  ⚠ fallback load failed (pickle/pandas version mismatch): {e}")
            print("  proceeding without lineage fallback — raw NaN rates visible")

    # Build features WITHOUT fallback to see raw NaN rates
    X_raw = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=None,
    )
    # Build features WITH fallback to see what model actually gets
    X_filled = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )

    # Subset to target rows
    target_idx = df.index[target_mask]
    X_raw_target = X_raw.loc[target_idx] if not target_idx.empty else X_raw.head(0)
    X_filled_target = X_filled.loc[target_idx] if not target_idx.empty else X_filled.head(0)

    nan_rates_raw = {}
    nan_rates_filled = {}
    for col in feature_cols:
        if col in X_raw_target.columns:
            val = X_raw_target[col]
            # For categoricals, check for NA/empty
            if hasattr(val, "cat"):
                nan_rates_raw[col] = float(val.isna().mean())
            else:
                nan_rates_raw[col] = float(pd.to_numeric(val, errors="coerce").isna().mean())
        else:
            nan_rates_raw[col] = 1.0

        if col in X_filled_target.columns:
            val = X_filled_target[col]
            if hasattr(val, "cat"):
                nan_rates_filled[col] = float(val.isna().mean())
            else:
                nan_rates_filled[col] = float(pd.to_numeric(val, errors="coerce").isna().mean())
        else:
            nan_rates_filled[col] = 1.0

    return nan_rates_raw, nan_rates_filled


def main() -> int:
    p = argparse.ArgumentParser(description="Feature parity audit: train vs live NaN rates")
    p.add_argument("--artifact", type=Path, default=ARTIFACTS_DIR / "4year_v9")
    p.add_argument("--sample", type=int, default=100_000,
                   help="Number of training rows to sample (default 100K)")
    p.add_argument("--max-flights", type=int, default=1000,
                   help="Max live flights to rebuild features for")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-training", action="store_true",
                   help="Skip training data scan (only compute live NaN rates)")
    args = p.parse_args()

    meta = load_artifact(args.artifact)
    feature_cols = meta["feature_cols"]
    print(f"Model has {len(feature_cols)} features\n")

    # Training NaN rates
    train_nans = {}
    if not args.skip_training:
        train_nans = training_nan_rates(feature_cols, args.sample)
    else:
        train_nans = {col: None for col in feature_cols}

    # Live NaN rates
    result = live_nan_rates(feature_cols, meta, args.max_flights)
    if isinstance(result, tuple):
        live_nans_raw, live_nans_filled = result
    else:
        live_nans_raw = result
        live_nans_filled = result

    # --- Report ---
    print("\n" + "=" * 90)
    print("FEATURE PARITY AUDIT")
    print("=" * 90)
    print(f"\n{'Feature':<35} {'Train NaN%':>10} {'Live Raw%':>10} {'Live Filled%':>12} {'Status':>10}")
    print("-" * 90)

    flagged = []
    for col in feature_cols:
        t = train_nans.get(col)
        lr = live_nans_raw.get(col)
        lf = live_nans_filled.get(col)

        t_str = f"{t:.1%}" if t is not None else "N/A"
        lr_str = f"{lr:.1%}" if lr is not None else "N/A"
        lf_str = f"{lf:.1%}" if lf is not None else "N/A"

        # Determine status
        if lr is not None and lr > 0.5:
            status = "🔴 BROKEN"
            flagged.append({"feature": col, "live_nan_raw": lr, "live_nan_filled": lf, "train_nan": t})
        elif lr is not None and t is not None and t > 0 and lr / max(t, 0.001) > 3:
            status = "🟡 DRIFT"
            flagged.append({"feature": col, "live_nan_raw": lr, "live_nan_filled": lf, "train_nan": t})
        elif lr is not None and lr > 0.1:
            status = "🟡 WARN"
        else:
            status = "✅ OK"

        print(f"{col:<35} {t_str:>10} {lr_str:>10} {lf_str:>12} {status:>10}")

    if flagged:
        print(f"\n⚠ {len(flagged)} features flagged:")
        for f in flagged:
            print(f"  - {f['feature']}: live_raw={f['live_nan_raw']:.1%}, "
                  f"filled={f['live_nan_filled']:.1%}, "
                  f"train={f['train_nan']:.1%}" if f['train_nan'] is not None
                  else f"  - {f['feature']}: live_raw={f['live_nan_raw']:.1%}, "
                  f"filled={f['live_nan_filled']:.1%}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact),
        "n_features": len(feature_cols),
        "n_flagged": len(flagged),
        "flagged_features": flagged,
        "per_feature": {
            col: {
                "train_nan": train_nans.get(col),
                "live_nan_raw": live_nans_raw.get(col),
                "live_nan_filled": live_nans_filled.get(col),
            }
            for col in feature_cols
        },
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nJSON report → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
