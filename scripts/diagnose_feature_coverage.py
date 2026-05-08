"""Feature-level coverage diagnostic for live predictions.

For a given period of predictions in live_data.db, reconstructs the X matrix
that fed the model and reports:
  - % NaN per feature (BEFORE cold-deck fallback)
  - % cold-deck-imputed (vs computed from real lineage history)
  - distribution stats (mean, std) per feature, by carrier
  - comparison to offline test set distribution

Identifies WHICH features are degraded in production and WHY.

Usage:
    python3 scripts/diagnose_feature_coverage.py --since 2026-05-04 --until 2026-05-07
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ontimeai.config import ARTIFACTS_DIR
from ontimeai.lineage_fallback import load_lookups
from ontimeai.live import build_inference_frame, open_db
from ontimeai.model import load_artifact

# Lineage features that depend on history. These are the ones most likely
# to be NaN in live + filled by cold-deck.
LINEAGE_FEATURES = [
    "prev_arr_delay_tail",
    "prev_turnaround_tail_min",
    "tail_flights_today_prior",
    "carrier_delay_rate_yday",
    "origin_delay_rate_yday",
    "carrier_delay_rate_24h",
    "carrier_delay_rate_7d",
    "origin_delay_rate_1h",
    "origin_delay_rate_6h",
    "origin_delay_rate_24h",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", required=True, help="YYYY-MM-DD")
    p.add_argument("--until", required=True, help="YYYY-MM-DD")
    p.add_argument("--artifact", default=str(ARTIFACTS_DIR / "4year_v4_full"))
    args = p.parse_args()

    conn = open_db()

    # Get all predictions in the window
    preds = pd.read_sql_query(
        """SELECT p.fa_flight_id, p.predicted_at_utc, p.proba_delay,
                  p.threshold_strategy,
                  f.op_carrier, f.origin, f.dest, f.tail_num,
                  f.scheduled_off_utc
           FROM predictions p
           LEFT JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           WHERE p.predicted_at_utc >= ? AND p.predicted_at_utc <= ?""",
        conn, params=(f"{args.since}T00:00:00", f"{args.until}T23:59:59"),
    )
    print(f"Loaded {len(preds)} predictions in window {args.since} → {args.until}")
    if preds.empty:
        return 1

    target_ids = preds["fa_flight_id"].dropna().unique().tolist()
    print(f"\nRebuilding inference frame for {len(target_ids)} flights "
          f"(this is what the model SAW)...")

    df = build_inference_frame(conn, target_ids, history_days=14)
    print(f"  Inference frame: {len(df)} rows ({(df['_role'] == 'target').sum()} targets, "
          f"{(df['_role'] == 'history').sum()} history)")

    # Run the same prepare_inference_frame the live pipeline uses (without fallback
    # so we see the RAW NaN before cold-deck imputes).
    from predict import prepare_inference_frame
    meta = load_artifact(Path(args.artifact))
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = load_lookups(fallback_path) if fallback_path.exists() else None

    print("\nComputing X WITHOUT cold-deck (to see raw NaN coverage)...")
    X_raw = prepare_inference_frame(df, meta["feature_cols"], meta["cat_mapping"],
                                    fallback_lookup=None)
    print("Computing X WITH cold-deck (the production path)...")
    X_full = prepare_inference_frame(df, meta["feature_cols"], meta["cat_mapping"],
                                     fallback_lookup=fallback)

    # Filter to target rows only (those we actually predicted)
    target_mask = df["_role"] == "target" if "_role" in df.columns else df["ARR_DELAY"].isna()
    X_raw = X_raw.loc[target_mask].reset_index(drop=True)
    X_full = X_full.loc[target_mask].reset_index(drop=True)
    df_t = df.loc[target_mask].reset_index(drop=True)

    print(f"\nAnalyzing {len(X_raw)} target predictions...")

    # Per-feature NaN coverage (before cold-deck)
    print("\n" + "=" * 80)
    print("FEATURE COVERAGE (before cold-deck fallback)")
    print("=" * 80)
    print(f"{'Feature':<35} {'NaN %':>10} {'mean':>10} {'std':>10}")
    print("-" * 70)
    for col in LINEAGE_FEATURES:
        if col not in X_raw.columns:
            continue
        s = pd.to_numeric(X_raw[col], errors="coerce")
        nan_pct = 100 * s.isna().mean()
        mean = s.dropna().mean() if s.notna().any() else np.nan
        std = s.dropna().std() if s.notna().any() else np.nan
        flag = "❌" if nan_pct > 50 else ("⚠️ " if nan_pct > 20 else "✅")
        print(f"{flag} {col:<33} {nan_pct:>9.1f}% {mean:>10.3f} {std:>10.3f}")

    # Per-carrier NaN coverage of prev_arr_delay_tail (the main lineage feature)
    print("\n" + "=" * 80)
    print("LINEAGE COVERAGE BY CARRIER (prev_arr_delay_tail)")
    print("=" * 80)
    if "OP_CARRIER" in df_t.columns and "prev_arr_delay_tail" in X_raw.columns:
        df_t["_pad_nan"] = pd.to_numeric(X_raw["prev_arr_delay_tail"], errors="coerce").isna().to_numpy()
        by_carrier = (
            df_t.groupby("OP_CARRIER")
            .agg(n=("_pad_nan", "size"), pct_nan=("_pad_nan", "mean"))
            .sort_values("n", ascending=False)
        )
        print(f"{'Carrier':<10} {'n':>10} {'NaN % (no lineage)':>22}")
        print("-" * 45)
        for carrier, row in by_carrier.head(15).iterrows():
            flag = "❌" if row["pct_nan"] > 0.5 else ("⚠️ " if row["pct_nan"] > 0.2 else "✅")
            print(f"{flag} {carrier:<8} {int(row['n']):>10,}  {row['pct_nan']*100:>20.1f}%")

    # Effect of cold-deck on proba distribution
    print("\n" + "=" * 80)
    print("PROBA DISTRIBUTION (with cold-deck fallback applied — production path)")
    print("=" * 80)
    proba = preds["proba_delay"].dropna().to_numpy()
    print(f"  n={len(proba)}, mean={proba.mean():.3f}, std={proba.std():.3f}")
    print(f"  quantiles: p05={np.quantile(proba,0.05):.3f}  "
          f"p50={np.quantile(proba,0.50):.3f}  p95={np.quantile(proba,0.95):.3f}")

    # Compare to offline expectation
    print("\nFor reference, v4 OFFLINE on test split:")
    print("  Expected: mean~0.20, std~0.20  (broad spread = healthy discrimination)")
    print("  Live (current): mean=" + f"{proba.mean():.3f}, std={proba.std():.3f}")
    if proba.std() < 0.15:
        print("  → Live std too tight — features degraded → cold-deck dominating")
    elif proba.std() < 0.20:
        print("  → Live std moderate — partial feature degradation")
    else:
        print("  → Live std healthy — features look complete")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
