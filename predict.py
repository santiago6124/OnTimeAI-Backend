"""CLI entry point: score flights with a saved OnTimeAI artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ontimeai.config import ARTIFACTS_DIR, ARR_DELAY_COL
from ontimeai.data import drop_leaky_target_columns, filter_valid_flights
from ontimeai.features import (
    add_absorb_score,
    add_congestion_features,
    add_cyclical_features,
    add_holiday_features,
    add_weather_interactions,
    apply_categorical_mapping,
    normalize_weather_flags,
)
from ontimeai.lineage import (
    add_carrier_day_lag,
    add_carrier_rolling_features,
    add_dest_rolling_features,
    add_origin_day_lag,
    add_origin_rolling_features,
    add_tail_lineage_features,
)
from ontimeai.lineage_fallback import apply_lineage_fallback, load_lookups
from ontimeai.model import load_artifact, predict_label, predict_proba

_PAGERANK_PATH = Path(__file__).resolve().parent / "artifacts" / "airport_pagerank.json"


def prepare_inference_frame(
    df_raw: pd.DataFrame,
    feature_cols: list[str],
    cat_mapping: dict[str, list],
    fallback_lookup: dict | None = None,
) -> pd.DataFrame:
    df = df_raw.copy()
    has_history = (
        ARR_DELAY_COL in df.columns
        and "EVENT_ORIGIN_UTC" in df.columns
        and "EVENT_DEST_UTC" in df.columns
    )

    if has_history:
        df = add_tail_lineage_features(df)
        df = add_carrier_day_lag(df)
        df = add_origin_day_lag(df)
        df = add_carrier_rolling_features(df)
        df = add_origin_rolling_features(df)
        df = add_dest_rolling_features(df)
        df = add_absorb_score(df)
        df = drop_leaky_target_columns(df)

    if fallback_lookup is not None:
        df = apply_lineage_fallback(df, fallback_lookup)

    # v8 features — silently no-op if not in feature_cols
    try:
        from feature_engineering_v7.v8_features import add_tail_delay_decay, add_pagerank_features
        if "TAIL_DELAY_DECAY" in feature_cols:
            df = add_tail_delay_decay(df)
        if "ORIGIN_PAGERANK" in feature_cols or "DEST_PAGERANK" in feature_cols:
            df = add_pagerank_features(df, lookup_path=_PAGERANK_PATH)
    except Exception:
        pass

    # v9: AIRCRAFT_FAMILY
    if "AIRCRAFT_FAMILY" in feature_cols:
        try:
            from feature_engineering_v7.aircraft_type import add_aircraft_family
            _lookup_path = Path(__file__).resolve().parent / "artifacts" / "tail_to_aircraft_family.json"
            df = add_aircraft_family(df, lookup_path=str(_lookup_path))
        except Exception:
            df["AIRCRAFT_FAMILY"] = "OTHER"

    df = normalize_weather_flags(df)
    df = add_holiday_features(df)
    df = add_cyclical_features(df)
    df = add_congestion_features(df)
    df = add_weather_interactions(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = pd.NA
    X = df[feature_cols].copy()
    return apply_categorical_mapping(X, cat_mapping)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score flights using a trained OnTimeAI artifact.")
    p.add_argument("--artifact", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--input", type=Path, required=True, help="CSV of flights to score.")
    p.add_argument("--output", type=Path, default=Path("predictions.csv"))
    p.add_argument(
        "--fallback-lookup",
        type=Path,
        default=ARTIFACTS_DIR / "lineage_fallback.joblib",
        help="Path to cold-deck lookup joblib (built by build_lineage_fallback.py). "
             "Skipped silently if file does not exist.",
    )
    args = p.parse_args(argv)

    meta = load_artifact(args.artifact)
    if Path(args.input).suffix == ".parquet":
        df_raw = pd.read_parquet(args.input)
        df_raw["FL_DATE"] = pd.to_datetime(df_raw["FL_DATE"], errors="coerce")
    else:
        df_raw = pd.read_csv(args.input, parse_dates=["FL_DATE"], low_memory=False)

    fallback = load_lookups(args.fallback_lookup) if args.fallback_lookup.exists() else None
    if fallback is not None:
        print(f"Loaded cold-deck fallback from {args.fallback_lookup}")
    X = prepare_inference_frame(
        df_raw, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )

    proba = predict_proba(meta["booster"], X)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)
    if meta["target"] == "binary":
        pred = predict_label(proba, meta["threshold"], "binary")
        out = pd.DataFrame({"proba_delay": proba, "predicted_delay": pred})
    else:
        pred = predict_label(proba, meta["threshold"], "multiclass")
        proba_df = pd.DataFrame(proba, columns=[f"proba_class_{i}" for i in range(proba.shape[1])])
        out = pd.concat([proba_df, pd.Series(pred, name="predicted_class")], axis=1)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
