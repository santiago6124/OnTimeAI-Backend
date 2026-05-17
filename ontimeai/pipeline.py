"""End-to-end training pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ontimeai.config import MULTICLASS_LABELS, TARGET_COL, TrainConfig
from ontimeai.data import drop_leaky_target_columns, filter_valid_flights, load_master
from ontimeai.evaluation import binary_metrics, confusion_df, multiclass_metrics
from ontimeai.calibration import Calibrator, fit_calibrator
from ontimeai.features import (
    add_absorb_score,
    add_congestion_features,
    add_cyclical_features,
    add_holiday_features,
    add_weather_interactions,
    build_feature_matrix,
    build_target,
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
from ontimeai.model import (
    predict_label,
    predict_proba,
    save_artifact,
    train_booster,
    tune_threshold,
)
from ontimeai.split import temporal_split


@dataclass
class PipelineResult:
    booster: Any
    threshold: float
    feature_cols: list[str]
    cat_cols: list[str]
    cat_mapping: dict[str, list]
    metrics: dict[str, Any]
    artifact_dir: Path | None
    calibrator: Calibrator | None = None


def prepare_dataset(
    df_raw: pd.DataFrame,
    cfg: TrainConfig,
    *,
    already_filtered: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    # When already_filtered=True the caller loaded via load_master(valid_only=True),
    # so cancelled/diverted/null-target rows were excluded at scan time. Skipping
    # filter_valid_flights avoids an 8.5 GB copy of the full dataset.
    if already_filtered:
        df = df_raw
    else:
        df = filter_valid_flights(df_raw)
    y = build_target(df, cfg.target, cfg.delay_threshold_min)
    # Lineage features must run BEFORE drop_leaky because they consume ARR_DELAY
    # + EVENT_*_UTC. They emit only leakage-safe summary columns.
    if cfg.use_tail_lineage:
        df = add_tail_lineage_features(df)
    if cfg.use_carrier_lag:
        df = add_carrier_day_lag(df)
    if cfg.use_origin_lag:
        df = add_origin_day_lag(df)
    if cfg.use_carrier_rolling:
        df = add_carrier_rolling_features(df)
    if cfg.use_origin_rolling:
        df = add_origin_rolling_features(df)
    if cfg.use_dest_rolling:
        df = add_dest_rolling_features(df)
    if cfg.use_absorb_score:
        df = add_absorb_score(df)
    df = drop_leaky_target_columns(df)

    # v8 features: TAIL_DELAY_DECAY + airport PageRank (gracefully no-op if unavailable)
    try:
        from feature_engineering_v7.v8_features import add_tail_delay_decay, add_pagerank_features
        from ontimeai.config import ARTIFACTS_DIR
        df = add_tail_delay_decay(df)
        df = add_pagerank_features(df, lookup_path=ARTIFACTS_DIR / "airport_pagerank.json")
    except Exception:
        pass

    # v9: AIRCRAFT_FAMILY — replaces TAIL_NUM, stable ICAO type family per aircraft
    try:
        from feature_engineering_v7.aircraft_type import add_aircraft_family
        from ontimeai.config import ARTIFACTS_DIR
        df = add_aircraft_family(
            df,
            lookup_path=str(ARTIFACTS_DIR / "tail_to_aircraft_family.json"),
        )
    except Exception:
        df["AIRCRAFT_FAMILY"] = "OTHER"

    df = normalize_weather_flags(df)
    if cfg.use_holiday_features:
        df = add_holiday_features(df)
    df = add_cyclical_features(df)
    df = add_congestion_features(df, cfg.congestion_window_min)
    df = add_weather_interactions(df)
    df[TARGET_COL] = y.to_numpy()
    return df, y


def run_training(
    cfg: TrainConfig,
    *,
    df_raw: pd.DataFrame | None = None,
    save_artifacts: bool = True,
) -> PipelineResult:
    already_filtered = False
    if df_raw is None:
        # Load with scan-time filter so cancelled/diverted/null-target rows are
        # never held in memory alongside the filtered copy.
        df_raw = load_master(valid_only=True)
        already_filtered = True

    df_ready, _ = prepare_dataset(df_raw, cfg, already_filtered=already_filtered)
    # df_ready IS df_raw when already_filtered=True (same object, in-place ops).
    # Remove the df_raw name so Python can GC the object once df_ready is also gone.
    del df_raw

    train_idx, val_idx, test_idx = temporal_split(
        df_ready, train_frac=cfg.train_frac, val_frac=cfg.val_frac
    )

    # Extract y before build_feature_matrix drops TARGET_COL from df_ready.
    y_full = df_ready[TARGET_COL].to_numpy()
    X_full, cat_cols, cat_mapping = build_feature_matrix(df_ready)
    # df_ready no longer needed — X_full contains all feature columns.
    del df_ready

    feature_col_names = list(X_full.columns)
    X_train = X_full.iloc[train_idx]
    X_val = X_full.iloc[val_idx]
    X_test = X_full.iloc[test_idx]
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]
    del X_full

    booster = train_booster(X_train, y_train, X_val, y_val, cat_cols, cfg)

    proba_val = predict_proba(booster, X_val)
    proba_test = predict_proba(booster, X_test)

    calibrator: Calibrator | None = None
    if cfg.target == "binary" and cfg.calibration:
        calibrator = fit_calibrator(proba_val, y_val, method=cfg.calibration)
        proba_val = calibrator.transform(proba_val)
        proba_test = calibrator.transform(proba_test)

    if cfg.target == "binary":
        threshold = tune_threshold(proba_val, y_val, metric=cfg.tune_threshold_metric)
        y_pred_val = predict_label(proba_val, threshold, "binary")
        y_pred_test = predict_label(proba_test, threshold, "binary")
        val_m = binary_metrics(y_val, y_pred_val, proba_val)
        test_m = binary_metrics(y_test, y_pred_test, proba_test)
        cm_labels = [0, 1]
    else:
        threshold = 0.5
        y_pred_val = predict_label(proba_val, threshold, "multiclass")
        y_pred_test = predict_label(proba_test, threshold, "multiclass")
        val_m = multiclass_metrics(y_val, y_pred_val, proba_val, list(MULTICLASS_LABELS))
        test_m = multiclass_metrics(y_test, y_pred_test, proba_test, list(MULTICLASS_LABELS))
        cm_labels = list(MULTICLASS_LABELS)

    test_cm = confusion_df(y_test, y_pred_test, labels=cm_labels)

    metrics = {
        "target": cfg.target,
        "threshold": threshold,
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "calibration": cfg.calibration,
        "val": val_m,
        "test": test_m,
        "confusion_test": test_cm.to_dict(),
        "feature_cols": feature_col_names,
    }

    artifact_dir = None
    if save_artifacts:
        artifact_dir = save_artifact(
            booster,
            threshold=threshold,
            feature_cols=feature_col_names,
            cat_cols=cat_cols,
            cat_mapping=cat_mapping,
            target=cfg.target,
            metadata=metrics,
            out_dir=cfg.artifacts_dir,
            calibrator=calibrator,
        )

    return PipelineResult(
        booster=booster,
        threshold=threshold,
        feature_cols=feature_col_names,
        cat_cols=cat_cols,
        cat_mapping=cat_mapping,
        metrics=metrics,
        artifact_dir=artifact_dir,
        calibrator=calibrator,
    )
