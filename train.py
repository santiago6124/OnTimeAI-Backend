"""CLI entry point: train the OnTimeAI LightGBM delay classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH, TrainConfig
from ontimeai.data import load_master
from ontimeai.pipeline import run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the OnTimeAI delay classifier.")
    p.add_argument("--target", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--threshold-min", type=float, default=15.0)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    p.add_argument("--num-boost-round", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--subsample", type=int, default=0, help="Subsample rows for smoke runs (0=all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--best-params", type=Path, default=None, help="Load best lgb_params from Optuna JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        target=args.target,
        delay_threshold_min=args.threshold_min,
        artifacts_dir=args.artifacts,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        balance_classes=not args.no_balance,
        random_state=args.seed,
    )

    if args.best_params is not None:
        best = json.loads(args.best_params.read_text())
        params = dict(cfg.lgb_params)
        params.update(best.get("best_params", best))
        cfg.lgb_params = params
        print(f"Loaded tuned params from {args.best_params}")

    df_raw = None
    if args.data != DATA_PATH or args.subsample > 0:
        # `load_master` autodetects parquet vs csv based on extension.
        df_raw = load_master(args.data)
        if args.subsample > 0:
            df_raw = df_raw.sort_values(["FL_DATE", "CRS_DEP_MIN"]).head(args.subsample).reset_index(drop=True)

    result = run_training(cfg, df_raw=df_raw)
    print(json.dumps(result.metrics, indent=2, default=str))
    print(f"\nArtifacts saved to: {result.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
