"""CLI entry point: Optuna hyperparameter search for OnTimeAI LightGBM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH, TrainConfig
from ontimeai.tuning import save_best_params, tune_hyperparams


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning.")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--num-boost-round", type=int, default=800)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "best_params.json")
    p.add_argument("--no-balance", action="store_true")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        target="binary",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=50,
        balance_classes=not args.no_balance,
        random_state=args.seed,
    )

    df_raw = pd.read_csv(args.data, parse_dates=["FL_DATE"], low_memory=False)
    if args.subsample > 0:
        df_raw = df_raw.sort_values(["FL_DATE", "CRS_DEP_MIN"]).head(args.subsample).reset_index(drop=True)

    print(f"Starting Optuna tuning: {args.n_trials} trials, {len(df_raw):,} rows")
    best = tune_hyperparams(df_raw, cfg, n_trials=args.n_trials, sampler_seed=args.seed)
    print(f"\nBest val AUC: {best['best_value']:.4f}")
    print(f"Best params: {json.dumps(best['best_params'], indent=2)}")

    save_best_params(best, args.out)
    print(f"\nSaved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
