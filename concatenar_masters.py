"""Concatenate several master datasets into one, sorted by FL_DATE + CRS_DEP_MIN.

Usage:
    python3 concatenar_masters.py dataset_maestro_ATL_2022-2024_*.csv dataset_maestro_ATL_2025_*.csv
    python3 concatenar_masters.py --out dataset_maestro_ATL_2022-2025_BTS_IEM_ORIG_DEST.csv <files>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Concatenate OnTimeAI master CSVs.")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    inputs = sorted(args.inputs)
    print(f"Inputs ({len(inputs)}):")
    for pth in inputs:
        print(f"  {pth} ({pth.stat().st_size / 1e6:.1f} MB)")

    frames: list[pd.DataFrame] = []
    first_cols: list[str] | None = None
    for pth in inputs:
        df = pd.read_csv(pth, parse_dates=["FL_DATE"], low_memory=False)
        if first_cols is None:
            first_cols = list(df.columns)
        else:
            missing = set(first_cols) - set(df.columns)
            extra = set(df.columns) - set(first_cols)
            if missing or extra:
                print(f"  [warn] {pth.name} schema diff — missing={missing}, extra={extra}")
            df = df.reindex(columns=first_cols)
        print(f"  {pth.name}: {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", "CRS_DEP_TIME"],
        keep="first",
    )
    deduped = before - len(combined)
    if deduped:
        print(f"  deduped {deduped:,} duplicate rows")

    combined = combined.sort_values(["FL_DATE", "CRS_DEP_MIN"], kind="stable").reset_index(drop=True)

    if args.out is None:
        years = sorted({int(y) for y in combined["FL_DATE"].dt.year.dropna().unique()})
        label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
        args.out = Path(f"dataset_maestro_ATL_{label}_BTS_IEM_ORIG_DEST.csv")

    combined.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    print(f"  {len(combined):,} rows × {len(combined.columns)} cols")
    print(f"  date range: {combined['FL_DATE'].min().date()} ... {combined['FL_DATE'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
