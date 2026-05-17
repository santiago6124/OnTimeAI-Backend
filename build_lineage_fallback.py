"""Precompute the cold-deck lookup table from the BTS master.

Run once after every refresh of the master CSV:

    python3 build_lineage_fallback.py
    python3 build_lineage_fallback.py --master path/to/other_master.csv --out artifacts/fb_v2.joblib

The output (~few MB) is loaded by predict.py / live_pull.py at inference time
to fill lineage / rolling features that came back NaN from the live history
buffer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH
from ontimeai.data import load_master
from ontimeai.lineage_fallback import build_lookups, save_lookups


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build lineage cold-deck lookup table.")
    p.add_argument("--master", type=Path, default=DATA_PATH,
                   help=f"Master CSV path (default: {DATA_PATH.name})")
    p.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "lineage_fallback.joblib",
                   help="Output joblib path")
    args = p.parse_args(argv)

    print(f"Loading master from {args.master} ...")
    master = load_master(args.master)
    print(f"  {len(master):,} rows × {len(master.columns)} columns")

    print("Computing lookups ...")
    lookups = build_lookups(master)

    for name, val in lookups.items():
        if name == "_meta":
            print(f"  meta: {val}")
        elif hasattr(val, "__len__"):
            print(f"  {name}: {len(val):,} buckets")
        else:
            print(f"  {name}: {val:.4f}")

    out = save_lookups(lookups, args.out)
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
