"""Export the current live_data.db state to CSV snapshots in artifacts/.

Useful for archiving session results in git (the SQLite DB itself is
gitignored as mutable runtime state). Run this at the end of a session,
or whenever you want a frozen snapshot for the thesis defense.

Usage:
    python scripts/export_session_snapshot.py                     # default name "session"
    python scripts/export_session_snapshot.py --label demo_2026-05-04
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "live_data.db"
ARTIFACTS = REPO / "artifacts"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                   help="Suffix for the CSV files (default: today UTC)")
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS / "live_snapshots",
                   help="Output directory")
    args = p.parse_args()

    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))

    # 1. predictions joined with their flight metadata
    preds = pd.read_sql_query(
        """SELECT
              p.fa_flight_id, p.predicted_at_utc, p.proba_delay,
              p.predicted_delay, p.threshold_used, p.threshold_strategy,
              f.ident_iata, f.tail_num, f.op_carrier, f.origin, f.dest,
              f.scheduled_off_utc, f.scheduled_on_utc,
              f.inbound_fa_flight_id, f.aircraft_type
           FROM predictions p
           LEFT JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           ORDER BY p.predicted_at_utc, p.proba_delay DESC""",
        conn,
    )
    preds_path = args.out_dir / f"predictions_{args.label}.csv"
    preds.to_csv(preds_path, index=False)
    print(f"Wrote {preds_path}  ({len(preds)} rows)")

    # 2. actuals joined with flight metadata (the inbounds chain-walk hydrated).
    # `scheduled_in_utc` lives in flights, not actuals.
    acts = pd.read_sql_query(
        """SELECT
              a.fa_flight_id, f.scheduled_in_utc, a.actual_in_utc,
              a.arr_delay_min, a.departure_delay_min, a.cancelled, a.diverted,
              a.settled_at_utc,
              f.ident_iata, f.tail_num, f.op_carrier, f.origin, f.dest
           FROM actuals a
           LEFT JOIN flights f ON f.fa_flight_id = a.fa_flight_id
           ORDER BY a.actual_in_utc DESC""",
        conn,
    )
    acts_path = args.out_dir / f"actuals_{args.label}.csv"
    acts.to_csv(acts_path, index=False)
    print(f"Wrote {acts_path}  ({len(acts)} rows)")

    # 3. runs ledger
    runs = pd.read_sql_query("SELECT * FROM runs ORDER BY run_id", conn)
    runs_path = args.out_dir / f"runs_{args.label}.csv"
    runs.to_csv(runs_path, index=False)
    print(f"Wrote {runs_path}  ({len(runs)} rows)")

    # 4. summary JSON for quick stats
    summary = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "counts": {
            "flights": int(pd.read_sql_query("SELECT COUNT(*) c FROM flights", conn).iloc[0, 0]),
            "predictions": len(preds),
            "actuals": len(acts),
            "weather_obs": int(pd.read_sql_query("SELECT COUNT(*) c FROM weather_obs", conn).iloc[0, 0]),
            "runs": len(runs),
        },
        "predictions_by_strategy": (
            preds.groupby("threshold_strategy", dropna=False)
            .agg(n=("proba_delay", "size"),
                 mean_proba=("proba_delay", "mean"),
                 std_proba=("proba_delay", "std"),
                 pos_pred_rate=("predicted_delay", "mean"))
            .round(4)
            .to_dict(orient="index")
            if not preds.empty else {}
        ),
        "predictions_proba_quantiles": (
            preds["proba_delay"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_dict()
            if not preds.empty else {}
        ),
        "actuals_arr_delay_quantiles": (
            acts["arr_delay_min"].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict()
            if not acts.empty else {}
        ),
    }
    # chain-walk efficiency: of predictions whose flight had an inbound_fa_flight_id,
    # how many of those inbounds were hydrated in the actuals table?
    if not preds.empty:
        with_inbound = preds.dropna(subset=["inbound_fa_flight_id"])
        n_with_inbound = len(with_inbound)
        if n_with_inbound > 0:
            hydrated = with_inbound["inbound_fa_flight_id"].isin(acts["fa_flight_id"]).sum()
            summary["chain_walk_efficiency"] = {
                "predictions_with_inbound": int(n_with_inbound),
                "inbounds_hydrated": int(hydrated),
                "hit_rate": round(float(hydrated) / n_with_inbound, 4),
            }
        else:
            summary["chain_walk_efficiency"] = {"predictions_with_inbound": 0}
    summary_path = args.out_dir / f"summary_{args.label}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
