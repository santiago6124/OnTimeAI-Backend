#!/usr/bin/env python3
"""OnTimeAI — Database Pruning Script.

Deletes records older than a specified number of days to keep the SQLite database
size compact and avoid Out of Memory (OOM) errors in Cloud Run.

Usage:
    python3 scripts/prune_db.py --db live_data.db --days 30
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_VACUUM_MIN_DELETED = 50_000


def _vacuum_threshold(configured: int | None) -> int:
    value = (
        os.environ.get(
            "PRUNE_VACUUM_MIN_DELETED",
            str(DEFAULT_VACUUM_MIN_DELETED),
        )
        if configured is None
        else configured
    )
    threshold = int(value)
    if threshold < 0:
        raise ValueError("PRUNE_VACUUM_MIN_DELETED must be >= 0")
    return threshold


def _vacuum(con: sqlite3.Connection) -> None:
    con.execute("VACUUM;")


def prune_db(
    db_path: Path,
    days: int = 30,
    dry_run: bool = False,
    vacuum_min_deleted: int | None = None,
) -> int:
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return 1

    try:
        vacuum_threshold = _vacuum_threshold(vacuum_min_deleted)
    except (TypeError, ValueError) as exc:
        print(f"Error: invalid VACUUM threshold: {exc}")
        return 1

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff_date.isoformat()
    # Also support date-only string format (e.g. YYYY-MM-DD) for simpler matching if needed
    cutoff_date_only = cutoff_date.strftime("%Y-%m-%d")

    print(f"=== Database Pruning Status ({'DRY RUN' if dry_run else 'EXECUTE'}) ===")
    print(f"Database: {db_path} ({db_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Pruning records older than: {days} days ({cutoff_iso})")

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    
    try:
        # Define tables, criteria, and timezone-aware queries
        # 1. Predictions
        pred_cnt = con.execute("SELECT count(*) FROM predictions WHERE predicted_at_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"predictions: {pred_cnt:,} rows older than cutoff")

        # 2. Prediction SHAP
        shap_cnt = con.execute("SELECT count(*) FROM prediction_shap WHERE predicted_at_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"prediction_shap: {shap_cnt:,} rows older than cutoff")

        # 3. Flights (using scheduled_out_utc)
        flight_cnt = con.execute("SELECT count(*) FROM flights WHERE scheduled_out_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"flights: {flight_cnt:,} rows older than cutoff")

        # 4. Actuals (to be deleted if their flight is deleted)
        if dry_run:
            actual_cnt = con.execute("""
                SELECT count(*) FROM actuals 
                WHERE fa_flight_id NOT IN (
                    SELECT fa_flight_id FROM flights WHERE scheduled_out_utc >= ?
                )
            """, (cutoff_iso,)).fetchone()[0]
        else:
            actual_cnt = 0 # Calculated post-flight-delete

        # 5. Weather observations
        weather_cnt = con.execute("SELECT count(*) FROM weather_obs WHERE valid_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"weather_obs: {weather_cnt:,} rows older than cutoff")

        # 6. Runs
        run_cnt = con.execute("SELECT count(*) FROM runs WHERE started_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"runs: {run_cnt:,} rows older than cutoff")

        # 7. Harvester runs
        harvester_cnt = con.execute("SELECT count(*) FROM harvester_runs WHERE run_at_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"harvester_runs: {harvester_cnt:,} rows older than cutoff")

        # 8. NAS status
        nas_cnt = con.execute("SELECT count(*) FROM nas_status WHERE captured_at_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"nas_status: {nas_cnt:,} rows older than cutoff")

        # 9. Aircraft positions
        pos_cnt = con.execute("SELECT count(*) FROM aircraft_position WHERE captured_at_utc < ?", (cutoff_iso,)).fetchone()[0]
        print(f"aircraft_position: {pos_cnt:,} rows older than cutoff")

        if dry_run:
            print(f"actuals: {actual_cnt:,} rows would be orphaned and deleted")
            print("Dry run completed. No modifications made.")
            return 0

        # Execute Deletions
        print("\nPruning data...")
        
        # Disable journal/foreign key speed limits if applicable, wrap in transaction
        con.execute("BEGIN TRANSACTION;")
        changes_before_prune = con.total_changes
        
        con.execute("DELETE FROM prediction_shap WHERE predicted_at_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM predictions WHERE predicted_at_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM flights WHERE scheduled_out_utc < ?", (cutoff_iso,))
        
        # Delete orphaned actuals
        cur = con.execute("DELETE FROM actuals WHERE fa_flight_id NOT IN (SELECT fa_flight_id FROM flights)")
        actual_deleted = cur.rowcount
        print(f"Deleted actuals: {actual_deleted:,} orphaned rows")

        con.execute("DELETE FROM weather_obs WHERE valid_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM runs WHERE started_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM harvester_runs WHERE run_at_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM nas_status WHERE captured_at_utc < ?", (cutoff_iso,))
        con.execute("DELETE FROM aircraft_position WHERE captured_at_utc < ?", (cutoff_iso,))
        
        total_deleted = con.total_changes - changes_before_prune
        con.commit()
        print(f"Data pruned successfully ({total_deleted:,} rows deleted).")

        # VACUUM rewrites the complete SQLite file.  Running it every predictor
        # cycle makes the GCS CAS window unnecessarily long, so only reclaim
        # pages after a material pruning pass.
        if total_deleted > vacuum_threshold:
            print(
                "Reclaiming SQLite disk space (VACUUM): "
                f"{total_deleted:,} deleted rows exceed the "
                f"{vacuum_threshold:,} threshold..."
            )
            _vacuum(con)
            print("VACUUM completed.")
        else:
            print(
                "VACUUM skipped: "
                f"{total_deleted:,} deleted rows do not exceed the "
                f"{vacuum_threshold:,} threshold."
            )

    except Exception as e:
        if con.in_transaction:
            con.rollback()
        print(f"Error during pruning: {e}")
        return 1
    finally:
        con.close()

    new_size = db_path.stat().st_size / 1e6
    print(f"Pruned Database size: {new_size:.2f} MB")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=os.environ.get("DB_PATH", "live_data.db"),
                    help="Path to the SQLite database file.")
    ap.add_argument("--days", type=int, default=30,
                    help="Number of days of history to retain (default: 30).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Check row counts to be pruned without deleting them.")
    ap.add_argument(
        "--vacuum-min-deleted",
        type=int,
        default=None,
        help=(
            "Run VACUUM only when deleted rows exceed this value "
            f"(default: PRUNE_VACUUM_MIN_DELETED or {DEFAULT_VACUUM_MIN_DELETED})."
        ),
    )
    args = ap.parse_args()

    if args.vacuum_min_deleted is not None and args.vacuum_min_deleted < 0:
        ap.error("--vacuum-min-deleted must be >= 0")

    db_path = Path(args.db)
    return prune_db(
        db_path,
        args.days,
        args.dry_run,
        vacuum_min_deleted=args.vacuum_min_deleted,
    )

if __name__ == "__main__":
    raise SystemExit(main())
