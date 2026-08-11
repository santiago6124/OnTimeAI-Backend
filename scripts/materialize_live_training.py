"""Materialize immutable live-training events into a leakage-safe Parquet set.

Examples:

    # From the production ledger in GCS
    python scripts/materialize_live_training.py \
      --bucket ontimeai-150917658060-training-us-central1 \
      --prefix live-training \
      --start-date 2026-08-01 --end-date 2026-10-31 \
      --horizons-hours 6 2 \
      --output /tmp/ontimeai-live-silver.parquet

    # From Parquet shards downloaded locally
    python scripts/materialize_live_training.py \
      --input-dir /tmp/ontimeai-training-ledger \
      --horizons-hours 2 \
      --output /tmp/ontimeai-live-silver.parquet
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.training_store import (  # noqa: E402
    MATERIALIZER_VERSION,
    OUTCOME_EVENT_TYPE,
    SNAPSHOT_EVENT_TYPE,
    materialize_fixed_horizons,
)


FeatureContractKey = tuple[str, str]
FeatureContracts = dict[FeatureContractKey, dict[str, Any]]
EventLoadResult = tuple[
    pd.DataFrame,
    pd.DataFrame,
    FeatureContracts,
    list[dict[str, Any]],
]


def _date_from_object_name(name: str, *, marker: str = "event_date=") -> date | None:
    if marker not in name:
        return None
    value = name.split(marker, 1)[1].split("/", 1)[0]
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _in_date_range(
    name: str,
    start: date | None,
    end: date | None,
    *,
    kind: str,
    outcome_lag_days: int,
) -> bool:
    if kind == OUTCOME_EVENT_TYPE and "flight_date=" in name:
        flight_partition = name.split("flight_date=", 1)[1].split("/", 1)[0]
        # Outcomes without a trustworthy physical flight date are rare but
        # cannot be excluded causally from an arbitrary requested window.
        if flight_partition == "unknown":
            return True
        partition_date = _date_from_object_name(name, marker="flight_date=")
        if partition_date is None:
            return False
        return (
            (start is None or partition_date >= start)
            and (end is None or partition_date <= end)
        )

    event_date = _date_from_object_name(name)
    if event_date is None:
        return False
    effective_end = end
    if kind == OUTCOME_EVENT_TYPE and end is not None:
        # Compatibility with pre-partition ledgers.  New outcome shards are
        # keyed by flight_date above, so arbitrarily late revisions are found
        # without expanding this observation-date window.
        effective_end = end + timedelta(days=outcome_lag_days)
    return (
        (start is None or event_date >= start)
        and (effective_end is None or event_date <= effective_end)
    )


def _frame_type(name: str) -> str | None:
    if f"/{SNAPSHOT_EVENT_TYPE}/" in name:
        return SNAPSHOT_EVENT_TYPE
    if f"/{OUTCOME_EVENT_TYPE}/" in name:
        return OUTCOME_EVENT_TYPE
    return None


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _feature_contract_from_bytes(
    body: bytes,
    *,
    source_name: str,
) -> tuple[FeatureContractKey, dict[str, Any]]:
    """Parse and verify one immutable feature-contract sidecar."""
    try:
        contract = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid feature contract JSON: {source_name}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"feature contract must be a JSON object: {source_name}")

    required = {
        "feature_schema_hash",
        "category_mapping_sha256",
        "feature_order_json",
        "categorical_features_json",
        "category_mapping_json",
    }
    missing = required.difference(contract)
    if missing:
        raise ValueError(
            f"feature contract {source_name} missing fields: {sorted(missing)!r}"
        )

    feature_schema_hash = str(contract["feature_schema_hash"])
    category_mapping_sha256 = str(contract["category_mapping_sha256"])
    feature_order_json = str(contract["feature_order_json"])
    categorical_features_json = str(contract["categorical_features_json"])
    category_mapping_json = str(contract["category_mapping_json"])
    try:
        feature_order = json.loads(feature_order_json)
        categorical_features = json.loads(categorical_features_json)
        category_mapping = json.loads(category_mapping_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid nested feature contract JSON: {source_name}") from exc
    if (
        not isinstance(feature_order, list)
        or not feature_order
        or not all(isinstance(value, str) for value in feature_order)
        or len(feature_order) != len(set(feature_order))
    ):
        raise ValueError(f"invalid feature order in contract: {source_name}")
    if (
        not isinstance(categorical_features, list)
        or not all(isinstance(value, str) for value in categorical_features)
        or not set(categorical_features).issubset(feature_order)
    ):
        raise ValueError(f"invalid categorical feature list in contract: {source_name}")
    if not isinstance(category_mapping, dict):
        raise ValueError(f"category mapping must be an object: {source_name}")
    missing_mappings = set(categorical_features).difference(category_mapping)
    if missing_mappings:
        raise ValueError(
            f"feature contract {source_name} lacks category mappings: "
            f"{sorted(missing_mappings)!r}"
        )

    actual_mapping_hash = sha256(category_mapping_json.encode("utf-8")).hexdigest()
    if actual_mapping_hash != category_mapping_sha256:
        raise ValueError(f"category mapping SHA-256 mismatch: {source_name}")
    categorical_set = set(categorical_features)
    schema_contract = [
        {
            "name": column,
            "type": "string" if column in categorical_set else "float64",
        }
        for column in feature_order
    ]
    schema_json = json.dumps(schema_contract, separators=(",", ":"))
    actual_schema_hash = sha256(schema_json.encode("utf-8")).hexdigest()
    if actual_schema_hash != feature_schema_hash:
        raise ValueError(f"feature schema SHA-256 mismatch: {source_name}")

    normalized = dict(contract)
    normalized.update(
        {
            "feature_schema_hash": feature_schema_hash,
            "category_mapping_sha256": category_mapping_sha256,
            "feature_order_json": feature_order_json,
            "categorical_features_json": categorical_features_json,
            "category_mapping_json": category_mapping_json,
        }
    )
    return (feature_schema_hash, category_mapping_sha256), normalized


def _register_feature_contract(
    contracts: FeatureContracts,
    contract_sources: dict[FeatureContractKey, dict[str, Any]],
    *,
    body: bytes,
    source: dict[str, Any],
) -> None:
    key, contract = _feature_contract_from_bytes(body, source_name=str(source["name"]))
    existing = contracts.get(key)
    if existing is not None and existing != contract:
        raise ValueError(f"conflicting feature-contract sidecars for {key!r}")
    contracts[key] = contract
    contract_sources.setdefault(key, source)


def _select_referenced_contracts(
    snapshots: pd.DataFrame,
    available: FeatureContracts,
    contract_sources: dict[FeatureContractKey, dict[str, Any]],
) -> tuple[FeatureContracts, list[dict[str, Any]]]:
    if snapshots.empty:
        return {}, []
    required = {"feature_schema_hash", "category_mapping_sha256"}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(
            f"snapshot shards missing contract references: {sorted(missing)!r}"
        )
    references = snapshots[
        ["feature_schema_hash", "category_mapping_sha256"]
    ].drop_duplicates()
    keys: list[FeatureContractKey] = []
    for row in references.itertuples(index=False, name=None):
        if any(pd.isna(value) for value in row):
            raise ValueError("snapshot shard contains a null feature-contract reference")
        keys.append((str(row[0]), str(row[1])))
    missing_keys = [key for key in keys if key not in available]
    if missing_keys:
        raise ValueError(f"missing feature-contract sidecars: {missing_keys!r}")
    selected = {key: available[key] for key in sorted(keys)}
    sources = [contract_sources[key] for key in sorted(keys)]
    return selected, sources


def _sorted_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sources,
        key=lambda source: (
            str(source.get("kind", "")),
            str(source.get("name", "")),
            int(source.get("generation", 0)),
        ),
    )


def load_local_events(
    root: Path,
    *,
    start: date | None,
    end: date | None,
    outcome_lag_days: int = 7,
) -> EventLoadResult:
    snapshots: list[pd.DataFrame] = []
    outcomes: list[pd.DataFrame] = []
    event_sources: list[dict[str, Any]] = []
    contracts: FeatureContracts = {}
    contract_sources: dict[FeatureContractKey, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if "/contracts/" not in "/" + relative:
            continue
        body = path.read_bytes()
        _register_feature_contract(
            contracts,
            contract_sources,
            body=body,
            source={
                "kind": "feature_contract",
                "name": relative,
                "size": len(body),
                "sha256": sha256(body).hexdigest(),
            },
        )
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root).as_posix()
        kind = _frame_type("/" + relative)
        if kind is None or not _in_date_range(
            relative, start, end, kind=kind, outcome_lag_days=outcome_lag_days,
        ):
            continue
        body = path.read_bytes()
        frame = pd.read_parquet(BytesIO(body))
        (snapshots if kind == SNAPSHOT_EVENT_TYPE else outcomes).append(frame)
        event_sources.append(
            {
                "kind": "event_shard",
                "event_type": kind,
                "name": relative,
                "size": len(body),
                "sha256": sha256(body).hexdigest(),
            }
        )
    snapshot_frame = _concat(snapshots)
    outcome_frame = _concat(outcomes)
    selected_contracts, selected_contract_sources = _select_referenced_contracts(
        snapshot_frame, contracts, contract_sources,
    )
    sources = _sorted_sources(event_sources + selected_contract_sources)
    return snapshot_frame, outcome_frame, selected_contracts, sources


def load_gcs_events(
    bucket_name: str,
    prefix: str,
    *,
    start: date | None,
    end: date | None,
    outcome_lag_days: int = 7,
) -> EventLoadResult:
    from google.cloud import storage as gcs

    client = gcs.Client()
    snapshots: list[pd.DataFrame] = []
    outcomes: list[pd.DataFrame] = []
    event_sources: list[dict[str, Any]] = []
    contracts: FeatureContracts = {}
    contract_sources: dict[FeatureContractKey, dict[str, Any]] = {}
    ledger_prefix = prefix.strip("/")
    for blob in client.list_blobs(
        bucket_name, prefix=ledger_prefix + "/contracts/",
    ):
        if not blob.name.endswith(".json"):
            continue
        generation = int(blob.generation)
        body = blob.download_as_bytes(if_generation_match=generation)
        _register_feature_contract(
            contracts,
            contract_sources,
            body=body,
            source={
                "kind": "feature_contract",
                "name": blob.name,
                "generation": generation,
                "size": len(body),
                "sha256": sha256(body).hexdigest(),
            },
        )
    for blob in client.list_blobs(bucket_name, prefix=ledger_prefix + "/raw/"):
        if not blob.name.endswith(".parquet"):
            continue
        kind = _frame_type("/" + blob.name)
        if kind is None or not _in_date_range(
            blob.name, start, end, kind=kind, outcome_lag_days=outcome_lag_days,
        ):
            continue
        generation = int(blob.generation)
        body = blob.download_as_bytes(if_generation_match=generation)
        frame = pd.read_parquet(BytesIO(body))
        (snapshots if kind == SNAPSHOT_EVENT_TYPE else outcomes).append(frame)
        event_sources.append(
            {
                "kind": "event_shard",
                "event_type": kind,
                "name": blob.name,
                "generation": generation,
                "size": len(body),
                "sha256": sha256(body).hexdigest(),
            }
        )
    snapshot_frame = _concat(snapshots)
    outcome_frame = _concat(outcomes)
    selected_contracts, selected_contract_sources = _select_referenced_contracts(
        snapshot_frame, contracts, contract_sources,
    )
    sources = _sorted_sources(event_sources + selected_contract_sources)
    return snapshot_frame, outcome_frame, selected_contracts, sources


def _contracts_for_manifest(contracts: FeatureContracts) -> list[dict[str, Any]]:
    return [dict(contracts[key]) for key in sorted(contracts)]


def _materializer_code_sha256() -> str:
    """Hash both source files that define materialization semantics."""
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "ontimeai" / "training_store.py",
    )
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        if not path.is_file():
            raise RuntimeError(f"materializer source file is missing: {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
    }


def _build_dataset_identity(
    *,
    sources: list[dict[str, Any]],
    contracts: FeatureContracts,
    start: date | None,
    end: date | None,
    horizons_hours: list[float],
    max_snapshot_age_minutes: float,
    outcome_lag_days: int,
    output_sha256: str,
) -> dict[str, Any]:
    contract_identities = [
        {
            "feature_schema_hash": key[0],
            "category_mapping_sha256": key[1],
            "contract_sha256": sha256(
                json.dumps(
                    contracts[key], sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        for key in sorted(contracts)
    ]
    materializer_code_sha256 = _materializer_code_sha256()
    runtime_versions = _runtime_versions()
    return {
        "materializer_version": MATERIALIZER_VERSION,
        "materializer_code_sha256": materializer_code_sha256,
        "runtime_versions": runtime_versions,
        "source_objects": _sorted_sources(sources),
        "feature_contracts": contract_identities,
        "selection": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "horizons_hours": sorted(float(value) for value in horizons_hours),
            "max_snapshot_age_minutes": float(max_snapshot_age_minutes),
            "outcome_lag_days": int(outcome_lag_days),
        },
        "output_sha256": output_sha256,
    }


def _dataset_id(identity: dict[str, Any]) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    return f"{MATERIALIZER_VERSION}-{digest[:20]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize causal live prediction snapshots with mature outcomes.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bucket", help="GCS bucket containing the append-only ledger")
    source.add_argument("--input-dir", type=Path, help="Locally downloaded ledger root")
    parser.add_argument("--prefix", default="live-training", help="Ledger prefix inside GCS")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--horizons-hours", nargs="+", type=float, default=[2.0])
    parser.add_argument("--max-snapshot-age-minutes", type=float, default=60.0)
    parser.add_argument(
        "--outcome-lag-days", type=int, default=7,
        help="Also load outcome revisions this many days after --end-date",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise SystemExit("--start-date must be <= --end-date")
    if args.outcome_lag_days < 0:
        raise SystemExit("--outcome-lag-days must be >= 0")
    if (
        not args.horizons_hours
        or any(not math.isfinite(value) or value < 0 for value in args.horizons_hours)
        or len(set(args.horizons_hours)) != len(args.horizons_hours)
    ):
        raise SystemExit("--horizons-hours must be unique, finite and >= 0")
    if (
        not math.isfinite(args.max_snapshot_age_minutes)
        or args.max_snapshot_age_minutes < 0
    ):
        raise SystemExit("--max-snapshot-age-minutes must be finite and >= 0")

    # A flight on --start-date can have its causal T-6/T-12 snapshot in an
    # earlier UTC partition.  Likewise, a late local departure can fall in the
    # next UTC partition.  Load boundary shards and trim the materialized rows
    # by flight date below; otherwise valid boundary examples disappear.
    causal_lookback_days = max(
        1,
        math.ceil(
            (
                max(args.horizons_hours)
                + args.max_snapshot_age_minutes / 60.0
            )
            / 24.0
        ),
    )
    load_start = (
        args.start_date - timedelta(days=causal_lookback_days)
        if args.start_date
        else None
    )
    load_end = args.end_date + timedelta(days=1) if args.end_date else None

    if args.bucket:
        snapshots, outcomes, contracts, sources = load_gcs_events(
            args.bucket,
            args.prefix,
            start=load_start,
            end=load_end,
            outcome_lag_days=args.outcome_lag_days,
        )
        source_description = f"gs://{args.bucket}/{args.prefix.strip('/')}"
    else:
        snapshots, outcomes, contracts, sources = load_local_events(
            args.input_dir,
            start=load_start,
            end=load_end,
            outcome_lag_days=args.outcome_lag_days,
        )
        source_description = str(args.input_dir.resolve())

    print(
        f"Loaded {len(snapshots):,} snapshot events and {len(outcomes):,} "
        f"outcome revisions from {len(sources):,} immutable shard(s)."
    )
    result, report = materialize_fixed_horizons(
        snapshots,
        outcomes,
        horizons_hours=args.horizons_hours,
        max_snapshot_age_minutes=args.max_snapshot_age_minutes,
        feature_contracts=contracts,
    )
    if not result.empty and (args.start_date or args.end_date):
        scheduled_dates = pd.to_datetime(
            result["scheduled_out_utc"], errors="coerce", utc=True,
        ).dt.date
        if "flight_date_local" in result.columns:
            local_dates = pd.to_datetime(
                result["flight_date_local"], errors="coerce",
            ).dt.date
            flight_dates = local_dates.where(local_dates.notna(), scheduled_dates)
        else:
            flight_dates = scheduled_dates
        date_mask = pd.Series(True, index=result.index)
        if args.start_date:
            date_mask &= flight_dates.ge(args.start_date)
        if args.end_date:
            date_mask &= flight_dates.le(args.end_date)
        result = result[date_mask].reset_index(drop=True)
        report["rows_before_flight_date_filter"] = int(report.get("rows", 0))
        report["rows"] = len(result)
        report["positive_rate"] = (
            float(result["TARGET"].mean()) if not result.empty else None
        )
        report["horizon_counts"] = {
            str(float(horizon)): int(
                pd.to_numeric(result["_horizon_hours"], errors="coerce")
                .eq(float(horizon))
                .sum()
            )
            for horizon in args.horizons_hours
        }
        report["flight_date_filter"] = {
            "start_date": args.start_date.isoformat() if args.start_date else None,
            "end_date": args.end_date.isoformat() if args.end_date else None,
            "boundary_lookback_days": causal_lookback_days,
        }
    if result.empty:
        print("No eligible rows for the requested horizons/date range.")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    output_sha = sha256(args.output.read_bytes()).hexdigest()
    dataset_identity = _build_dataset_identity(
        sources=sources,
        contracts=contracts,
        start=args.start_date,
        end=args.end_date,
        horizons_hours=args.horizons_hours,
        max_snapshot_age_minutes=args.max_snapshot_age_minutes,
        outcome_lag_days=args.outcome_lag_days,
        output_sha256=output_sha,
    )
    manifest = {
        "dataset_id": _dataset_id(dataset_identity),
        "materializer_version": MATERIALIZER_VERSION,
        "materializer_code_sha256": dataset_identity["materializer_code_sha256"],
        "runtime_versions": dataset_identity["runtime_versions"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_description,
        "start_date": args.start_date.isoformat() if args.start_date else None,
        "end_date": args.end_date.isoformat() if args.end_date else None,
        "horizons_hours": args.horizons_hours,
        "max_snapshot_age_minutes": args.max_snapshot_age_minutes,
        "outcome_lag_days": args.outcome_lag_days,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "dataset_identity": dataset_identity,
        "feature_contracts": _contracts_for_manifest(contracts),
        "report": report,
        "source_objects": _sorted_sources(sources),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(result):,} rows to {args.output}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
