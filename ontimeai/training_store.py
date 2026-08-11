"""Append-only live training-data ledger.

The serving SQLite database remains the operational source of truth for the API,
but it is intentionally pruned and mutable.  This module captures the exact
feature vectors seen by the model in a small transactional outbox and exports
them as immutable Parquet shards in a separate GCS bucket.

Delivery is deliberately two phase:

1. ``live_pull`` writes predictions and outbox events in the same SQLite
   transaction.
2. ``live_job`` uploads the winning SQLite generation, publishes its outbox with
   GCS ``if_generation_match=0``, and acknowledges it on the next winning cycle.

If a Cloud Run attempt loses the SQLite generation race, its new events are
never published.  If publication succeeds but acknowledgement is lost, the next
attempt resolves the same deterministic GCS object and safely acknowledges it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TRAINING_SCHEMA_VERSION = "live-training-v1"
IDENTITY_VERSION = "flight-key-v1"
HORIZON_DEFINITION_VERSION = "scheduled-out-v1"
SNAPSHOT_EVENT_TYPE = "prediction_snapshots"
OUTCOME_EVENT_TYPE = "flight_outcomes"
TARGET_DEFINITION = "arr_delay_min > 15"
MATERIALIZER_VERSION = "live-silver-v2"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRANSFORMER_CODE_ROOTS = (
    Path("ontimeai"),
    Path("feature_engineering_v7"),
)
_TRANSFORMER_CODE_FILES = (
    Path("live_pull.py"),
    Path("predict.py"),
)

_OUTBOX_DDL = (
    """CREATE TABLE IF NOT EXISTS training_export_outbox (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    event_date TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_training_outbox_type_date
    ON training_export_outbox(event_type, event_date)""",
    """CREATE TABLE IF NOT EXISTS training_export_delivery (
    event_id TEXT PRIMARY KEY,
    delivered_at_utc TEXT NOT NULL,
    object_name TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    payload_sha256 TEXT
    )""",
    """CREATE INDEX IF NOT EXISTS idx_training_delivery_time
    ON training_export_delivery(delivered_at_utc)""",
    """CREATE TABLE IF NOT EXISTS training_outcome_state (
    source_record_id TEXT PRIMARY KEY,
    source_fa_flight_id TEXT,
    state_sha256 TEXT NOT NULL,
    outcome_revision INTEGER NOT NULL,
    last_observed_at_utc TEXT NOT NULL,
    partition_flight_date TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS training_feature_contracts (
    feature_schema_hash TEXT NOT NULL,
    category_mapping_sha256 TEXT NOT NULL,
    contract_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (feature_schema_hash, category_mapping_sha256)
    )""",
    """CREATE INDEX IF NOT EXISTS idx_actuals_settled_at
    ON actuals(settled_at_utc)""",
)


@dataclass(frozen=True)
class PublishSummary:
    events: int
    objects: int
    object_names: tuple[str, ...]


def ensure_training_store_schema(conn: sqlite3.Connection) -> None:
    """Create the bounded transactional outbox without an implicit commit.

    ``sqlite3.Connection.executescript`` commits a pending transaction before
    running its script.  Executing each DDL statement separately is essential:
    the serving prediction and its training snapshot must remain in the same
    rollback boundary.
    """
    for statement in _OUTBOX_DDL:
        conn.execute(statement)
    outcome_state_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(training_outcome_state)").fetchall()
    }
    if "source_fa_flight_id" not in outcome_state_columns:
        conn.execute(
            "ALTER TABLE training_outcome_state "
            "ADD COLUMN source_fa_flight_id TEXT"
        )
    if "partition_flight_date" not in outcome_state_columns:
        conn.execute(
            "ALTER TABLE training_outcome_state "
            "ADD COLUMN partition_flight_date TEXT"
        )
    _backfill_outcome_partition_dates(conn)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_outcome_state_source_fa "
        "ON training_outcome_state(source_fa_flight_id)"
    )
    delivery_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(training_export_delivery)").fetchall()
    }
    if "payload_sha256" not in delivery_columns:
        conn.execute(
            "ALTER TABLE training_export_delivery ADD COLUMN payload_sha256 TEXT"
        )


def _backfill_outcome_partition_dates(conn: sqlite3.Connection) -> None:
    """Anchor pre-migration outcome states before a schedule can be revised."""
    missing_states = conn.execute(
        """SELECT source_record_id, source_fa_flight_id
           FROM training_outcome_state
           WHERE partition_flight_date IS NULL
              OR trim(partition_flight_date)=''"""
    ).fetchall()
    if not missing_states:
        return

    pending_anchors: dict[str, str] = {}
    for (payload_json,) in conn.execute(
        """SELECT payload_json
           FROM training_export_outbox
           WHERE event_type=?
           ORDER BY datetime(created_at_utc), created_at_utc, event_id""",
        (OUTCOME_EVENT_TYPE,),
    ).fetchall():
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            continue
        source_record_id = str(payload.get("source_record_id") or "").strip()
        if not source_record_id or source_record_id in pending_anchors:
            continue
        raw_anchor = (
            payload.get("partition_flight_date_local")
            or payload.get("flight_date_local")
        )
        normalized_anchor = _flight_date_partition(raw_anchor)
        if (
            normalized_anchor != "unknown"
            or str(raw_anchor or "").strip().lower() == "unknown"
        ):
            pending_anchors[source_record_id] = normalized_anchor

    for source_record_id, source_fa_flight_id in missing_states:
        anchor = pending_anchors.get(str(source_record_id))
        if anchor is None and source_fa_flight_id:
            flight_row = conn.execute(
                "SELECT fl_date FROM flights WHERE fa_flight_id=? LIMIT 1",
                (source_fa_flight_id,),
            ).fetchone()
            if flight_row is not None:
                anchor = _flight_date_partition(flight_row[0])
        conn.execute(
            "UPDATE training_outcome_state SET partition_flight_date=? "
            "WHERE source_record_id=?",
            (anchor or "unknown", source_record_id),
        )


def training_store_enabled() -> bool:
    explicit = read_bool_env("TRAINING_STORE_ENABLED", default=None)
    required = training_store_required()
    if required:
        if explicit is False:
            raise ValueError(
                "TRAINING_STORE_REQUIRED=true conflicts with "
                "TRAINING_STORE_ENABLED=false"
            )
    bucket_configured = bool(os.getenv("TRAINING_DATA_BUCKET", "").strip())
    enabled = required or explicit is True or (explicit is None and bucket_configured)
    if enabled and not bucket_configured:
        raise ValueError("training capture requires TRAINING_DATA_BUCKET")
    return enabled


def training_store_required() -> bool:
    return bool(read_bool_env("TRAINING_STORE_REQUIRED", default=False))


def read_bool_env(name: str, *, default: bool | None) -> bool | None:
    """Read a boolean environment value without silently accepting typos."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no or on/off; got {raw!r}"
    )


def _positive_int_env(name: str, *, default: int) -> int:
    """Read a strictly positive operational batch limit."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def _hash_text(*parts: Any) -> str:
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return sha256(joined.encode("utf-8")).hexdigest()


def _atomic_training_batch(function):
    """Rollback a whole enqueue batch while preserving the caller transaction."""
    savepoint = "batch_" + function.__name__.removeprefix("enqueue_")

    @wraps(function)
    def wrapped(conn: sqlite3.Connection, *args: Any, **kwargs: Any):
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            result = function(conn, *args, **kwargs)
        except BaseException:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        return result

    return wrapped


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def transformer_code_sha256() -> str:
    """Hash the exact source bundle that produces captured feature vectors.

    Cloud Run execution IDs identify invocations, not code.  Hashing the files
    copied into the predictor image gives every snapshot a deterministic
    transformer identity even when deployment metadata is absent.
    """
    paths = [
        _PROJECT_ROOT / relative
        for relative in _TRANSFORMER_CODE_FILES
    ]
    for relative_root in _TRANSFORMER_CODE_ROOTS:
        root = _PROJECT_ROOT / relative_root
        paths.extend(sorted(root.rglob("*.py")))
    paths = sorted(set(paths), key=lambda path: path.relative_to(_PROJECT_ROOT).as_posix())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        relative_missing = [
            path.relative_to(_PROJECT_ROOT).as_posix() for path in missing
        ]
        raise RuntimeError(
            f"transformer code bundle is incomplete: {relative_missing!r}"
        )
    if not paths:
        raise RuntimeError("transformer code bundle is empty")

    digest = sha256()
    for path in paths:
        relative = path.relative_to(_PROJECT_ROOT).as_posix().encode("utf-8")
        body = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _normal_text(value: Any) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip().upper()
    return text or None


def _normal_flight_number(value: Any) -> str | None:
    text = _normal_text(value)
    if text is None:
        return None
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def _normal_service_date(value: Any) -> str | None:
    scalar = _json_scalar(value)
    if scalar is None:
        return None
    try:
        parsed = pd.Timestamp(scalar)
    except (TypeError, ValueError):
        return _normal_text(scalar)
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _flight_date_partition(value: Any) -> str:
    scalar = _json_scalar(value)
    if scalar is None:
        return "unknown"
    try:
        parsed = pd.Timestamp(scalar)
    except (TypeError, ValueError):
        return "unknown"
    if pd.isna(parsed):
        return "unknown"
    return parsed.date().isoformat()


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if _json_scalar(value) is not None:
            return value
    return None


def _utc_iso(value: Any, *, minute: bool = False) -> str | None:
    if value is None or value is pd.NA or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if minute:
        ts = ts.floor("min")
    return ts.isoformat()


def _minutes_between(later: Any, earlier: Any) -> float | None:
    later_iso = _utc_iso(later)
    earlier_iso = _utc_iso(earlier)
    if later_iso is None or earlier_iso is None:
        return None
    return float((pd.Timestamp(later_iso) - pd.Timestamp(earlier_iso)).total_seconds() / 60.0)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return _utc_iso(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _categorical_value(value: Any) -> str | None:
    scalar = _json_scalar(value)
    return None if scalar is None else str(scalar)


def _numeric_feature_value(value: Any) -> float | None:
    scalar = _json_scalar(value)
    if scalar is None:
        return None
    try:
        number = float(scalar)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_flight_keys(row: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return a versioned canonical service key and an optional rotation key.

    The canonical key is source-independent when schedule identity is complete.
    The tail/scheduled-minute rotation key is also retained because it is useful
    for reconciling FR24 and AeroAPI aliases, but is never the only stored ID.
    """
    carrier = _normal_text(_first_present(row, "op_carrier", "OP_CARRIER"))
    number = _normal_flight_number(
        _first_present(row, "flight_number", "OP_CARRIER_FL_NUM"),
    )
    origin = _normal_text(_first_present(row, "origin", "ORIGIN"))
    dest = _normal_text(_first_present(row, "dest", "DEST"))
    fl_date = _normal_service_date(_first_present(row, "fl_date", "FL_DATE"))
    scheduled_out = _utc_iso(
        _first_present(row, "scheduled_out_utc", "EVENT_ORIGIN_UTC"), minute=True,
    )
    scheduled_rotation = _utc_iso(
        _first_present(
            row, "scheduled_off_utc", "scheduled_out_utc", "EVENT_ORIGIN_UTC",
        ),
        minute=True,
    )
    tail = _normal_text(_first_present(row, "tail_num", "TAIL_NUM"))

    service_parts = (carrier, number, origin, dest, fl_date, scheduled_out)
    if all(service_parts):
        canonical = f"{IDENTITY_VERSION}:{_hash_text(*service_parts)}"
    elif tail and scheduled_rotation:
        canonical = (
            f"{IDENTITY_VERSION}:rotation:"
            f"{_hash_text(tail, scheduled_rotation)}"
        )
    else:
        source_id = row.get("fa_flight_id") or row.get("source_flight_id")
        stable_id = row.get("stable_id")
        canonical = f"{IDENTITY_VERSION}:source:{_hash_text(stable_id, source_id)}"

    rotation = (
        f"rotation-v1:{_hash_text(tail, scheduled_rotation)}"
        if tail and scheduled_rotation
        else None
    )
    return canonical, rotation


def artifact_fingerprints(
    artifact_dir: Path | str,
    feature_cols: Sequence[str],
    cat_mapping: Mapping[str, Sequence[Any]],
    cat_cols: Sequence[str] = (),
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    mapping_json = json.dumps(cat_mapping, sort_keys=True, default=str, separators=(",", ":"))
    cat_set = set(cat_cols)
    schema_contract = [
        {"name": col, "type": "string" if col in cat_set else "float64"}
        for col in feature_cols
    ]
    schema_json = json.dumps(schema_contract, separators=(",", ":"))
    return {
        "model_version": artifact_dir.name,
        "artifact_model_sha256": _file_sha256(artifact_dir / "model.lgb"),
        "artifact_meta_sha256": _file_sha256(artifact_dir / "meta.joblib"),
        "feature_schema_hash": sha256(schema_json.encode("utf-8")).hexdigest(),
        "category_mapping_sha256": sha256(mapping_json.encode("utf-8")).hexdigest(),
        "category_mapping_json": mapping_json,
        "feature_count": len(feature_cols),
        "feature_order_json": json.dumps(list(feature_cols), separators=(",", ":")),
        "categorical_features_json": json.dumps(
            [col for col in feature_cols if col in cat_set], separators=(",", ":"),
        ),
    }


def _insert_outbox_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    event_type: str,
    event_date: str,
    created_at_utc: str,
    payload: Mapping[str, Any],
) -> int:
    stored_payload = dict(payload)
    payload_keys = sorted(stored_payload)
    canonical_payload = json.dumps(
        stored_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    stored_payload["payload_keys_json"] = json.dumps(
        payload_keys, separators=(",", ":"),
    )
    stored_payload["payload_sha256"] = sha256(
        canonical_payload.encode("utf-8"),
    ).hexdigest()
    payload_json = json.dumps(
        stored_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    existing = conn.execute(
        "SELECT payload_json FROM training_export_outbox WHERE event_id=?",
        (event_id,),
    ).fetchone()
    if existing is not None:
        existing_payload = json.loads(existing[0])
        if existing_payload.get("payload_sha256") != stored_payload["payload_sha256"]:
            raise ValueError(f"event_id collision with different payload: {event_id}")
        return 0
    delivered = conn.execute(
        "SELECT payload_sha256 FROM training_export_delivery WHERE event_id=?",
        (event_id,),
    ).fetchone()
    if delivered is not None:
        delivered_payload_sha256 = delivered[0]
        if delivered_payload_sha256 is None:
            raise ValueError(
                f"delivered event_id has no payload hash after migration: {event_id}"
            )
        if delivered_payload_sha256 != stored_payload["payload_sha256"]:
            raise ValueError(f"event_id collision with different payload: {event_id}")
        return 0
    before = conn.total_changes
    conn.execute(
        """INSERT OR IGNORE INTO training_export_outbox
           (event_id, event_type, event_date, schema_version, created_at_utc, payload_json)
           SELECT ?,?,?,?,?,?
           WHERE NOT EXISTS (
               SELECT 1 FROM training_export_delivery WHERE event_id=?
           )""",
        (
            event_id, event_type, event_date, TRAINING_SCHEMA_VERSION,
            created_at_utc, payload_json, event_id,
        ),
    )
    return conn.total_changes - before


@_atomic_training_batch
def enqueue_prediction_snapshots(
    conn: sqlite3.Connection,
    *,
    flight_frame: pd.DataFrame,
    raw_features: pd.DataFrame,
    model_features: pd.DataFrame,
    target_indices: Iterable[Any],
    contexts: Mapping[Any, Mapping[str, Any]],
    predicted_at_utc: str,
    run_id: int,
    data_source: str,
    artifact_dir: Path | str,
    feature_cols: Sequence[str],
    cat_cols: Sequence[str],
    cat_mapping: Mapping[str, Sequence[Any]],
) -> int:
    """Enqueue exact before/after-fallback feature snapshots transactionally."""
    ensure_training_store_schema(conn)
    fingerprints = artifact_fingerprints(
        artifact_dir, feature_cols, cat_mapping, cat_cols,
    )
    predicted_iso = _utc_iso(predicted_at_utc)
    if predicted_iso is None:
        raise ValueError(f"invalid predicted_at_utc: {predicted_at_utc!r}")
    event_date = predicted_iso[:10]
    cat_set = set(cat_cols)
    mapping_json = str(fingerprints.pop("category_mapping_json"))
    contract_key = (
        fingerprints["feature_schema_hash"],
        fingerprints["category_mapping_sha256"],
    )
    contract = {
        "feature_schema_hash": contract_key[0],
        "category_mapping_sha256": contract_key[1],
        "feature_order_json": fingerprints["feature_order_json"],
        "categorical_features_json": fingerprints["categorical_features_json"],
        "category_mapping_json": mapping_json,
    }
    contract_json = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    existing_contract = conn.execute(
        """SELECT contract_json FROM training_feature_contracts
           WHERE feature_schema_hash=? AND category_mapping_sha256=?""",
        contract_key,
    ).fetchone()
    if existing_contract is not None and existing_contract[0] != contract_json:
        raise ValueError(f"feature contract hash collision: {contract_key!r}")
    conn.execute(
        """INSERT OR IGNORE INTO training_feature_contracts
           (feature_schema_hash, category_mapping_sha256, contract_json, created_at_utc)
           VALUES (?,?,?,?)""",
        (*contract_key, contract_json, predicted_iso),
    )
    transformer_version = f"code-sha256:{transformer_code_sha256()}"
    source_commit = os.getenv("SOURCE_COMMIT", "").strip() or None

    inserted = 0
    for idx in target_indices:
        row = flight_frame.loc[idx]
        context = dict(contexts.get(idx, {}))
        canonical_key, rotation_key = build_flight_keys(row)
        fa_flight_id = _json_scalar(row.get("fa_flight_id"))
        stable = _json_scalar(row.get("stable_id"))
        scheduled_out = _utc_iso(row.get("scheduled_out_utc"))
        scheduled_in = _utc_iso(row.get("scheduled_in_utc"))
        estimated_out = _utc_iso(row.get("estimated_out_utc"))
        estimated_in = _utc_iso(row.get("estimated_in_utc"))
        event_id = "snapshot-v1:" + _hash_text(
            canonical_key, fa_flight_id, predicted_iso,
            fingerprints["artifact_model_sha256"], fingerprints["feature_schema_hash"],
        )

        raw_missing = 0
        model_missing = 0
        imputed_any = False
        feature_payload: dict[str, Any] = {}
        for col in feature_cols:
            raw_value = raw_features.at[idx, col] if col in raw_features.columns else None
            model_value = model_features.at[idx, col] if col in model_features.columns else None
            converter = _categorical_value if col in cat_set else _numeric_feature_value
            raw_scalar = converter(raw_value)
            model_scalar = converter(model_value)
            if raw_scalar is None:
                raw_missing += 1
            if model_scalar is None:
                model_missing += 1
            feature_payload[f"raw__{col}"] = raw_scalar
            feature_payload[f"model__{col}"] = model_scalar
            was_imputed = raw_scalar is None and model_scalar is not None
            feature_payload[f"imputed__{col}"] = was_imputed
            imputed_any = imputed_any or was_imputed

        payload: dict[str, Any] = {
            "event_id": event_id,
            "event_type": SNAPSHOT_EVENT_TYPE,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "identity_version": IDENTITY_VERSION,
            "horizon_definition_version": HORIZON_DEFINITION_VERSION,
            "canonical_flight_key": canonical_key,
            "rotation_key": rotation_key,
            "fa_flight_id": fa_flight_id,
            "stable_id": stable,
            "data_source": data_source,
            "carrier": _normal_text(row.get("op_carrier")),
            "flight_number": _normal_flight_number(row.get("flight_number")),
            "tail_num": _normal_text(row.get("tail_num")),
            "origin": _normal_text(row.get("origin")),
            "destination": _normal_text(row.get("dest")),
            "flight_date_local": _json_scalar(row.get("fl_date")),
            "scheduled_out_utc": scheduled_out,
            "scheduled_off_utc": _utc_iso(row.get("scheduled_off_utc")),
            "scheduled_on_utc": _utc_iso(row.get("scheduled_on_utc")),
            "scheduled_in_utc": scheduled_in,
            "estimated_out_utc": estimated_out,
            "estimated_in_utc": estimated_in,
            "source_flight_first_seen_utc": _utc_iso(row.get("first_seen_utc")),
            "source_flight_last_updated_utc": _utc_iso(row.get("last_updated_utc")),
            "origin_weather_valid_utc": _utc_iso(row.get("ORIG_WX_VALID_UTC")),
            "origin_weather_ingested_at_utc": _utc_iso(
                row.get("ORIG_WX_INGESTED_AT_UTC"),
            ),
            "destination_weather_valid_utc": _utc_iso(row.get("DEST_WX_VALID_UTC")),
            "destination_weather_ingested_at_utc": _utc_iso(
                row.get("DEST_WX_INGESTED_AT_UTC"),
            ),
            "predicted_at_utc": predicted_iso,
            "knowledge_cutoff_utc": predicted_iso,
            "lead_to_scheduled_out_min": _minutes_between(scheduled_out, predicted_iso),
            "lead_to_scheduled_in_min": _minutes_between(scheduled_in, predicted_iso),
            "prediction_phase": _json_scalar(context.get("prediction_phase")),
            "run_id": int(run_id),
            "cloud_run_job": os.getenv("CLOUD_RUN_JOB"),
            "cloud_run_execution": os.getenv("CLOUD_RUN_EXECUTION"),
            "cloud_run_task_index": os.getenv("CLOUD_RUN_TASK_INDEX"),
            "source_db_generation": os.getenv("LIVE_DB_BASE_GENERATION"),
            "transformer_version": transformer_version,
            "source_commit": source_commit,
            "target_definition": TARGET_DEFINITION,
            "raw_missing_count": raw_missing,
            "model_missing_count": model_missing,
            "raw_missing_rate": raw_missing / max(len(feature_cols), 1),
            "model_missing_rate": model_missing / max(len(feature_cols), 1),
            "fallback_applied": imputed_any,
            **fingerprints,
        }
        for key, value in context.items():
            if key not in payload:
                payload[key] = _json_scalar(value)
        availability_fields = (
            "source_flight_last_updated_utc",
            "origin_weather_valid_utc",
            "origin_weather_ingested_at_utc",
            "destination_weather_valid_utc",
            "destination_weather_ingested_at_utc",
            "nas_snapshot_captured_at_utc",
            "adsb_latest_captured_at_utc",
        )
        future_sources = []
        cutoff = pd.Timestamp(predicted_iso)
        for field in availability_fields:
            value = _utc_iso(payload.get(field))
            if value is not None and pd.Timestamp(value) > cutoff:
                future_sources.append(field)
        payload["causal_validation_status"] = (
            "failed_future_source"
            if future_sources
            else "pass_known_sources_partial"
        )
        payload["causal_future_sources_json"] = json.dumps(
            future_sources, separators=(",", ":"),
        )
        payload["provenance_completeness"] = "partial-v1"
        payload.update(feature_payload)
        inserted += _insert_outbox_event(
            conn,
            event_id=event_id,
            event_type=SNAPSHOT_EVENT_TYPE,
            event_date=event_date,
            created_at_utc=predicted_iso,
            payload=payload,
        )
    return inserted


@_atomic_training_batch
def enqueue_recent_outcomes(
    conn: sqlite3.Connection,
    *,
    observed_at_utc: datetime | str,
    lookback_days: int = 7,
    max_events: int | None = None,
) -> int:
    """Enqueue raw outcome revisions not already delivered.

    A rolling lookback makes the exporter self-healing after short outages.  The
    event ID includes the full revision, so corrected outcomes become new events
    while retries of the same revision remain idempotent.
    """
    ensure_training_store_schema(conn)
    observed_iso = _utc_iso(observed_at_utc)
    if observed_iso is None:
        raise ValueError(f"invalid observed_at_utc: {observed_at_utc!r}")
    if max_events is None:
        max_events = _positive_int_env(
            "TRAINING_OUTCOME_ENQUEUE_MAX_EVENTS", default=5_000,
        )
    elif max_events < 1:
        raise ValueError("max_events must be >= 1")
    since = pd.Timestamp(observed_iso) - pd.Timedelta(days=max(1, lookback_days))
    query = """
        WITH unambiguous_stable AS (
            SELECT stable_id
            FROM flights
            WHERE stable_id IS NOT NULL
            GROUP BY stable_id
            HAVING SUM(
                       CASE WHEN tail_num IS NULL OR scheduled_off_utc IS NULL
                            THEN 1 ELSE 0 END
                   ) = 0
               AND COUNT(
                       DISTINCT UPPER(TRIM(tail_num)) || '|' ||
                                strftime('%Y-%m-%dT%H:%M', scheduled_off_utc)
                   ) = 1
        ),
        candidate_flights AS (
            SELECT a.rowid AS actual_rowid, f.rowid AS flight_rowid,
                   CASE
                       WHEN NOT EXISTS (
                           SELECT 1 FROM training_outcome_state state
                           WHERE state.source_fa_flight_id = a.fa_flight_id
                       ) OR EXISTS (
                           SELECT 1 FROM training_outcome_state state
                           WHERE state.source_fa_flight_id = a.fa_flight_id
                             AND datetime(a.settled_at_utc) >
                                 datetime(state.last_observed_at_utc)
                       ) THEN 0 ELSE 1
                   END AS export_priority,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.rowid
                       ORDER BY CASE WHEN f.fa_flight_id = a.fa_flight_id THEN 0 ELSE 1 END,
                                datetime(f.last_updated_utc) DESC,
                                f.rowid DESC
                   ) AS candidate_rank
            FROM actuals a
            LEFT JOIN flights f
              ON f.fa_flight_id = a.fa_flight_id
              OR (
                  a.stable_id IS NOT NULL
                  AND f.stable_id = a.stable_id
                  AND a.stable_id IN (SELECT stable_id FROM unambiguous_stable)
              )
            WHERE datetime(a.settled_at_utc) >= datetime(?)
               OR NOT EXISTS (
                   SELECT 1
                   FROM training_outcome_state state
                   WHERE state.source_fa_flight_id = a.fa_flight_id
               )
               OR EXISTS (
                   SELECT 1
                   FROM training_outcome_state state
                   WHERE state.source_fa_flight_id = a.fa_flight_id
                     AND datetime(a.settled_at_utc) >
                         datetime(state.last_observed_at_utc)
               )
        )
        SELECT a.fa_flight_id, a.stable_id, a.actual_out_utc, a.actual_off_utc,
               a.actual_on_utc, a.actual_in_utc, a.arr_delay_min,
               a.departure_delay_min, a.cancelled, a.diverted, a.settled_at_utc,
               a.source_provider,
               f.op_carrier, f.flight_number, f.tail_num, f.origin, f.dest,
               f.fl_date, f.scheduled_out_utc, f.scheduled_off_utc,
               f.scheduled_on_utc, f.scheduled_in_utc
        FROM actuals a
        JOIN candidate_flights candidate
          ON candidate.actual_rowid = a.rowid AND candidate.candidate_rank = 1
        LEFT JOIN flights f ON f.rowid = candidate.flight_rowid
        ORDER BY candidate.export_priority, datetime(a.settled_at_utc), a.rowid
        LIMIT ?
    """
    rows = pd.read_sql_query(
        query, conn, params=(since.isoformat(), int(max_events)),
    )
    inserted = 0
    for _, row in rows.iterrows():
        canonical_key, rotation_key = build_flight_keys(row)
        revision_values = {
            "fa_flight_id": _json_scalar(row.get("fa_flight_id")),
            "stable_id": _json_scalar(row.get("stable_id")),
            "source_provider": _normal_text(row.get("source_provider")),
            "actual_out_utc": _utc_iso(row.get("actual_out_utc")),
            "actual_off_utc": _utc_iso(row.get("actual_off_utc")),
            "actual_on_utc": _utc_iso(row.get("actual_on_utc")),
            "actual_in_utc": _utc_iso(row.get("actual_in_utc")),
            "arr_delay_min": _json_scalar(row.get("arr_delay_min")),
            "departure_delay_min": _json_scalar(row.get("departure_delay_min")),
            "cancelled": _json_scalar(row.get("cancelled")),
            "diverted": _json_scalar(row.get("diverted")),
        }
        source_record_id = "live-actual-v1:" + _hash_text(
            revision_values["fa_flight_id"] or revision_values["stable_id"]
            or canonical_key,
        )
        state_json = json.dumps(
            revision_values, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        state_sha256 = sha256(state_json.encode("utf-8")).hexdigest()
        previous_state = conn.execute(
            """SELECT state_sha256, outcome_revision, source_fa_flight_id,
                      partition_flight_date
               FROM training_outcome_state WHERE source_record_id=?""",
            (source_record_id,),
        ).fetchone()
        if (
            previous_state is not None
            and previous_state[2] is not None
            and previous_state[2] != revision_values["fa_flight_id"]
        ):
            raise ValueError(
                "source_record_id is already associated with a different "
                f"fa_flight_id: {source_record_id}"
            )
        current_partition_date = _flight_date_partition(row.get("fl_date"))
        partition_flight_date = (
            str(previous_state[3])
            if previous_state is not None and previous_state[3]
            else current_partition_date
        )
        if previous_state is not None and previous_state[0] == state_sha256:
            # Existing databases gain source_fa_flight_id through a nullable
            # migration.  Backfill matching states even when no new revision is
            # needed, otherwise every old actual would remain "unseen" forever.
            conn.execute(
                "UPDATE training_outcome_state SET source_fa_flight_id=?, "
                "last_observed_at_utc=?, partition_flight_date=? "
                "WHERE source_record_id=?",
                (
                    revision_values["fa_flight_id"], observed_iso,
                    partition_flight_date, source_record_id,
                ),
            )
            continue
        outcome_revision = int(previous_state[1]) + 1 if previous_state else 1
        event_id = "outcome-v1:" + _hash_text(
            source_record_id, outcome_revision, state_sha256,
        )
        source_settled = _utc_iso(row.get("settled_at_utc"))
        if source_settled is not None and pd.Timestamp(source_settled) > pd.Timestamp(observed_iso):
            raise ValueError(
                "outcome observed_at_utc precedes source settled_at_utc for "
                f"{revision_values['fa_flight_id']!r}"
            )
        payload = {
            "event_id": event_id,
            "event_type": OUTCOME_EVENT_TYPE,
            "schema_version": TRAINING_SCHEMA_VERSION,
            "identity_version": IDENTITY_VERSION,
            "canonical_flight_key": canonical_key,
            "rotation_key": rotation_key,
            "observed_at_utc": observed_iso,
            "source_settled_at_utc": source_settled,
            "source_name": "live_data",
            "source_provider": _normal_text(row.get("source_provider")),
            "source_record_id": source_record_id,
            "outcome_revision": outcome_revision,
            "outcome_state_sha256": state_sha256,
            "partition_flight_date_local": partition_flight_date,
            "carrier": _normal_text(row.get("op_carrier")),
            "flight_number": _normal_flight_number(row.get("flight_number")),
            "tail_num": _normal_text(row.get("tail_num")),
            "origin": _normal_text(row.get("origin")),
            "destination": _normal_text(row.get("dest")),
            "flight_date_local": _json_scalar(row.get("fl_date")),
            "scheduled_out_utc": _utc_iso(row.get("scheduled_out_utc")),
            "scheduled_off_utc": _utc_iso(row.get("scheduled_off_utc")),
            "scheduled_on_utc": _utc_iso(row.get("scheduled_on_utc")),
            "scheduled_in_utc": _utc_iso(row.get("scheduled_in_utc")),
            "is_final": bool(
                revision_values["actual_in_utc"] is not None
                or revision_values["cancelled"] == 1
                or revision_values["diverted"] == 1
            ),
            **revision_values,
        }
        event_inserted = _insert_outbox_event(
            conn,
            event_id=event_id,
            event_type=OUTCOME_EVENT_TYPE,
            event_date=observed_iso[:10],
            created_at_utc=observed_iso,
            payload=payload,
        )
        # Reconcile the watermark even when the identical event was already
        # pending or delivered.  This repairs a missing/migrated state row and
        # prevents the same old actual from being selected on every cycle.
        conn.execute(
            """INSERT INTO training_outcome_state
               (source_record_id, source_fa_flight_id, state_sha256,
                outcome_revision, last_observed_at_utc, partition_flight_date)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(source_record_id) DO UPDATE SET
                   source_fa_flight_id=excluded.source_fa_flight_id,
                   state_sha256=excluded.state_sha256,
                   outcome_revision=excluded.outcome_revision,
                   last_observed_at_utc=excluded.last_observed_at_utc,
                   partition_flight_date=excluded.partition_flight_date""",
            (
                source_record_id, revision_values["fa_flight_id"], state_sha256,
                outcome_revision, observed_iso, partition_flight_date,
            ),
        )
        inserted += event_inserted
    return inserted


def pending_outbox_count(conn: sqlite3.Connection) -> int:
    ensure_training_store_schema(conn)
    return int(conn.execute("SELECT count(*) FROM training_export_outbox").fetchone()[0])


def prune_delivery_log(conn: sqlite3.Connection, *, keep_days: int = 14) -> int:
    ensure_training_store_schema(conn)
    before = conn.total_changes
    conn.execute(
        "DELETE FROM training_export_delivery "
        "WHERE datetime(delivered_at_utc) < datetime('now', ?)",
        (f"-{max(1, keep_days)} days",),
    )
    return conn.total_changes - before


def prune_outcome_state(conn: sqlite3.Connection, *, keep_days: int = 45) -> int:
    """Bound exported outcome watermarks without risking an undelivered event."""
    ensure_training_store_schema(conn)
    if conn.execute(
        "SELECT 1 FROM training_export_outbox LIMIT 1",
    ).fetchone() is not None:
        return 0
    before = conn.total_changes
    conn.execute(
        """DELETE FROM training_outcome_state
           WHERE source_fa_flight_id IS NOT NULL
             AND datetime(last_observed_at_utc) < datetime('now', ?)
             AND NOT EXISTS (
                 SELECT 1 FROM actuals
                 WHERE actuals.fa_flight_id =
                       training_outcome_state.source_fa_flight_id
             )""",
        (f"-{max(1, keep_days)} days",),
    )
    return conn.total_changes - before


_PARQUET_STRING_COLUMNS = {
    "event_id", "event_type", "schema_version", "identity_version",
    "horizon_definition_version", "canonical_flight_key", "rotation_key",
    "fa_flight_id", "stable_id", "data_source", "source_name", "source_provider", "carrier",
    "flight_number", "tail_num", "origin", "destination", "flight_date_local",
    "partition_flight_date_local",
    "scheduled_out_utc", "scheduled_off_utc", "scheduled_on_utc",
    "scheduled_in_utc", "estimated_out_utc", "estimated_in_utc",
    "source_flight_first_seen_utc", "source_flight_last_updated_utc",
    "origin_weather_valid_utc", "origin_weather_ingested_at_utc",
    "destination_weather_valid_utc", "destination_weather_ingested_at_utc",
    "actual_out_utc", "actual_off_utc", "actual_on_utc", "actual_in_utc",
    "predicted_at_utc", "knowledge_cutoff_utc", "observed_at_utc",
    "source_settled_at_utc", "prediction_phase", "cloud_run_job",
    "cloud_run_execution", "cloud_run_task_index", "source_db_generation",
    "transformer_version", "source_commit", "target_definition", "model_version",
    "artifact_model_sha256", "artifact_meta_sha256", "feature_schema_hash",
    "category_mapping_sha256", "feature_order_json",
    "categorical_features_json", "threshold_strategy", "payload_sha256",
    "category_mapping_json", "source_record_id", "outcome_state_sha256",
    "nas_snapshot_captured_at_utc", "adsb_latest_captured_at_utc",
    "causal_validation_status", "causal_future_sources_json",
    "provenance_completeness", "payload_keys_json",
}
_PARQUET_FLOAT_COLUMNS = {
    "lead_to_scheduled_out_min", "lead_to_scheduled_in_min", "raw_missing_rate",
    "model_missing_rate", "booster_probability", "calibrated_probability",
    "final_probability", "threshold_used", "probability_after_gdp",
    "probability_after_departure", "probability_after_adsb_eta",
    "gdp_orig_delay_min", "gdp_dest_delay_min", "estimated_dep_delay_min",
    "intermediate_dep_delay_min", "adsb_eta_delay_min", "adsb_holding_min",
    "carrier_delay_rate_smooth", "arr_delay_min", "departure_delay_min",
}
_PARQUET_INTEGER_COLUMNS = {
    "run_id", "feature_count", "raw_missing_count", "model_missing_count",
    "predicted_label", "atl_arrivals_in_window_30min", "cancelled", "diverted",
    "outcome_revision",
}
_PARQUET_BOOLEAN_COLUMNS = {
    "fallback_applied", "is_final", "gdp_adjust_enabled",
    "estimated_delay_adjust_enabled", "departure_delay_adjust_enabled",
    "adsb_adjust_enabled",
}


def _coerce_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply stable Arrow-compatible dtypes, including all-null shards."""
    result = frame.copy()
    feature_order: list[str] = []
    categorical: set[str] = set()
    if "feature_order_json" in result:
        values = result["feature_order_json"].dropna()
        if not values.empty:
            feature_order = list(json.loads(str(values.iloc[0])))
    if "categorical_features_json" in result:
        values = result["categorical_features_json"].dropna()
        if not values.empty:
            categorical = set(json.loads(str(values.iloc[0])))

    for col in result.columns:
        feature_name = None
        for prefix in ("raw__", "model__", "imputed__"):
            if col.startswith(prefix):
                feature_name = col.removeprefix(prefix)
                break
        if feature_name is not None:
            if col.startswith("imputed__"):
                result[col] = result[col].astype("boolean")
            elif feature_name in categorical:
                result[col] = result[col].astype("string")
            else:
                result[col] = pd.to_numeric(result[col], errors="coerce").astype("Float64")
        elif col in _PARQUET_STRING_COLUMNS:
            result[col] = result[col].astype("string")
        elif col in _PARQUET_FLOAT_COLUMNS:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Float64")
        elif col in _PARQUET_INTEGER_COLUMNS:
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        elif col in _PARQUET_BOOLEAN_COLUMNS:
            result[col] = result[col].astype("boolean")

    # A malformed payload should not silently omit a feature from the schema.
    expected = {
        f"{prefix}{name}"
        for name in feature_order
        for prefix in ("raw__", "model__", "imputed__")
    }
    missing = expected.difference(result.columns)
    if missing:
        raise ValueError(f"snapshot payload is missing feature fields: {sorted(missing)!r}")
    return result


def _upload_create_only(
    blob: Any,
    body: bytes,
    *,
    content_type: str,
    expected_metadata: Mapping[str, str],
    object_name: str,
) -> None:
    """Create an immutable object or verify an identical prior creation."""
    blob.metadata = dict(expected_metadata)
    try:
        blob.upload_from_string(
            body,
            content_type=content_type,
            if_generation_match=0,
        )
    except Exception as exc:
        from google.api_core.exceptions import PreconditionFailed

        if not isinstance(exc, PreconditionFailed):
            raise
        blob.reload()
        existing_metadata = blob.metadata or {}
        mismatches = {
            key: (existing_metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if existing_metadata.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"immutable training object collision for {object_name}: "
                f"metadata mismatch {mismatches!r}"
            ) from exc
        try:
            generation = int(blob.generation)
            existing_body = blob.download_as_bytes(
                if_generation_match=generation,
            )
        except Exception as verify_exc:
            raise RuntimeError(
                f"could not verify immutable training object {object_name}"
            ) from verify_exc
        expected_content_sha256 = sha256(body).hexdigest()
        existing_content_sha256 = sha256(existing_body).hexdigest()
        if existing_content_sha256 == expected_content_sha256:
            return

        # Parquet bytes can legitimately change across PyArrow versions while
        # representing the exact same ledger rows.  Verify the remote payload
        # semantically and cryptographically instead of wedging the outbox on an
        # encoder upgrade.  Invalid/corrupt bodies still fail closed.
        try:
            if content_type == "application/vnd.apache.parquet":
                remote_frame = pd.read_parquet(BytesIO(existing_body))
                _verify_event_frame_integrity(
                    remote_frame,
                    name=f"existing object {object_name}",
                )
                remote_identity = sorted(
                    (
                        str(event_id), str(payload_sha256),
                    )
                    for event_id, payload_sha256 in zip(
                        remote_frame["event_id"],
                        remote_frame["payload_sha256"],
                        strict=False,
                    )
                )
                remote_event_ids = [item[0] for item in remote_identity]
                remote_payload_hashes = [item[1] for item in remote_identity]
                remote_event_digest = sha256(
                    "\n".join(remote_event_ids).encode("utf-8")
                ).hexdigest()
                remote_logical_digest = sha256(
                    "\n".join(remote_payload_hashes).encode("utf-8")
                ).hexdigest()
                if (
                    len(remote_frame) != int(expected_metadata["event_count"])
                    or remote_event_digest != expected_metadata["event_digest"]
                    or remote_logical_digest != expected_metadata["logical_sha256"]
                ):
                    raise ValueError("remote Parquet logical digest mismatch")
                return
            if content_type == "application/json":
                if json.loads(existing_body) == json.loads(body):
                    return
                raise ValueError("remote JSON content differs")
        except Exception as semantic_exc:
            raise RuntimeError(
                f"immutable training object collision for {object_name}: "
                "content SHA-256 mismatch; semantic payload mismatch"
            ) from semantic_exc
        raise RuntimeError(
            f"immutable training object collision for {object_name}: "
            "content SHA-256 mismatch"
        ) from exc


def publish_pending_outbox(
    conn: sqlite3.Connection,
    *,
    bucket_name: str,
    prefix: str = "live-training",
    mark_delivered: bool,
    storage_client: Any | None = None,
    max_created_at_groups: int | None = None,
) -> PublishSummary:
    """Publish pending events as immutable Parquet shards.

    ``mark_delivered=False`` is used immediately after the winning SQLite upload:
    the events become durable in GCS but remain pending in that uploaded SQLite
    generation.  The next winning attempt publishes the same deterministic object
    and acknowledges/deletes the outbox rows locally before its single DB upload.
    """
    ensure_training_store_schema(conn)
    if max_created_at_groups is None:
        max_created_at_groups = _positive_int_env(
            "TRAINING_PUBLISH_MAX_GROUPS", default=8,
        )
    elif max_created_at_groups < 1:
        raise ValueError("max_created_at_groups must be >= 1")
    rows = conn.execute(
        """WITH selected_groups AS (
               SELECT created_at_utc
               FROM training_export_outbox
               GROUP BY created_at_utc
               ORDER BY datetime(created_at_utc), created_at_utc
               LIMIT ?
           )
           SELECT event_id, event_type, event_date, schema_version,
                  created_at_utc, payload_json
           FROM training_export_outbox
           WHERE created_at_utc IN (SELECT created_at_utc FROM selected_groups)
           ORDER BY datetime(created_at_utc), created_at_utc,
                    event_type, event_date, event_id""",
        (int(max_created_at_groups),),
    ).fetchall()
    if not rows:
        # mark_delivered=True is the mutating/acknowledgement phase.  Schema
        # migrations and outcome-anchor backfills performed above must survive
        # even when there are no pending events to acknowledge.
        if mark_delivered:
            conn.commit()
        return PublishSummary(events=0, objects=0, object_names=())
    if not bucket_name:
        raise ValueError("TRAINING_DATA_BUCKET is required to publish the outbox")

    if storage_client is None:
        from google.cloud import storage as gcs

        storage_client = gcs.Client()
    bucket = storage_client.bucket(bucket_name)

    grouped: dict[
        tuple[str, str, str, str, str, str, str, str],
        list[tuple[tuple, dict]],
    ] = {}
    for row in rows:
        payload = json.loads(row[5])
        flight_partition = "none"
        if row[1] == OUTCOME_EVENT_TYPE:
            flight_partition = _flight_date_partition(
                payload.get("partition_flight_date_local")
            )
        group_key = (
            row[1], row[2], row[3], row[4],
            str(payload.get("feature_schema_hash") or "none"),
            str(payload.get("category_mapping_sha256") or "none"),
            str(payload.get("model_version") or "none"),
            flight_partition,
        )
        grouped.setdefault(group_key, []).append((row, payload))

    delivered: list[tuple[str, str, str, str, str]] = []
    object_names: list[str] = []
    delivered_at = datetime.now(timezone.utc).isoformat()
    clean_prefix = prefix.strip("/")

    for group_key, group in grouped.items():
        (
            event_type, event_date, schema_version, created_at_utc, feature_schema_hash,
            category_mapping_sha256, model_version, flight_partition,
        ) = group_key
        payloads = [dict(payload) for _, payload in group]
        if event_type == SNAPSHOT_EVENT_TYPE:
            contract_row = conn.execute(
                """SELECT contract_json FROM training_feature_contracts
                   WHERE feature_schema_hash=? AND category_mapping_sha256=?""",
                (feature_schema_hash, category_mapping_sha256),
            ).fetchone()
            if contract_row is None:
                raise ValueError(
                    "missing feature contract for snapshot group "
                    f"{feature_schema_hash}/{category_mapping_sha256}"
                )
            contract = json.loads(contract_row[0])
            mapping_json = str(contract["category_mapping_json"])
            if sha256(mapping_json.encode("utf-8")).hexdigest() != category_mapping_sha256:
                raise ValueError("feature contract category mapping hash mismatch")
            for item in payloads:
                if (
                    item.get("feature_order_json") != contract["feature_order_json"]
                    or item.get("categorical_features_json")
                    != contract["categorical_features_json"]
                ):
                    raise ValueError("snapshot does not match normalized feature contract")
            contract_body = contract_row[0].encode("utf-8")
            contract_digest = sha256(contract_body).hexdigest()
            contract_name = (
                f"{clean_prefix}/contracts/schema={schema_version}/"
                f"contract-{feature_schema_hash}-{category_mapping_sha256}.json"
            )
            _upload_create_only(
                bucket.blob(contract_name),
                contract_body,
                content_type="application/json",
                expected_metadata={
                    "logical_sha256": contract_digest,
                    "schema_version": schema_version,
                    "feature_schema_hash": feature_schema_hash,
                    "category_mapping_sha256": category_mapping_sha256,
                },
                object_name=contract_name,
            )
        payloads.sort(key=lambda item: item["event_id"])
        event_digest = sha256(
            "\n".join(item["event_id"] for item in payloads).encode("utf-8")
        ).hexdigest()
        logical_digest = sha256(
            "\n".join(item["payload_sha256"] for item in payloads).encode("utf-8")
        ).hexdigest()
        frame = _coerce_parquet_frame(pd.DataFrame(payloads))
        buffer = BytesIO()
        frame.to_parquet(buffer, index=False, compression="zstd")
        body = buffer.getvalue()
        content_digest = sha256(body).hexdigest()
        base_name = (
            f"{clean_prefix}/raw/{event_type}/schema={schema_version}/"
            f"contract={feature_schema_hash[:16]}-{category_mapping_sha256[:16]}/"
        )
        if event_type == OUTCOME_EVENT_TYPE:
            partition_name = (
                f"flight_date={flight_partition}/observed_date={event_date}/"
            )
        else:
            partition_name = f"event_date={event_date}/"
        object_name = (
            f"{base_name}{partition_name}"
            f"part-{event_digest[:16]}-{logical_digest}.parquet"
        )
        blob = bucket.blob(object_name)
        expected_metadata = {
            "logical_sha256": logical_digest,
            "event_digest": event_digest,
            "event_count": str(len(payloads)),
            "created_at_utc": created_at_utc,
            "schema_version": schema_version,
            "feature_schema_hash": "" if feature_schema_hash == "none" else feature_schema_hash,
            "category_mapping_sha256": (
                "" if category_mapping_sha256 == "none" else category_mapping_sha256
            ),
            "model_version": "" if model_version == "none" else model_version,
        }
        _upload_create_only(
            blob,
            body,
            content_type="application/vnd.apache.parquet",
            expected_metadata=expected_metadata,
            object_name=object_name,
        )
        object_names.append(object_name)
        delivered.extend(
            (
                item["event_id"], delivered_at, object_name, content_digest,
                item["payload_sha256"],
            )
            for item in payloads
        )

    if mark_delivered:
        conn.executemany(
            """INSERT OR REPLACE INTO training_export_delivery
               (event_id, delivered_at_utc, object_name, content_sha256,
                payload_sha256)
               VALUES (?,?,?,?,?)""",
            delivered,
        )
        conn.executemany(
            "DELETE FROM training_export_outbox WHERE event_id=?",
            [(event_id,) for event_id, *_ in delivered],
        )
        prune_outcome_state(conn)
        prune_delivery_log(conn)
        conn.commit()

    return PublishSummary(
        events=len(delivered),
        objects=len(object_names),
        object_names=tuple(object_names),
    )


def _verify_event_frame_integrity(frame: pd.DataFrame, *, name: str) -> None:
    """Recompute the canonical payload hash for every raw ledger row."""
    for row_number, (_, row) in enumerate(frame.iterrows()):
        try:
            keys = json.loads(str(row["payload_keys_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} row {row_number} has invalid payload_keys_json") from exc
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"{name} row {row_number} payload keys must be strings")
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise ValueError(f"{name} row {row_number} missing hashed keys: {missing!r}")
        payload = {key: _json_scalar(row[key]) for key in keys}
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        actual_hash = sha256(canonical.encode("utf-8")).hexdigest()
        expected_hash = str(row["payload_sha256"])
        if actual_hash != expected_hash:
            event_id = row.get("event_id")
            raise ValueError(
                f"{name} payload integrity failure for event_id={event_id!r}"
            )


_DEPLOYMENT_CONTRACT_COLUMNS = (
    "model_version",
    "artifact_model_sha256",
    "artifact_meta_sha256",
    "transformer_version",
)


def _single_deployment_contract(
    frame: pd.DataFrame,
    *,
    name: str,
) -> dict[str, str]:
    """Require one model and transformer identity for a trainable dataset."""
    missing = set(_DEPLOYMENT_CONTRACT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"{name} missing deployment identity columns: {sorted(missing)!r}"
        )
    identities = frame[list(_DEPLOYMENT_CONTRACT_COLUMNS)].drop_duplicates()
    if len(identities) != 1:
        raise ValueError(
            f"{name} mixes {len(identities)} model/transformer deployments; "
            "partition or filter by model_version, artifact hashes and "
            "transformer_version"
        )
    row = identities.iloc[0]
    contract: dict[str, str] = {}
    for column in _DEPLOYMENT_CONTRACT_COLUMNS:
        value = _json_scalar(row[column])
        if value is None or not str(value).strip():
            raise ValueError(f"{name} has an empty deployment identity: {column}")
        contract[column] = str(value)
    return contract


def materialize_fixed_horizons(
    snapshots: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    horizons_hours: Sequence[float] = (2.0,),
    max_snapshot_age_minutes: float = 60.0,
    feature_contracts: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a leakage-safe silver dataset with one row per flight/horizon.

    Selection uses only the known scheduled departure cutoff.  It never chooses
    a cycle by probability or relative to the eventual actual departure.
    """
    normalized_horizons = [float(value) for value in horizons_hours]
    if not normalized_horizons:
        raise ValueError("at least one horizon is required")
    if any(not math.isfinite(value) or value < 0 for value in normalized_horizons):
        raise ValueError("horizons must be finite and >= 0 hours")
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError("horizons must be unique")
    if not math.isfinite(max_snapshot_age_minutes) or max_snapshot_age_minutes < 0:
        raise ValueError("max_snapshot_age_minutes must be finite and >= 0")

    if snapshots.empty or outcomes.empty:
        return pd.DataFrame(), {
            "snapshots": len(snapshots), "outcomes": len(outcomes), "rows": 0,
        }

    required_snapshot = {
        "event_id", "payload_sha256", "payload_keys_json", "schema_version",
        "canonical_flight_key",
        "rotation_key", "fa_flight_id", "stable_id", "prediction_phase",
        "predicted_at_utc",
        "scheduled_out_utc", "feature_schema_hash", "category_mapping_sha256",
        "feature_order_json", "categorical_features_json",
        "model_version", "artifact_model_sha256", "artifact_meta_sha256",
        "transformer_version", "target_definition",
        "model_missing_count", "causal_validation_status",
        "booster_probability", "calibrated_probability", "final_probability",
        "threshold_used", "threshold_strategy",
    }
    required_outcome = {
        "event_id", "payload_sha256", "payload_keys_json", "schema_version",
        "canonical_flight_key",
        "rotation_key", "fa_flight_id", "stable_id", "observed_at_utc",
        "is_final", "arr_delay_min",
        "cancelled", "diverted", "actual_off_utc", "actual_in_utc",
        "source_provider", "source_record_id", "outcome_revision",
        "outcome_state_sha256",
    }
    for name, frame, required in (
        ("snapshots", snapshots, required_snapshot),
        ("outcomes", outcomes, required_outcome),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)!r}")
        _verify_event_frame_integrity(frame, name=name)
        conflicts = frame.groupby("event_id", dropna=False)["payload_sha256"].nunique()
        conflicting_ids = conflicts[conflicts > 1]
        if not conflicting_ids.empty:
            raise ValueError(
                f"{name} contain conflicting payloads for event IDs: "
                f"{conflicting_ids.index[:5].tolist()!r}"
            )

    snap = snapshots.drop_duplicates("event_id", keep="last").copy()
    out = outcomes.drop_duplicates("event_id", keep="last").copy()
    target_definitions = set(snap["target_definition"].dropna())
    if target_definitions != {TARGET_DEFINITION}:
        raise ValueError(
            f"unsupported or mixed target_definition: {sorted(target_definitions)!r}"
        )
    snap = snap[
        snap["prediction_phase"].eq("PRE_DEPARTURE")
        & snap["schema_version"].eq(TRAINING_SCHEMA_VERSION)
        & snap["causal_validation_status"].eq("pass_known_sources_partial")
    ].copy()
    out = out[out["schema_version"].eq(TRAINING_SCHEMA_VERSION)].copy()
    out["observed_at_utc"] = pd.to_datetime(
        out["observed_at_utc"], errors="coerce", utc=True,
    )
    out["outcome_revision"] = pd.to_numeric(
        out["outcome_revision"], errors="coerce",
    )
    # Collapse each append-only source record to its latest revision before
    # identity reconciliation.  A correction can itself change schedule-derived
    # canonical/rotation keys; retaining the old identity would resurrect a
    # stale normal label after a later cancellation.
    out = out.sort_values(
        ["source_record_id", "outcome_revision", "observed_at_utc", "event_id"],
    ).groupby("source_record_id", as_index=False, sort=False).tail(1)

    # Reconcile outcome identities as connected components.  Different providers
    # routinely assign different canonical service identities to the same physical
    # rotation (for example, a prefixed flight number or a revised scheduled-out
    # time).  Grouping conflicts by canonical key alone would let those labels pass
    # as two independent flights.  Rotation is safe as a bridge here because it is
    # tail + scheduled-off minute; stable_id is deliberately excluded because it is
    # reused across unrelated physical rotations in production.
    IdentityNode = tuple[str, str]
    parents: dict[IdentityNode, IdentityNode] = {}

    def _identity_node(kind: str, value: Any) -> IdentityNode | None:
        scalar = _json_scalar(value)
        if scalar is None:
            return None
        text = str(scalar).strip()
        return (kind, text) if text else None

    def _find(node: IdentityNode) -> IdentityNode:
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def _union(left: IdentityNode, right: IdentityNode) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root == right_root:
            return
        # A stable ordering makes the component independent of input row order.
        lower, higher = sorted((left_root, right_root))
        parents[higher] = lower

    for canonical_value, rotation_value in zip(
        out["canonical_flight_key"], out["rotation_key"], strict=False,
    ):
        canonical_node = _identity_node("canonical", canonical_value)
        rotation_node = _identity_node("rotation", rotation_value)
        if canonical_node is not None:
            _find(canonical_node)
        if rotation_node is not None:
            _find(rotation_node)
        if canonical_node is not None and rotation_node is not None:
            _union(canonical_node, rotation_node)

    component_nodes: dict[IdentityNode, set[IdentityNode]] = {}
    for node in parents:
        component_nodes.setdefault(_find(node), set()).add(node)
    node_to_training_key: dict[IdentityNode, str] = {}
    for nodes in component_nodes.values():
        canonical_values = sorted(value for kind, value in nodes if kind == "canonical")
        rotation_values = sorted(value for kind, value in nodes if kind == "rotation")
        representative = (
            canonical_values[0]
            if canonical_values
            else f"{IDENTITY_VERSION}:rotation-component:{_hash_text(*rotation_values)}"
        )
        for node in nodes:
            node_to_training_key[node] = representative

    canonical_to_training_key = {
        value: training_key
        for (kind, value), training_key in node_to_training_key.items()
        if kind == "canonical"
    }
    rotation_to_training_key = {
        value: training_key
        for (kind, value), training_key in node_to_training_key.items()
        if kind == "rotation"
    }
    out["_training_flight_key"] = [
        canonical_to_training_key.get(str(canonical).strip())
        or rotation_to_training_key.get(str(rotation).strip())
        for canonical, rotation in zip(
            out["canonical_flight_key"], out["rotation_key"], strict=False,
        )
    ]
    if out["_training_flight_key"].isna().any():
        raise ValueError("outcomes contain no usable canonical or rotation identity")
    snap["_training_flight_key"] = snap["canonical_flight_key"]

    resolved = pd.Series(False, index=snap.index)
    exact_aliases = out[
        out["fa_flight_id"].notna() & out["fa_flight_id"].ne("")
    ].sort_values(
        ["observed_at_utc", "outcome_revision", "event_id"],
    ).drop_duplicates(
        "fa_flight_id", keep="last",
    )
    exact_to_training_key = dict(
        zip(
            exact_aliases["fa_flight_id"],
            exact_aliases["_training_flight_key"],
            strict=False,
        )
    )
    exact_match = snap["fa_flight_id"].map(exact_to_training_key)
    exact_mask = exact_match.notna()
    snap.loc[exact_mask, "_training_flight_key"] = exact_match.loc[exact_mask]
    resolved |= exact_mask

    canonical_match = snap["canonical_flight_key"].map(canonical_to_training_key)
    canonical_mask = (~resolved) & canonical_match.notna()
    snap.loc[canonical_mask, "_training_flight_key"] = canonical_match.loc[
        canonical_mask
    ]
    resolved |= canonical_mask

    # ``stable_id`` is deliberately not a fallback: production contains reused
    # stable IDs spanning multiple physical rotations and conflicting labels.
    rotation_match = snap["rotation_key"].map(rotation_to_training_key)
    rotation_mask = (~resolved) & rotation_match.notna()
    snap.loc[rotation_mask, "_training_flight_key"] = rotation_match.loc[
        rotation_mask
    ]
    resolved |= rotation_mask

    for frame, columns in (
        (snap, ["predicted_at_utc", "scheduled_out_utc"]),
        (out, ["observed_at_utc", "actual_off_utc", "actual_in_utc"]),
    ):
        for col in columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    out["arr_delay_min"] = pd.to_numeric(out["arr_delay_min"], errors="coerce")
    out["cancelled"] = pd.to_numeric(out["cancelled"], errors="coerce").fillna(0)
    out["diverted"] = pd.to_numeric(out["diverted"], errors="coerce").fillna(0)
    # Compare the latest revision of every source before choosing one provider.
    # A later observation time is not evidence that a conflicting source is more
    # correct.  Until provider lineage/authority is stored, quarantine any state
    # or continuous-delay disagreement, even when sources settled on different days.
    grouped_outcomes = out.groupby("_training_flight_key", dropna=False)
    delay_spread = grouped_outcomes["arr_delay_min"].agg(
        lambda values: (
            float(values.dropna().max() - values.dropna().min())
            if len(values.dropna()) > 1
            else 0.0
        )
    )
    delay_presence_counts = grouped_outcomes["arr_delay_min"].agg(
        lambda values: values.notna().nunique(),
    )
    state_signature = out[
        ["cancelled", "diverted", "is_final"]
    ].astype("string").fillna("<null>").agg("\x1f".join, axis=1)
    out["_outcome_state_signature"] = state_signature
    state_counts = grouped_outcomes["_outcome_state_signature"].nunique()
    conflicting_outcome_keys = (
        set(delay_spread[delay_spread > 1e-6].index)
        | set(delay_presence_counts[delay_presence_counts > 1].index)
        | set(state_counts[state_counts > 1].index)
    )
    conflicting_provider_counts = {
        str(provider): int(count)
        for provider, count in out[
            out["_training_flight_key"].isin(conflicting_outcome_keys)
        ]["source_provider"].fillna("UNKNOWN").value_counts().items()
    }
    latest_candidates = out[
        ~out["_training_flight_key"].isin(conflicting_outcome_keys)
    ].copy()
    out = latest_candidates.sort_values(
        ["observed_at_utc", "outcome_revision", "event_id"],
    ).groupby("_training_flight_key", as_index=False, sort=False).tail(1)
    out = out[
        out["is_final"].fillna(False)
        & out["arr_delay_min"].notna()
        & out["cancelled"].eq(0)
        & out["diverted"].eq(0)
    ].copy()

    if snap.empty or out.empty:
        return pd.DataFrame(), {
            "snapshots": len(snapshots), "outcomes": len(outcomes), "rows": 0,
            "eligible_outcomes": len(out),
            "conflicting_outcomes": len(conflicting_outcome_keys),
            "conflicting_outcome_provider_counts": conflicting_provider_counts,
        }

    outcome_columns = out[
        [
            "_training_flight_key", "event_id", "payload_sha256",
            "source_provider", "source_record_id", "outcome_revision",
            "outcome_state_sha256",
            "arr_delay_min", "actual_off_utc", "actual_in_utc", "observed_at_utc",
        ]
    ].rename(
        columns={
            "event_id": "outcome_event_id",
            "payload_sha256": "outcome_payload_sha256",
            "source_provider": "outcome_source_provider",
            "source_record_id": "outcome_source_record_id",
            "outcome_revision": "outcome_revision",
            "outcome_state_sha256": "outcome_state_sha256",
            "observed_at_utc": "outcome_observed_at_utc",
        }
    )
    merged = snap.merge(
        outcome_columns,
        on="_training_flight_key",
        how="inner",
        validate="many_to_one",
    )
    merged = merged[
        merged["predicted_at_utc"].notna()
        & merged["scheduled_out_utc"].notna()
        & (
            merged["actual_off_utc"].isna()
            | (merged["predicted_at_utc"] < merged["actual_off_utc"])
        )
        & (
            merged["actual_in_utc"].isna()
            | (merged["predicted_at_utc"] < merged["actual_in_utc"])
        )
    ].copy()

    selected_parts: list[pd.DataFrame] = []
    horizon_counts: dict[str, int] = {}
    for horizon in normalized_horizons:
        cutoff = merged["scheduled_out_utc"] - pd.to_timedelta(float(horizon), unit="h")
        age_min = (cutoff - merged["predicted_at_utc"]).dt.total_seconds() / 60.0
        eligible = merged[
            age_min.ge(0) & age_min.le(float(max_snapshot_age_minutes))
        ].copy()
        eligible["_snapshot_age_at_cutoff_min"] = age_min.loc[eligible.index]
        eligible["_horizon_hours"] = float(horizon)
        eligible["_model_missing_sort"] = pd.to_numeric(
            eligible["model_missing_count"], errors="coerce",
        ).fillna(float("inf"))
        eligible = eligible.sort_values(
            ["predicted_at_utc", "_model_missing_sort", "event_id"],
            ascending=[True, False, True],
        ).groupby(
            "_training_flight_key", as_index=False, sort=False,
        ).tail(1)
        horizon_counts[str(horizon)] = len(eligible)
        selected_parts.append(eligible)

    if not selected_parts:
        return pd.DataFrame(), {
            "snapshots": len(snapshots), "outcomes": len(outcomes), "rows": 0,
        }
    selected = pd.concat(selected_parts, ignore_index=True)
    if selected.empty:
        return pd.DataFrame(), {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "snapshots": len(snapshots),
            "unique_snapshots": len(snap),
            "outcomes": len(outcomes),
            "eligible_outcomes": len(out),
            "conflicting_outcomes": len(conflicting_outcome_keys),
            "conflicting_outcome_provider_counts": conflicting_provider_counts,
            "joined_cycles": len(merged),
            "rows": 0,
            "horizon_counts": horizon_counts,
            "positive_rate": None,
        }
    selected["TARGET"] = (selected["arr_delay_min"] > 15.0).astype(np.int8)

    deployment_contract = _single_deployment_contract(
        selected,
        name="materialization",
    )

    # Preserve the explicit artifact order.  JSON serialization sorts object
    # keys, so relying on Parquet/DataFrame column order would silently change
    # the LightGBM contract.
    schema_contracts = selected[
        [
            "feature_schema_hash", "category_mapping_sha256",
            "feature_order_json", "categorical_features_json",
        ]
    ].drop_duplicates()
    if len(schema_contracts) != 1:
        raise ValueError(
            "cannot mix feature schemas/category mappings in one materialization; "
            f"found {len(schema_contracts)} contracts"
        )
    schema_row = schema_contracts.iloc[0]
    feature_order = json.loads(str(schema_row["feature_order_json"]))
    if not isinstance(feature_order, list) or not all(
        isinstance(col, str) for col in feature_order
    ):
        raise ValueError("feature_order_json must contain a JSON list of names")
    contract_key = (
        str(schema_row["feature_schema_hash"]),
        str(schema_row["category_mapping_sha256"]),
    )
    if feature_contracts is not None:
        contract_source = feature_contracts.get(contract_key)
        if contract_source is None:
            raise ValueError(f"missing feature contract sidecar: {contract_key!r}")
        feature_contract = dict(contract_source)
    elif "category_mapping_json" in selected.columns:
        mapping_values = selected["category_mapping_json"].dropna().drop_duplicates()
        if len(mapping_values) != 1:
            raise ValueError("embedded category mapping must be present and unique")
        feature_contract = {
            "feature_schema_hash": contract_key[0],
            "category_mapping_sha256": contract_key[1],
            "feature_order_json": str(schema_row["feature_order_json"]),
            "categorical_features_json": str(schema_row["categorical_features_json"]),
            "category_mapping_json": str(mapping_values.iloc[0]),
        }
    else:
        raise ValueError(f"feature contract sidecar is required: {contract_key!r}")
    if (
        str(feature_contract.get("feature_schema_hash")) != contract_key[0]
        or str(feature_contract.get("category_mapping_sha256")) != contract_key[1]
        or str(feature_contract.get("feature_order_json"))
        != str(schema_row["feature_order_json"])
        or str(feature_contract.get("categorical_features_json"))
        != str(schema_row["categorical_features_json"])
    ):
        raise ValueError("feature contract sidecar does not match snapshot contract")
    mapping_json = str(feature_contract.get("category_mapping_json"))
    if sha256(mapping_json.encode("utf-8")).hexdigest() != contract_key[1]:
        raise ValueError("feature contract category mapping SHA-256 mismatch")
    model_cols = [f"model__{col}" for col in feature_order]
    raw_cols = [f"raw__{col}" for col in feature_order]
    imputed_cols = [f"imputed__{col}" for col in feature_order]
    missing_feature_cols = [
        col for col in model_cols + raw_cols + imputed_cols
        if col not in selected.columns
    ]
    if missing_feature_cols:
        raise ValueError(
            f"materialization is missing captured features: {missing_feature_cols!r}"
        )
    metadata_cols = [
        "event_id", "canonical_flight_key", "rotation_key",
        "_training_flight_key", "fa_flight_id", "stable_id",
        "predicted_at_utc", "scheduled_out_utc", "actual_off_utc",
        "arr_delay_min", "outcome_event_id", "outcome_payload_sha256",
        "outcome_source_record_id", "outcome_revision", "outcome_state_sha256",
        "outcome_source_provider",
        "outcome_observed_at_utc", "_horizon_hours", "_snapshot_age_at_cutoff_min",
        "model_version", "artifact_model_sha256", "feature_schema_hash",
        "artifact_meta_sha256", "category_mapping_sha256",
        "feature_order_json", "categorical_features_json",
        "transformer_version", "target_definition", "schema_version", "TARGET",
        "causal_validation_status", "provenance_completeness",
    ]
    if "source_commit" in selected.columns:
        metadata_cols.insert(metadata_cols.index("target_definition"), "source_commit")
    operational_context_cols = [
        "booster_probability", "calibrated_probability", "final_probability",
        "predicted_label", "threshold_used", "threshold_strategy",
        "probability_after_gdp", "probability_after_departure",
        "probability_after_adsb_eta", "gdp_orig_delay_min", "gdp_dest_delay_min",
        "estimated_dep_delay_min", "intermediate_dep_delay_min",
        "adsb_eta_delay_min", "adsb_holding_min",
        "atl_arrivals_in_window_30min", "carrier_delay_rate_smooth",
        "fallback_applied", "gdp_adjust_enabled", "estimated_delay_adjust_enabled",
        "departure_delay_adjust_enabled", "adsb_adjust_enabled",
        "raw_missing_count", "model_missing_count", "raw_missing_rate",
        "model_missing_rate", "estimated_out_utc", "estimated_in_utc",
        "source_flight_first_seen_utc", "source_flight_last_updated_utc",
        "origin_weather_valid_utc", "origin_weather_ingested_at_utc",
        "destination_weather_valid_utc", "destination_weather_ingested_at_utc",
        "nas_snapshot_captured_at_utc", "adsb_latest_captured_at_utc",
        "causal_future_sources_json",
    ]
    operational_context_cols = [
        col for col in operational_context_cols
        if col in selected.columns and col not in metadata_cols
    ]
    if "flight_date_local" in selected.columns:
        metadata_cols.insert(6, "flight_date_local")
    result = selected[
        metadata_cols + operational_context_cols + model_cols + raw_cols + imputed_cols
    ].copy()
    result = result.rename(columns={col: col.removeprefix("model__") for col in model_cols})
    result = result.sort_values(
        ["scheduled_out_utc", "_training_flight_key", "_horizon_hours"],
    )
    result = result.reset_index(drop=True)
    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "snapshots": len(snapshots),
        "unique_snapshots": len(snap),
        "outcomes": len(outcomes),
        "eligible_outcomes": len(out),
        "conflicting_outcomes": len(conflicting_outcome_keys),
        "conflicting_outcome_provider_counts": conflicting_provider_counts,
        "joined_cycles": len(merged),
        "rows": len(result),
        "horizon_counts": horizon_counts,
        "positive_rate": float(result["TARGET"].mean()) if len(result) else None,
        "feature_contract": feature_contract,
        "deployment_contract": deployment_contract,
    }
    return result, report


def prepare_silver_training_data(
    silver: pd.DataFrame,
    *,
    horizon_hours: float | None = None,
    representation: str = "model_exact",
    feature_contract: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Reconstruct a LightGBM-ready ``X, y`` from a silver materialization.

    Categorical values remain readable strings in Parquet.  This adapter applies
    the captured category mapping and exact feature order before training, so a
    caller never depends on incidental Parquet column ordering or pandas codes.
    """
    required = {
        "TARGET", "feature_order_json", "categorical_features_json",
        "category_mapping_sha256", "feature_schema_hash", "target_definition",
        "arr_delay_min", *_DEPLOYMENT_CONTRACT_COLUMNS,
        "_horizon_hours",
    }
    missing = required.difference(silver.columns)
    if missing:
        raise ValueError(f"silver dataset missing contract columns: {sorted(missing)!r}")
    if silver.empty:
        raise ValueError("silver dataset is empty")
    available_horizons = sorted(
        pd.to_numeric(silver["_horizon_hours"], errors="raise").unique().tolist(),
    )
    if horizon_hours is None:
        if len(available_horizons) != 1:
            raise ValueError(
                "select exactly one horizon before training; available horizons: "
                f"{available_horizons!r}"
            )
        selected_silver = silver
        selected_horizon = float(available_horizons[0])
    else:
        selected_horizon = float(horizon_hours)
        selected_silver = silver[
            pd.to_numeric(silver["_horizon_hours"], errors="coerce").eq(
                selected_horizon,
            )
        ].copy()
        if selected_silver.empty:
            raise ValueError(
                f"horizon {selected_horizon} not present; available: {available_horizons!r}"
            )
    deployment_contract = _single_deployment_contract(
        selected_silver,
        name="silver dataset",
    )
    contracts = selected_silver[
        [
            "feature_order_json", "categorical_features_json",
            "category_mapping_sha256", "feature_schema_hash", "target_definition",
        ]
    ].drop_duplicates()
    if len(contracts) != 1:
        raise ValueError(f"silver dataset mixes {len(contracts)} feature contracts")

    contract_row = contracts.iloc[0]
    feature_order = json.loads(str(contract_row["feature_order_json"]))
    categorical_features = json.loads(
        str(contract_row["categorical_features_json"]),
    )
    if feature_contract is None:
        if "category_mapping_json" not in selected_silver.columns:
            raise ValueError(
                "feature_contract from the materialization manifest is required"
            )
        embedded = selected_silver["category_mapping_json"].dropna().drop_duplicates()
        if len(embedded) != 1:
            raise ValueError("embedded category mapping must be unique")
        category_mapping_json = str(embedded.iloc[0])
    else:
        if (
            str(feature_contract.get("feature_schema_hash"))
            != str(contract_row["feature_schema_hash"])
            or str(feature_contract.get("category_mapping_sha256"))
            != str(contract_row["category_mapping_sha256"])
            or str(feature_contract.get("feature_order_json"))
            != str(contract_row["feature_order_json"])
            or str(feature_contract.get("categorical_features_json"))
            != str(contract_row["categorical_features_json"])
        ):
            raise ValueError("feature_contract does not match silver metadata")
        category_mapping_json = str(feature_contract.get("category_mapping_json"))
    category_mapping = json.loads(category_mapping_json)
    mapping_hash = sha256(category_mapping_json.encode("utf-8")).hexdigest()
    if mapping_hash != str(contract_row["category_mapping_sha256"]):
        raise ValueError("category mapping content does not match its SHA-256")
    if set(selected_silver["target_definition"].dropna()) != {TARGET_DEFINITION}:
        raise ValueError("unsupported or mixed target_definition in silver dataset")
    if representation not in {"model_exact", "raw_challenger"}:
        raise ValueError("representation must be 'model_exact' or 'raw_challenger'")
    source_columns = (
        feature_order
        if representation == "model_exact"
        else [f"raw__{col}" for col in feature_order]
    )
    missing_features = [col for col in source_columns if col not in selected_silver.columns]
    if missing_features:
        raise ValueError(f"silver dataset missing features: {missing_features!r}")

    X = selected_silver[source_columns].copy()
    if representation == "raw_challenger":
        X.columns = feature_order
    categorical_set = set(categorical_features)
    for col in feature_order:
        if col in categorical_set:
            if representation == "model_exact":
                categories = category_mapping.get(col)
                if categories is None:
                    raise ValueError(f"missing category mapping for {col!r}")
                X[col] = pd.Categorical(X[col], categories=categories)
            else:
                # Keep readable strings, including categories unknown to the
                # champion.  Fit vocabularies on the TRAIN split only.
                X[col] = X[col].astype("string")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    y = pd.to_numeric(selected_silver["TARGET"], errors="raise").astype(np.int8)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("TARGET must be binary {0,1}")
    expected_target = (
        pd.to_numeric(selected_silver["arr_delay_min"], errors="raise") > 15.0
    ).astype(np.int8)
    if not np.array_equal(y.to_numpy(), expected_target.to_numpy()):
        raise ValueError("TARGET is inconsistent with arr_delay_min > 15")
    contract = {
        "feature_order": feature_order,
        "categorical_features": categorical_features,
        "category_mapping": category_mapping,
        "feature_schema_hash": str(contract_row["feature_schema_hash"]),
        "category_mapping_sha256": mapping_hash,
        "horizon_hours": selected_horizon,
        "representation": representation,
        "requires_train_only_category_fit": representation == "raw_challenger",
        **deployment_contract,
    }
    return X, y, contract


def temporal_split_silver(
    silver: pd.DataFrame,
    *,
    train_end_utc: str | datetime,
    validation_end_utc: str | datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by explicit calendar cutoffs while keeping each physical flight whole."""
    required = {"scheduled_out_utc", "_training_flight_key"}
    missing = required.difference(silver.columns)
    if missing:
        raise ValueError(f"silver dataset missing split columns: {sorted(missing)!r}")
    train_end = pd.Timestamp(train_end_utc)
    validation_end = pd.Timestamp(validation_end_utc)
    train_end = train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    validation_end = (
        validation_end.tz_localize("UTC")
        if validation_end.tzinfo is None
        else validation_end.tz_convert("UTC")
    )
    if train_end >= validation_end:
        raise ValueError("train_end_utc must be earlier than validation_end_utc")

    frame = silver.copy()
    scheduled = pd.to_datetime(frame["scheduled_out_utc"], errors="coerce", utc=True)
    if scheduled.isna().any():
        raise ValueError("scheduled_out_utc contains invalid/null values")
    frame["_split"] = np.where(
        scheduled < train_end,
        "train",
        np.where(scheduled < validation_end, "validation", "test"),
    )
    split_counts = frame.groupby("_training_flight_key")["_split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("at least one physical flight crosses temporal splits")
    parts = tuple(
        frame[frame["_split"].eq(name)].drop(columns="_split").reset_index(drop=True)
        for name in ("train", "validation", "test")
    )
    return parts  # type: ignore[return-value]
