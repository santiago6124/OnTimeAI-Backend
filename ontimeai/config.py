"""Configuration constants and training dataclass for OnTimeAI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_data_path() -> Path:
    """Pick the widest-year master CSV available in the project root."""
    candidates = sorted(PROJECT_ROOT.glob("dataset_maestro_ATL_*_BTS_IEM_ORIG_DEST.csv"))
    if not candidates:
        return PROJECT_ROOT / "dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv"
    # Prefer a multi-year master (contains a '-' in the year label).
    multi = [p for p in candidates if "-" in p.name.split("ATL_")[-1].split("_")[0]]
    return (multi[-1] if multi else candidates[-1])


DATA_PATH = _default_data_path()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Post-hoc columns that leak the label (unavailable at CRS_DEP_TIME in production).
LEAKY_COLS: tuple[str, ...] = (
    "DEP_TIME",
    "ARR_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "ARR_DELAY_NEW",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
)

FILTER_COLS: tuple[str, ...] = ("CANCELLED", "DIVERTED")

# Columns dropped before the final feature matrix (IDs, redundant datetimes,
# raw scheduled-time fields replaced by cyclical encodings).
DROP_COLS: tuple[str, ...] = (
    "YEAR",
    "FL_DATE",
    "DEP_LOCAL_DT",
    "EVENT_ORIGIN_UTC",
    "EVENT_DEST_UTC",
    "ORIG_WX_VALID_UTC",
    "DEST_WX_VALID_UTC",
    "OP_CARRIER_FL_NUM",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
)

CATEGORICAL_COLS: tuple[str, ...] = (
    "OP_CARRIER",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "PAR_AIRPORT",
    "FLOW_ATL",
    "ORIG_WX_CODES",
    "DEST_WX_CODES",
)

TARGET_COL = "TARGET"
ARR_DELAY_COL = "ARR_DELAY"

MULTICLASS_BINS: tuple[float, ...] = (-float("inf"), 15.0, 30.0, 60.0, float("inf"))
MULTICLASS_LABELS: tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class TrainConfig:
    target: str = "binary"
    delay_threshold_min: float = 15.0
    train_frac: float = 0.60
    val_frac: float = 0.20
    congestion_window_min: int = 30
    random_state: int = 42
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    balance_classes: bool = True
    tune_threshold_metric: str = "f1"
    use_tail_lineage: bool = True
    use_carrier_lag: bool = True
    use_origin_lag: bool = True
    use_carrier_rolling: bool = True
    use_origin_rolling: bool = True
    calibration: str | None = "isotonic"  # "isotonic" | "sigmoid" | None
    artifacts_dir: Path = ARTIFACTS_DIR
    lgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 200,
            "verbose": -1,
            "seed": 42,
        }
    )

    def resolved_lgb_params(self) -> dict[str, Any]:
        params = dict(self.lgb_params)
        if self.target == "binary":
            params["objective"] = "binary"
            params["metric"] = ["auc", "binary_logloss"]
        elif self.target == "multiclass":
            params["objective"] = "multiclass"
            params["num_class"] = len(MULTICLASS_LABELS)
            params["metric"] = ["multi_logloss"]
        else:
            raise ValueError(f"Unsupported target: {self.target}")
        params["seed"] = self.random_state
        return params
