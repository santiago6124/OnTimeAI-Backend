"""Stateful weighted airport sampler for unbiased live pulls.

Each airport is assigned a weight proportional to sqrt(BTS flight volume 2022-2025).
Selection score = elapsed_since_last_pull / ideal_interval, where
ideal_interval = BASE_INTERVAL / weight. Higher-weight airports are expected
to be pulled more often; all airports get covered over time (weighted round-robin).

State is persisted to a JSON file so priority carries across ticks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Top-30 US airports weighted by sqrt(BTS volume 2022-2025), normalized to ATL=1.0.
# Source: dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet (27.5M flights)
AIRPORT_WEIGHTS: dict[str, float] = {
    "ATL": 1.000, "DFW": 0.957, "DEN": 0.954, "ORD": 0.928,
    "CLT": 0.783, "LAX": 0.770, "LAS": 0.754, "PHX": 0.751,
    "SEA": 0.714, "MCO": 0.697, "LGA": 0.695, "DCA": 0.658,
    "BOS": 0.658, "SFO": 0.648, "EWR": 0.631, "DTW": 0.623,
    "JFK": 0.616, "MSP": 0.607, "IAH": 0.602, "SLC": 0.587,
    "MIA": 0.577, "BNA": 0.548, "BWI": 0.539, "PHL": 0.538,
    "SAN": 0.532, "AUS": 0.523, "FLL": 0.522, "MDW": 0.493,
    "TPA": 0.488, "DAL": 0.468,
}

# Baseline: ATL (weight=1.0) will be pulled every BASE_INTERVAL_S seconds.
# All other airports have ideal_interval = BASE_INTERVAL_S / weight.
BASE_INTERVAL_S: float = 1800.0  # 30 min — matches default cron cadence

STATE_FILE_DEFAULT: Path = Path(__file__).parent / "sampler_state.json"


class AirportSampler:
    """Selects the K most-overdue airports each tick.

    Overdue score = (now - last_pull) / (BASE_INTERVAL / weight).
    At cold start all airports score proportionally to their weight,
    so the first tick naturally picks the highest-volume airports.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        base_interval: float = BASE_INTERVAL_S,
        state_file: Path = STATE_FILE_DEFAULT,
    ) -> None:
        self.weights = weights or AIRPORT_WEIGHTS
        self.base_interval = base_interval
        self.state_file = Path(state_file)
        self._state: dict[str, float] = self._load()

    def _load(self) -> dict[str, float]:
        if self.state_file.exists():
            try:
                raw = json.loads(self.state_file.read_text())
                # Initialise missing airports at epoch 0 (maximally overdue)
                return {ap: float(raw.get(ap, 0.0)) for ap in self.weights}
            except Exception:
                pass
        return {ap: 0.0 for ap in self.weights}

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def overdue_score(self, airport: str, now: float) -> float:
        """Higher = more overdue relative to its ideal pull interval."""
        weight = self.weights[airport]
        ideal_interval = self.base_interval / weight
        elapsed = now - self._state.get(airport, 0.0)
        return elapsed / ideal_interval

    def select(self, k: int, now: float | None = None) -> list[str]:
        """Return top-k airports sorted by overdue score (highest first)."""
        now = now or time.time()
        return sorted(
            self.weights,
            key=lambda ap: self.overdue_score(ap, now),
            reverse=True,
        )[:k]

    def mark_pulled(self, airports: list[str], now: float | None = None) -> None:
        """Record pull timestamp for each airport and persist to disk."""
        now = now or time.time()
        for ap in airports:
            if ap in self._state:
                self._state[ap] = now
        self._save()

    def report(self, now: float | None = None) -> None:
        """Print a sorted table of overdue scores, ages, and ideal intervals."""
        now = now or time.time()
        rows = []
        for ap, weight in self.weights.items():
            score = self.overdue_score(ap, now)
            age_min = (now - self._state.get(ap, 0.0)) / 60.0
            ideal_min = (self.base_interval / weight) / 60.0
            rows.append((ap, weight, score, age_min, ideal_min))
        rows.sort(key=lambda r: r[2], reverse=True)
        print(f"\n{'Airport':<8} {'Weight':>6} {'Score':>7} {'Age(min)':>10} {'Ideal(min)':>12}")
        print("-" * 50)
        for ap, w, sc, age, ideal in rows:
            age_str = f"{age:.0f}" if age < 1e6 else "never"
            print(f"{ap:<8} {w:>6.3f} {sc:>7.3f} {age_str:>10} {ideal:>12.1f}")

    def coverage_summary(self, now: float | None = None) -> dict:
        """Return {airport: overdue_score} for all airports."""
        now = now or time.time()
        return {ap: self.overdue_score(ap, now) for ap in self.weights}
