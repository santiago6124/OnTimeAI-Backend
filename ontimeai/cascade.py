"""Cascade effect (delay propagation) — TODO for post-MVP iterations.

Roadmap for this module once the MVP individual-flight classifier is stable:

1. Tail-number lineage features (prev_arr_delay_tail, prev_turnaround_tail,
   tail_flights_today), computed strictly from data available before the target
   flight's CRS_DEP_TIME to preserve the leakage discipline of the MVP pipeline.
2. Sequence model (LSTM / Transformer) over same-day rotations per tail_num
   to model intra-aircraft propagation (Qu, Wu & Zhang 2023).
3. Optional Graph Neural Network over the flight network for inter-airport
   cascade (Zhang et al. 2026 Edge-GNN; Cai et al. 2024 CausalNet;
   Zeng et al. 2021 Graph-LSTM).
4. API endpoint returning the downstream cascade projection for an input flight.
"""
from __future__ import annotations


def predict_cascade(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "Cascade propagation model not implemented in the MVP. "
        "See ontimeai/cascade.py docstring for the implementation roadmap."
    )
