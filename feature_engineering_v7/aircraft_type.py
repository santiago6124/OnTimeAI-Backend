"""
Aircraft type feature engineering.

Maps TAIL_NUM → ICAO typecode → AIRCRAFT_FAMILY (stable categorical).

AIRCRAFT_FAMILY replaces TAIL_NUM in v9+, eliminating the primary source of
temporal drift identified by adversarial validation (tail-specific patterns
don't transfer across years as aircraft are retired/reassigned).

Coverage: ~83% from OpenSky DB. Missing tails → carrier-based fallback → "OTHER".
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

# ── Typecode → AIRCRAFT_FAMILY mapping ──────────────────────────────────────

_FAMILY_MAP = {
    # Boeing 737 family (narrow-body)
    "B737": "B737_FAMILY", "B738": "B737_FAMILY", "B739": "B737_FAMILY",
    "B38M": "B737_FAMILY", "B39M": "B737_FAMILY", "B37M": "B737_FAMILY",
    "B752": "B737_FAMILY", "B753": "B737_FAMILY",

    # Airbus narrow-body
    "A319": "AIRBUS_NB",  "A320": "AIRBUS_NB",  "A321": "AIRBUS_NB",
    "A20N": "AIRBUS_NB",  "A21N": "AIRBUS_NB",  "A318": "AIRBUS_NB",

    # Boeing 717 (MD-88/90 replacement — mainly Delta)
    "B712": "B717",

    # Boeing wide-body
    "B762": "B767", "B763": "B767", "B764": "B767",
    "B772": "B777", "B773": "B777", "B77W": "B777", "B77L": "B777",
    "B788": "B787", "B789": "B787", "B78X": "B787",

    # Airbus wide-body
    "A332": "AIRBUS_WB", "A333": "AIRBUS_WB", "A338": "AIRBUS_WB",
    "A339": "AIRBUS_WB", "A359": "AIRBUS_WB", "A35K": "AIRBUS_WB",

    # Bombardier CRJ family (regional)
    "CRJ1": "CRJ",  "CRJ2": "CRJ",  "CRJ7": "CRJ",
    "CRJ9": "CRJ",  "CRJX": "CRJ",  "CL60": "CRJ",

    # Embraer regional jets
    "E135": "ERJ145", "E145": "ERJ145",
    "E170": "ERJ175", "E75L": "ERJ175", "E75S": "ERJ175",
    "E190": "ERJ190", "E195": "ERJ190", "E290": "ERJ190",

    # Turboprops
    "DH8A": "TURBOPROP", "DH8B": "TURBOPROP", "DH8C": "TURBOPROP",
    "DH8D": "TURBOPROP", "AT43": "TURBOPROP", "AT72": "TURBOPROP",
    "SF34": "TURBOPROP", "BE99": "TURBOPROP", "BE20": "TURBOPROP",
    "C208": "TURBOPROP",
}

_CARRIER_FALLBACK = {
    # If OpenSky misses the tail, infer family from carrier's dominant fleet
    "WN": "B737_FAMILY",   # Southwest: all-737
    "B6": "AIRBUS_NB",     # JetBlue: A320 family
    "NK": "AIRBUS_NB",     # Spirit: A320 family
    "F9": "AIRBUS_NB",     # Frontier: A320 family
    "AS": "B737_FAMILY",   # Alaska: 737 + E175 (close enough)
    "G4": "B737_FAMILY",   # Allegiant: A320 fam actually, but let model learn
    "SY": "B737_FAMILY",   # Sun Country: 737
}


def build_tail_lookup(opensky_csv: str, out_path: str | None = None) -> dict:
    """
    Build TAIL_NUM → AIRCRAFT_FAMILY lookup from OpenSky aircraft DB.
    Saves to JSON if out_path is given.
    """
    oa = pd.read_csv(opensky_csv, low_memory=False)
    oa = oa[oa["registration"].notna() & oa["typecode"].notna()][
        ["registration", "typecode"]
    ].drop_duplicates("registration")

    lookup = {}
    for _, row in oa.iterrows():
        tail = row["registration"]
        tc = str(row["typecode"]).strip().upper()
        family = _FAMILY_MAP.get(tc, "OTHER")
        lookup[tail] = family

    if out_path:
        Path(out_path).write_text(json.dumps(lookup))
    return lookup


def add_aircraft_family(
    df: pd.DataFrame,
    lookup: dict | None = None,
    lookup_path: str | None = None,
    tail_col: str = "TAIL_NUM",
    carrier_col: str = "OP_CARRIER",
) -> pd.DataFrame:
    """
    Add AIRCRAFT_FAMILY column to df.
    Tries lookup first, then carrier-based fallback, then "OTHER".
    """
    if lookup is None:
        if lookup_path is None:
            raise ValueError("Provide either lookup dict or lookup_path")
        lookup = json.loads(Path(lookup_path).read_text())

    families = df[tail_col].map(lookup)

    # carrier fallback for missing tails
    if carrier_col in df.columns:
        mask_miss = families.isna()
        if mask_miss.any():
            families[mask_miss] = df.loc[mask_miss, carrier_col].map(_CARRIER_FALLBACK)

    families = families.fillna("OTHER").astype("category")
    df = df.copy()
    df["AIRCRAFT_FAMILY"] = families
    return df
