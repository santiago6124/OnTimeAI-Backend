"""Build the US airports universe CSV from BTS On-Time Performance files.

Scans `data_raw/bts/*.csv` (BTS PREZIP format), collects all unique
(Origin, OriginState) pairs, derives timezone + IEM ASOS network for each,
and writes `airports_universe.csv` at the repo root.

This file becomes the source of truth for `descarga_data_iem_full.py`,
`construir_dataset_maestro_full_us.py`, and the runtime `AIRPORTS` set in
`ontimeai/live.py`.

Usage:
    python3 build_airports_universe.py
    python3 build_airports_universe.py --bts-dir data_raw/bts
    python3 build_airports_universe.py --out airports_universe.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
BTS_DIR = PROJECT_ROOT / "data_raw" / "bts"


# US state → primary IANA timezone. Split-TZ states (FL, IN, KY, MI, NE, ND,
# OR, SD, TN, TX) use the timezone covering the majority of their airports.
# Within merge_asof's 90-min tolerance, edge cases (Indiana, El Paso, FL panhandle)
# stay safe.
STATE_TZ: dict[str, str] = {
    "AL": "America/Chicago",      "AK": "America/Anchorage",
    "AZ": "America/Phoenix",      "AR": "America/Chicago",
    "CA": "America/Los_Angeles",  "CO": "America/Denver",
    "CT": "America/New_York",     "DE": "America/New_York",
    "DC": "America/New_York",     "FL": "America/New_York",
    "GA": "America/New_York",     "HI": "Pacific/Honolulu",
    "ID": "America/Boise",        "IL": "America/Chicago",
    "IN": "America/Indiana/Indianapolis",
    "IA": "America/Chicago",      "KS": "America/Chicago",
    "KY": "America/New_York",     "LA": "America/Chicago",
    "ME": "America/New_York",     "MD": "America/New_York",
    "MA": "America/New_York",     "MI": "America/New_York",
    "MN": "America/Chicago",      "MS": "America/Chicago",
    "MO": "America/Chicago",      "MT": "America/Denver",
    "NE": "America/Chicago",      "NV": "America/Los_Angeles",
    "NH": "America/New_York",     "NJ": "America/New_York",
    "NM": "America/Denver",       "NY": "America/New_York",
    "NC": "America/New_York",     "ND": "America/Chicago",
    "OH": "America/New_York",     "OK": "America/Chicago",
    "OR": "America/Los_Angeles",  "PA": "America/New_York",
    "RI": "America/New_York",     "SC": "America/New_York",
    "SD": "America/Chicago",      "TN": "America/Chicago",
    "TX": "America/Chicago",      "UT": "America/Denver",
    "VT": "America/New_York",     "VA": "America/New_York",
    "WA": "America/Los_Angeles",  "WV": "America/New_York",
    "WI": "America/Chicago",      "WY": "America/Denver",
    # US territories
    "PR": "America/Puerto_Rico",  "VI": "America/St_Thomas",
    "GU": "Pacific/Guam",         "AS": "Pacific/Pago_Pago",
    "MP": "Pacific/Saipan",
}


def state_to_iem_network(state: str) -> str:
    """IEM ASOS network code, e.g. 'GA_ASOS', 'PR_ASOS'."""
    return f"{state.upper()}_ASOS"


# IEM catalog: GeoJSON FeatureCollection of all stations in a network.
# Each feature has properties.sid (3-letter station id, IATA-equivalent for major
# airports), properties.sname (long name), geometry.coordinates ([lon, lat]).
IEM_CATALOG_URLS = (
    "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson",
    "https://mesonet.agron.iastate.edu/geojson/network.php?network={network}",
)


def fetch_iem_network_catalog(network: str, timeout: int = 30) -> dict[str, dict]:
    """Returns dict keyed by station id (sid) with name + coordinates.

    Tries multiple URL patterns (IEM has shifted endpoints over the years).
    Returns empty dict if all of them fail.
    """
    last_err: Exception | None = None
    for url_tpl in IEM_CATALOG_URLS:
        url = url_tpl.format(network=network)
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            last_err = e
            continue

        out: dict[str, dict] = {}
        for feat in data.get("features", []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            sid = props.get("sid") or feat.get("id")
            if not sid:
                continue
            coords = geom.get("coordinates") or [None, None]
            out[str(sid).upper()] = {
                "iem_name": props.get("sname"),
                "iem_lon": coords[0],
                "iem_lat": coords[1],
            }
        if out:
            return out
        # endpoint responded but had no features — try next URL template

    if last_err:
        print(f"    [warn] IEM catalog {network}: all URL patterns failed ({last_err})")
    return {}


def validate_against_iem(df: pd.DataFrame, sleep_between: float = 1.0) -> pd.DataFrame:
    """Add iem_validated / iem_name / iem_lat / iem_lon columns.

    Hits the IEM catalog once per unique network (not per station), so the cost
    is ~52 API calls regardless of universe size.
    """
    networks = sorted(df["iem_network"].unique())
    print(f"\nValidating against IEM catalog ({len(networks)} networks)...")

    catalog: dict[tuple[str, str], dict] = {}
    for i, net in enumerate(networks, 1):
        stations = fetch_iem_network_catalog(net)
        print(f"  [{i}/{len(networks)}] {net} → {len(stations)} stations", flush=True)
        for sid, meta in stations.items():
            catalog[(net, sid)] = meta
        time.sleep(sleep_between)

    out = df.copy()
    keys = list(zip(out["iem_network"], out["iata"]))
    out["iem_validated"] = [k in catalog for k in keys]
    out["iem_name"] = [catalog.get(k, {}).get("iem_name") for k in keys]
    out["iem_lat"] = [catalog.get(k, {}).get("iem_lat") for k in keys]
    out["iem_lon"] = [catalog.get(k, {}).get("iem_lon") for k in keys]
    return out


def scan_bts_for_airports(bts_paths: list[Path]) -> pd.DataFrame:
    """Walk BTS CSVs once and return a DataFrame of unique (iata, state)."""
    seen: dict[str, str] = {}  # iata -> state
    for path in bts_paths:
        print(f"  scan <- {path.name}", flush=True)
        # Sample columns to figure out the schema variant
        try:
            head = pd.read_csv(path, nrows=0)
        except Exception as e:
            print(f"    [skip] cannot read header: {e}")
            continue
        cols = {c.strip(): c for c in head.columns}

        # PREZIP variant: Origin / OriginState. Legacy: ORIGIN / ORIGIN_STATE_ABR.
        if "Origin" in cols and "OriginState" in cols:
            origin_col, state_col = "Origin", "OriginState"
            dest_col, dest_state_col = "Dest", "DestState"
        elif "ORIGIN" in cols and "ORIGIN_STATE_ABR" in cols:
            origin_col, state_col = "ORIGIN", "ORIGIN_STATE_ABR"
            dest_col, dest_state_col = "DEST", "DEST_STATE_ABR"
        else:
            print(f"    [skip] no recognizable airport+state columns")
            continue

        for chunk in pd.read_csv(path, chunksize=200_000,
                                 usecols=[origin_col, state_col, dest_col, dest_state_col],
                                 low_memory=False):
            for iata_c, st_c in ((origin_col, state_col), (dest_col, dest_state_col)):
                sub = chunk[[iata_c, st_c]].dropna()
                if sub.empty:
                    continue
                sub[iata_c] = sub[iata_c].astype("string").str.strip().str.upper()
                sub[st_c] = sub[st_c].astype("string").str.strip().str.upper()
                pairs = sub.drop_duplicates()
                for iata, st in zip(pairs[iata_c], pairs[st_c], strict=False):
                    if not iata or not st:
                        continue
                    seen.setdefault(iata, st)

    rows = []
    for iata, state in sorted(seen.items()):
        tz = STATE_TZ.get(state)
        if not tz:
            print(f"  [warn] no timezone for state '{state}' (airport {iata}) — skipping")
            continue
        rows.append({
            "iata": iata,
            "state": state,
            "timezone": tz,
            "iem_network": state_to_iem_network(state),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bts-dir", type=Path, default=BTS_DIR,
                   help=f"Directory containing BTS monthly CSVs (default: {BTS_DIR})")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "airports_universe.csv",
                   help="Output CSV path")
    p.add_argument("--no-validate-iem", action="store_true",
                   help="Skip IEM catalog cross-check (faster but downloader will waste "
                        "calls on stations that don't exist in IEM)")
    args = p.parse_args(argv)

    if not args.bts_dir.is_dir():
        print(f"ERROR: BTS dir does not exist: {args.bts_dir}")
        print("Run `python3 descarga_bts.py 2022 2023 2024 2025` first.")
        return 1

    bts_paths = sorted(args.bts_dir.glob("*.csv"))
    if not bts_paths:
        print(f"ERROR: no CSVs in {args.bts_dir}")
        return 1

    print(f"Scanning {len(bts_paths)} BTS files...")
    df = scan_bts_for_airports(bts_paths)
    print(f"\nFound {len(df)} unique US airports across {df['state'].nunique()} states/territories.")
    print("Top 10 states by airport count:")
    print(df["state"].value_counts().head(10).to_string())

    if not args.no_validate_iem:
        df = validate_against_iem(df)
        n_validated = int(df["iem_validated"].sum())
        n_invalid = len(df) - n_validated
        print(f"\nIEM cross-check: {n_validated} validated, {n_invalid} not found in IEM catalog")
        if n_invalid > 0:
            invalid_sample = df.loc[~df["iem_validated"], ["iata", "state"]].head(15)
            print("Sample of airports NOT in IEM catalog (will be skipped by downloader):")
            print(invalid_sample.to_string(index=False))
    else:
        df["iem_validated"] = True  # caller opted out → trust everything
        df["iem_name"] = None
        df["iem_lat"] = None
        df["iem_lon"] = None

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print(f"\nPreview (validated only):")
    preview = df[df["iem_validated"]].head(15) if "iem_validated" in df.columns else df.head(15)
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
