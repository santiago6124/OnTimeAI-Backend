from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BASE_DIR = Path("/Users/mateopappalardo/Downloads/tesis_local")
WEATHER_PATH = BASE_DIR / "clima_iem_asos_2025_utc.csv"
OUT_PATH = BASE_DIR / "dataset_maestro_ATL_2025_BTS_IEM_ORIG_DEST.csv"

# Solo aeropuertos con meteo descargada por ahora.
AIRPORTS = {
    "ATL",
    "LGA",
    "MCO",
    "FLL",
    "MIA",
    "DFW",
    "DCA",
    "EWR",
    "TPA",
    "ORD",
    "PHL",
    "DEN",
    "LAX",
    "BWI",
    "LAS",
    "BOS",
}

TZ_BY_AIRPORT = {
    "ATL": "America/New_York",
    "LGA": "America/New_York",
    "MCO": "America/New_York",
    "FLL": "America/New_York",
    "MIA": "America/New_York",
    "DFW": "America/Chicago",
    "DCA": "America/New_York",
    "EWR": "America/New_York",
    "TPA": "America/New_York",
    "ORD": "America/Chicago",
    "PHL": "America/New_York",
    "DEN": "America/Denver",
    "LAX": "America/Los_Angeles",
    "BWI": "America/New_York",
    "LAS": "America/Los_Angeles",
    "BOS": "America/New_York",
}

WX_NUM_COLS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]
MONTH_PREFIXES = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    hours = np.floor(values / 100)
    minutes = values % 100
    invalid = (values < 0) | (hours > 23) | (minutes >= 60)
    out = hours * 60 + minutes
    out = out.astype("float64")
    out[invalid] = np.nan
    return out


def local_series_to_utc_naive(local_naive: pd.Series, tz_name: str) -> pd.Series:
    tz = ZoneInfo(tz_name)

    def _convert(dt):
        if pd.isna(dt):
            return pd.NaT
        try:
            return dt.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        except Exception:
            return pd.NaT

    return local_naive.apply(_convert)


def add_utc_event_times(flights: pd.DataFrame) -> pd.DataFrame:
    out = flights.copy()
    out["CRS_DEP_MIN"] = hhmm_to_minutes(out["CRS_DEP_TIME"])
    out["DEP_LOCAL_DT"] = out["FL_DATE"] + pd.to_timedelta(out["CRS_DEP_MIN"], unit="m")
    out["EVENT_ORIGIN_UTC"] = pd.NaT

    for airport, tz_name in TZ_BY_AIRPORT.items():
        mask = out["ORIGIN"].eq(airport)
        if mask.any():
            out.loc[mask, "EVENT_ORIGIN_UTC"] = local_series_to_utc_naive(out.loc[mask, "DEP_LOCAL_DT"], tz_name)

    out["EVENT_ORIGIN_UTC"] = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce")
    elapsed = pd.to_numeric(out["CRS_ELAPSED_TIME"], errors="coerce")
    out["EVENT_DEST_UTC"] = out["EVENT_ORIGIN_UTC"] + pd.to_timedelta(elapsed, unit="m")
    return out


def load_weather() -> pd.DataFrame:
    wx = pd.read_csv(WEATHER_PATH, low_memory=False)
    wx.columns = [c.strip().lower() for c in wx.columns]
    wx["station"] = wx["station"].astype("string").str.strip().str.upper()
    wx["valid"] = pd.to_datetime(wx["valid"], errors="coerce", utc=True).dt.tz_convert(None)

    for col in WX_NUM_COLS:
        wx[col] = pd.to_numeric(wx[col].replace({"M": np.nan, "T": 0.001}), errors="coerce")

    wx["wx_precip_flag"] = wx["p01m"].fillna(0) > 0
    wx["wx_low_vis_flag"] = wx["vsby"] < 3
    wx["wx_strong_wind_flag"] = (wx["sknt"] >= 20) | (wx["gust"] >= 30)

    cols = [
        "station",
        "valid",
        "tmpc",
        "dwpc",
        "relh",
        "drct",
        "sknt",
        "alti",
        "p01m",
        "vsby",
        "gust",
        "wxcodes",
        "wx_precip_flag",
        "wx_low_vis_flag",
        "wx_strong_wind_flag",
    ]
    return wx[cols].sort_values(["station", "valid"]).reset_index(drop=True)


def standardize_bts_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    if "OP_UNIQUE_CARRIER" in df.columns and "OP_CARRIER" not in df.columns:
        df["OP_CARRIER"] = df["OP_UNIQUE_CARRIER"]

    if "DEP_DELAY" not in df.columns and "DEP_DELAY_NEW" in df.columns:
        df["DEP_DELAY"] = df["DEP_DELAY_NEW"]

    if "ARR_DELAY" not in df.columns and "ARR_DELAY_NEW" in df.columns:
        df["ARR_DELAY"] = df["ARR_DELAY_NEW"]

    df["ORIGIN"] = df["ORIGIN"].astype("string").str.strip().str.upper()
    df["DEST"] = df["DEST"].astype("string").str.strip().str.upper()

    date_fmt = "%m/%d/%Y %I:%M:%S %p"
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format=date_fmt, errors="coerce")

    num_cols = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "OP_CARRIER_FL_NUM",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "DEP_TIME",
        "ARR_TIME",
        "DEP_DELAY",
        "ARR_DELAY",
        "DEP_DELAY_NEW",
        "ARR_DELAY_NEW",
        "CANCELLED",
        "DIVERTED",
        "CRS_ELAPSED_TIME",
        "ACTUAL_ELAPSED_TIME",
        "AIR_TIME",
        "DISTANCE",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "YEAR",
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "FL_DATE",
        "OP_CARRIER",
        "TAIL_NUM",
        "OP_CARRIER_FL_NUM",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "DEP_TIME",
        "ARR_TIME",
        "DEP_DELAY",
        "ARR_DELAY",
        "DEP_DELAY_NEW",
        "ARR_DELAY_NEW",
        "CANCELLED",
        "DIVERTED",
        "CRS_ELAPSED_TIME",
        "ACTUAL_ELAPSED_TIME",
        "AIR_TIME",
        "DISTANCE",
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ]

    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols]


def load_bts_atl_flights() -> pd.DataFrame:
    bts_files = []
    for path in sorted(BASE_DIR.glob("*.csv")):
        name = path.name.upper()
        if not any(name.startswith(f"{m}_2025_") for m in MONTH_PREFIXES):
            continue
        if "IEM" in name or "CLIMA" in name or "DATASET_MAESTRO" in name:
            continue

        # Verificamos estructura mínima BTS para evitar colar otros CSV.
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        cols_upper = {c.strip().upper() for c in cols}
        required = {"FL_DATE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ELAPSED_TIME"}
        if required.issubset(cols_upper):
            bts_files.append(path)

    if not bts_files:
        raise FileNotFoundError("No se encontraron archivos mensuales BTS 2025 compatibles")

    print("Archivos BTS detectados:", [f.name for f in bts_files])
    frames = []
    for path in bts_files:
        print(f"Procesando {path.name} ...")
        for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
            std = standardize_bts_chunk(chunk)
            mask_atl = std["ORIGIN"].eq("ATL") | std["DEST"].eq("ATL")
            mask_known = std["ORIGIN"].isin(AIRPORTS) & std["DEST"].isin(AIRPORTS)
            sub = std.loc[mask_atl & mask_known].copy()
            if not sub.empty:
                frames.append(sub)

    if not frames:
        raise ValueError("No quedaron vuelos luego de aplicar filtros ATL + aeropuertos con meteo")

    flights = pd.concat(frames, ignore_index=True)
    flights = flights.drop_duplicates().reset_index(drop=True)
    flights["FLOW_ATL"] = np.where(flights["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    flights["PAR_AIRPORT"] = np.where(flights["ORIGIN"].eq("ATL"), flights["DEST"], flights["ORIGIN"])
    return flights


def merge_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    def asof_by_station(
        left: pd.DataFrame,
        wx_df: pd.DataFrame,
        station_col: str,
        event_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        out_parts = []
        for station in sorted(left[station_col].dropna().astype(str).unique()):
            lsub = left[left[station_col].eq(station)].copy()
            wsub = wx_df[wx_df["station"].eq(station)].copy()
            if lsub.empty:
                continue

            lsub = lsub.sort_values(event_col)
            wsub = wsub.sort_values("valid")

            merged = pd.merge_asof(
                lsub,
                wsub,
                left_on=event_col,
                right_on="valid",
                direction="nearest",
                tolerance=pd.Timedelta("90min"),
            )
            out_parts.append(merged)

        if not out_parts:
            raise ValueError(f"No hubo datos para merge_asof en {prefix}")

        merged_all = pd.concat(out_parts, ignore_index=True)
        gap = (merged_all[event_col] - merged_all["valid"]).abs().dt.total_seconds() / 60
        merged_all = merged_all.rename(
            columns={
                "valid": f"{prefix}_WX_VALID_UTC",
                "tmpc": f"{prefix}_WX_TMPC",
                "dwpc": f"{prefix}_WX_DWPC",
                "relh": f"{prefix}_WX_RELH",
                "drct": f"{prefix}_WX_DRCT",
                "sknt": f"{prefix}_WX_SKNT",
                "alti": f"{prefix}_WX_ALTI",
                "p01m": f"{prefix}_WX_P01M",
                "vsby": f"{prefix}_WX_VSBY",
                "gust": f"{prefix}_WX_GUST",
                "wxcodes": f"{prefix}_WX_CODES",
                "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
                "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
                "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
            }
        )
        merged_all[f"{prefix}_WX_MATCH_GAP_MIN"] = gap
        return merged_all.drop(columns=["station"])

    origin_merge = asof_by_station(
        left=flights,
        wx_df=wx,
        station_col="ORIGIN",
        event_col="EVENT_ORIGIN_UTC",
        prefix="ORIG",
    )

    dest_merge = asof_by_station(
        left=origin_merge,
        wx_df=wx,
        station_col="DEST",
        event_col="EVENT_DEST_UTC",
        prefix="DEST",
    )

    return dest_merge


def main() -> None:
    wx = load_weather()
    flights = load_bts_atl_flights()
    flights = add_utc_event_times(flights)

    master = merge_weather(flights, wx)

    master.to_csv(OUT_PATH, index=False)

    print("\nDataset maestro creado:")
    print("- Ruta:", OUT_PATH)
    print("- Filas:", len(master))
    print("- Columnas:", len(master.columns))
    print("- Cobertura meteo ORIG (%):", round(master["ORIG_WX_VALID_UTC"].notna().mean() * 100, 2))
    print("- Cobertura meteo DEST (%):", round(master["DEST_WX_VALID_UTC"].notna().mean() * 100, 2))


if __name__ == "__main__":
    main()
