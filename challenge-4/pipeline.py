"""
pipeline.py
Turns raw NOAA/BOEM Marine Cadastre AIS data into port arrival/departure
events and KPIs for the Seaport Operations Dashboard (Challenge 4).

Usage:
    from pipeline import PORT_NAME, load_ais, load_port_info, compute_events, compute_kpis

    ais = load_ais("data/AIS_June_Daily_2025")   # folder of daily .csv.zst files
    events = compute_events(ais)
    kpis = compute_kpis(events, ais)
"""

import glob
import io
import os

import numpy as np
import pandas as pd
import zstandard as zstd

# ---------------------------------------------------------------------------
# Config: Port of Long Beach harbor entrance + detection radius
# ---------------------------------------------------------------------------
PORT_NAME = "Port of Long Beach"
PORT_LAT = 33.75
PORT_LON = -118.19
RADIUS_NM = 2.5          # nautical miles from port center = "in port area"
SOG_MIN_KNOTS = 2.0      # minimum speed to count a radius crossing as real transit

# Long Beach / San Pedro Bay bounding box — used to filter each chunk of
# each daily file down BEFORE it's kept in memory.
LAT_MIN, LAT_MAX = 33.6, 33.8
LON_MIN, LON_MAX = -118.3, -118.1

# How many raw rows to read into memory at a time per file. Lower this if
# you're still hitting memory limits; raise it if loading feels too slow
# once memory isn't the bottleneck anymore.
CHUNKSIZE = 200_000

# NOAA AIS VesselType codes are numeric; this maps common ranges to a
# human-readable class. Extend as needed based on your data_dictionary.pdf.
VESSEL_CLASS_MAP = {
    range(70, 80): "Cargo",
    range(80, 90): "Tanker",
    range(30, 39): "Fishing",
}

# This extract uses lowercase snake_case headers, not NOAA's older
# BaseDateTime/LAT/LON/VesselName style. We select these raw columns,
# then rename to canonical names immediately after load so the rest of
# the pipeline (which uses MMSI, BaseDateTime, LAT, LON, etc.) is unaffected.
AIS_COLUMNS = {
    "mmsi", "base_date_time", "latitude", "longitude", "sog", "cog",
    "vessel_name", "vessel_type", "status", "length", "width",
}

COLUMN_RENAME = {
    "mmsi": "MMSI",
    "base_date_time": "BaseDateTime",
    "latitude": "LAT",
    "longitude": "LON",
    "sog": "SOG",
    "cog": "COG",
    "vessel_name": "VesselName",
    "vessel_type": "VesselType",
    "status": "Status",
    "length": "Length",
    "width": "Width",
}


def classify_vessel(vessel_type: float) -> str:
    """Map a numeric AIS VesselType code to a readable class label."""
    if pd.isna(vessel_type):
        return "Other"
    vt = int(vessel_type)
    for rng, label in VESSEL_CLASS_MAP.items():
        if vt in rng:
            return label
    return "Other"


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in nautical miles."""
    R_NM = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_NM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Loading: folder of daily .csv.zst AIS files (chunked to control memory)
# ---------------------------------------------------------------------------
def _read_zst_csv(path: str, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """
    Decompress a single .csv.zst file and read it in chunks, filtering
    each chunk to the port bounding box immediately and discarding
    everything outside it. This keeps peak memory to roughly one
    chunk's worth of data, regardless of the full file size — important
    on memory-constrained environments like small Codespaces containers.
    """
    kept_chunks = []
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            reader_iter = pd.read_csv(
                text_stream,
                usecols=lambda c: c in AIS_COLUMNS,
                chunksize=chunksize,
            )
            for chunk in reader_iter:
                chunk = chunk.rename(columns=COLUMN_RENAME)
                chunk = chunk[
                    chunk["LAT"].between(LAT_MIN, LAT_MAX)
                    & chunk["LON"].between(LON_MIN, LON_MAX)
                ]
                if not chunk.empty:
                    for col in ["LAT", "LON", "SOG", "COG"]:
                        if col in chunk.columns:
                            chunk[col] = pd.to_numeric(chunk[col], downcast="float")
                    if "MMSI" in chunk.columns:
                        chunk["MMSI"] = pd.to_numeric(chunk["MMSI"], downcast="integer")
                    if "VesselType" in chunk.columns:
                        chunk["VesselType"] = pd.to_numeric(chunk["VesselType"], downcast="integer")
                    kept_chunks.append(chunk.copy())

    if not kept_chunks:
        return pd.DataFrame(columns=list(COLUMN_RENAME.values()))

    return pd.concat(kept_chunks, ignore_index=True)


def _load_ais_folder(folder_path: str, pattern: str = "*.csv.zst") -> pd.DataFrame:
    """
    Read every daily AIS file in folder_path matching pattern, using the
    chunked/filtered reader above, then concatenate the (already slimmed)
    per-file results.
    """
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder_path}")

    frames = []
    for f in files:
        try:
            frames.append(_read_zst_csv(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")  # swap for logging later

    if not frames:
        raise ValueError(f"None of the files in {folder_path} could be read.")

    return pd.concat(frames, ignore_index=True)


def load_ais(path: str) -> pd.DataFrame:
    """
    Load AIS data for the dashboard.

    `path` is a FOLDER containing one or more daily .csv.zst Marine Cadastre
    extracts (not a single CSV) — set DATA_PATH in app.py to that folder.
    """
    df = _load_ais_folder(path)

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    df = df.sort_values(["MMSI", "BaseDateTime"]).drop_duplicates().reset_index(drop=True)

    df["dist_nm"] = haversine_nm(df["LAT"], df["LON"], PORT_LAT, PORT_LON)
    df["in_area"] = df["dist_nm"] <= RADIUS_NM
    df["vessel_class"] = df.get("VesselType", pd.Series(dtype=float)).apply(classify_vessel)

    # Fill missing SOG rather than silently dropping real crossings later —
    # a NaN here previously evaluated as "too slow to count" by accident.
    if "SOG" in df.columns:
        df["SOG"] = df["SOG"].fillna(0.0)
    else:
        df["SOG"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Loading: NGA World Port Index (Pub 150)
# ---------------------------------------------------------------------------
def load_port_info(path: str = "data/world_ports_pub150.csv") -> pd.DataFrame:
    """
    Load the NGA World Port Index (Pub 150) export and rename its
    Excel-truncated headers to readable names (per the WPI field guide).
    """
    df = pd.read_csv(path)

    rename_map = {
        "wpinumber": "wpi_number",
        "regionname": "region_name",
        "main_port_": "main_port_name",
        "alternate_": "alternate_port_name",
        "unlocode": "un_locode",
        "countryCode": "country_code",
        "harbor_siz": "harbor_size",
        "harbor_typ": "harbor_type",
        "channel_de": "channel_depth_m",
        "anchorage_": "anchorage_depth_m",
        "cargo_pier": "cargo_pier_depth_m",
        "maxvessell": "max_vessel_length_m",
        "maxvesselb": "max_vessel_beam_m",
        "maxvesseld": "max_vessel_draft_m",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Event detection + KPIs (vectorized — no per-row Python loops)
# ---------------------------------------------------------------------------
def compute_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect arrival/departure events per vessel: a crossing of the port
    radius boundary while moving faster than SOG_MIN_KNOTS.
    """
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    # Previous in_area value per vessel (NaN for each vessel's first row)
    df["prev_in_area"] = df.groupby("MMSI")["in_area"].shift(1)

    # A "crossing" is any row where in_area differs from the previous row
    # for the same vessel, and it's not the vessel's very first ping.
    crossed = df["prev_in_area"].notna() & (df["in_area"] != df["prev_in_area"])

    # Apply the speed threshold, treating NaN SOG as not qualifying.
    fast_enough = df["SOG"].notna() & (df["SOG"] >= SOG_MIN_KNOTS)

    mask = crossed & fast_enough
    events_df = df.loc[
        mask, ["MMSI", "VesselName", "vessel_class", "in_area", "BaseDateTime", "LAT", "LON"]
    ].copy()

    if events_df.empty:
        return pd.DataFrame(
            columns=["MMSI", "VesselName", "vessel_class", "event", "timestamp", "lat", "lon"]
        )

    events_df["event"] = np.where(events_df["in_area"], "arrival", "departure")
    events_df = events_df.rename(columns={"BaseDateTime": "timestamp", "LAT": "lat", "LON": "lon"})
    events_df = events_df.drop(columns=["in_area"])

    return events_df.sort_values("timestamp").reset_index(drop=True)


def compute_dwell_times(events_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each arrival with the next departure per vessel to get dwell time (hrs)."""
    if events_df.empty:
        return pd.DataFrame(columns=["MMSI", "dwell_hours"])

    df = events_df.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

    # Next event's type/timestamp within the same vessel group
    df["next_event"] = df.groupby("MMSI")["event"].shift(-1)
    df["next_timestamp"] = df.groupby("MMSI")["timestamp"].shift(-1)

    pairs = df[(df["event"] == "arrival") & (df["next_event"] == "departure")].copy()
    if pairs.empty:
        return pd.DataFrame(columns=["MMSI", "dwell_hours"])

    pairs["dwell_hours"] = (pairs["next_timestamp"] - pairs["timestamp"]).dt.total_seconds() / 3600

    return pairs[["MMSI", "dwell_hours"]].reset_index(drop=True)


def compute_kpis(events_df: pd.DataFrame, ais_df: pd.DataFrame) -> dict:
    """Compute the four MVP KPIs for the dashboard."""
    dwell_df = compute_dwell_times(events_df)

    # "in area now" = each vessel's MOST RECENT ping, not any historical ping
    latest_per_vessel = ais_df.sort_values("BaseDateTime").groupby("MMSI").tail(1)
    vessels_in_area_now = int(latest_per_vessel["in_area"].sum())

    arrivals = int((events_df["event"] == "arrival").sum()) if not events_df.empty else 0
    departures = int((events_df["event"] == "departure").sum()) if not events_df.empty else 0

    return {
        "vessels_in_area": vessels_in_area_now,
        "arrivals": arrivals,
        "departures": departures,
        "avg_dwell_hours": round(dwell_df["dwell_hours"].mean(), 1) if not dwell_df.empty else 0.0,
    }

def compute_hourly_activity(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate arrival/departure counts by hour-of-day, collapsed across
    all days in the filtered range. Useful for spotting intraday patterns
    (e.g. morning arrival rush, evening departure peak).
    """
    if events_df.empty:
        return pd.DataFrame(columns=["hour", "event", "count"])

    df = events_df.copy()
    df["hour"] = df["timestamp"].dt.hour

    hourly = (
        df.groupby(["hour", "event"])
        .size()
        .reset_index(name="count")
    )

    # Ensure all 24 hours appear even if some have zero events, so the
    # x-axis doesn't skip hours with no activity.
    full_hours = pd.DataFrame({"hour": range(24)})
    events_types = df["event"].unique()
    grid = pd.MultiIndex.from_product([range(24), events_types], names=["hour", "event"]).to_frame(index=False)
    hourly = grid.merge(hourly, on=["hour", "event"], how="left").fillna({"count": 0})
    hourly["count"] = hourly["count"].astype(int)

    return hourly 

def compute_congestion(events_df: pd.DataFrame, resample: str = "H", threshold: int = 5) -> pd.DataFrame:
    """
    Reconstruct vessel occupancy over time from arrival/departure events
    (running count: +1 per arrival, -1 per departure), resampled to a
    regular interval. Flags periods where occupancy exceeds `threshold`
    as congested.
    """
    if events_df.empty:
        return pd.DataFrame(columns=["timestamp", "occupancy", "congested"])

    df = events_df.sort_values("timestamp").copy()
    df["delta"] = np.where(df["event"] == "arrival", 1, -1)

    ts = df.set_index("timestamp")["delta"].resample(resample).sum().fillna(0)
    occupancy = ts.cumsum()
    occupancy = occupancy.clip(lower=0)  # guard against negative counts from partial data

    result = occupancy.reset_index()
    result.columns = ["timestamp", "occupancy"]
    result["congested"] = result["occupancy"] >= threshold

    return result


def detect_dwell_anomalies(events_df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Flag vessels whose dwell time is unusually long relative to the rest
    of the fleet (z-score based). False positives matter more than
    catching every possible anomaly, so this uses a conservative threshold.
    """
    dwell_df = compute_dwell_times(events_df)
    if dwell_df.empty or len(dwell_df) < 2:
        return pd.DataFrame(columns=["MMSI", "dwell_hours", "z_score"])

    mean_dwell = dwell_df["dwell_hours"].mean()
    std_dwell = dwell_df["dwell_hours"].std()

    if std_dwell == 0 or pd.isna(std_dwell):
        return pd.DataFrame(columns=["MMSI", "dwell_hours", "z_score"])

    dwell_df = dwell_df.copy()
    dwell_df["z_score"] = (dwell_df["dwell_hours"] - mean_dwell) / std_dwell

    anomalies = dwell_df[dwell_df["z_score"] >= z_thresh].sort_values("z_score", ascending=False)
    return anomalies.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Situation report
# ---------------------------------------------------------------------------
def generate_situation_report(kpis: dict, port_name: str) -> str:
    """Rule-based situation report generated from the computed KPIs."""
    lines = [f"**{port_name} — Situation Summary**", ""]
    lines.append(f"- {kpis['vessels_in_area']} vessels currently in the monitored area.")
    lines.append(f"- {kpis['arrivals']} arrivals and {kpis['departures']} departures in the selected period.")
    if kpis["avg_dwell_hours"] > 0:
        lines.append(f"- Average dwell time is {kpis['avg_dwell_hours']} hours.")
    else:
        lines.append("- No completed dwell periods (arrival+departure pairs) in this window yet.")
    if kpis["arrivals"] > kpis["departures"]:
        lines.append("- More arrivals than departures — port congestion may be building.")
    elif kpis["departures"] > kpis["arrivals"]:
        lines.append("- More departures than arrivals — traffic is clearing.")
    return "\n".join(lines)

