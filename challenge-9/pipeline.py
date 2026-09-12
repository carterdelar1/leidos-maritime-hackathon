"""
pipeline.py
Detects potential vessel-to-vessel collision risks from AIS data using
CPA (Closest Point of Approach) and TCPA (Time to Closest Point of
Approach) — Challenge 9: Ocean Vessel Collision Risk Predictor.

Usage:
    from pipeline import load_ais, detect_collision_risks, collapse_encounters

    ais = load_ais("data/AIS_June_Daily_2025")
    risks = detect_collision_risks(ais)
    risks = collapse_encounters(risks)
"""

import glob
import io
import os

import numpy as np
import pandas as pd
import zstandard as zstd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 33.6, 33.8
LON_MIN, LON_MAX = -118.3, -118.1

CHUNKSIZE = 200_000

# Collision-risk thresholds — placeholders, tune against real data.
CPA_THRESHOLD_NM = 1.0     # flag pairs projected to pass within this distance
TCPA_MAX_HR = 0.5          # only flag near-future risk (next 30 min), not past
SNAPSHOT_FLOOR = "min"     # bucket AIS pings to the nearest minute for comparison

# Minimum relative speed (knots) required before two vessels are even
# considered "converging." Below this, they're either both stationary or
# moving in near-identical fashion (e.g. moored side by side) and should
# NOT be flagged as a collision risk, even if they're currently close.
MIN_REL_SPEED_KNOTS = 1.0

# When collapsing repeated per-minute flags into single encounter episodes,
# two flags for the same vessel pair within this many minutes of each other
# count as the same ongoing episode rather than separate encounters.
EPISODE_GAP_MINUTES = 5

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


# ---------------------------------------------------------------------------
# Loading (same pattern as Challenge 4)
# ---------------------------------------------------------------------------
def _read_zst_csv(path: str, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Decompress a .csv.zst file in chunks, filter to bbox, downcast dtypes."""
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
                    kept_chunks.append(chunk.copy())

    if not kept_chunks:
        return pd.DataFrame(columns=list(COLUMN_RENAME.values()))

    return pd.concat(kept_chunks, ignore_index=True)


def _load_ais_folder(folder_path: str, pattern: str = "*.csv.zst") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder_path}")

    frames = []
    for f in files:
        try:
            frames.append(_read_zst_csv(f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not frames:
        raise ValueError(f"None of the files in {folder_path} could be read.")

    return pd.concat(frames, ignore_index=True)


def load_ais(path: str) -> pd.DataFrame:
    """Load AIS data: folder of daily .csv.zst files -> cleaned DataFrame."""
    df = _load_ais_folder(path)

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    df = df.sort_values(["MMSI", "BaseDateTime"]).drop_duplicates().reset_index(drop=True)

    # SOG/COG are essential for trajectory projection here — drop rows
    # missing either rather than silently treating them as stationary.
    df = df.dropna(subset=["SOG", "COG"])

    return df


# ---------------------------------------------------------------------------
# CPA / TCPA math
# ---------------------------------------------------------------------------
def latlon_to_xy_nm(lat, lon, ref_lat, ref_lon):
    """
    Convert lat/lon to a local flat-earth (x, y) coordinate system in
    nautical miles, centered on a reference point. Valid at the small
    geographic scale of a single port approach area.
    """
    x = (lon - ref_lon) * 60 * np.cos(np.radians(ref_lat))  # nm, east-positive
    y = (lat - ref_lat) * 60                                  # nm, north-positive
    return x, y


def velocity_components(sog_knots, cog_deg):
    """Convert speed-over-ground + course-over-ground into (vx, vy) in nm/hr."""
    heading_rad = np.radians(cog_deg)
    vx = sog_knots * np.sin(heading_rad)  # east component
    vy = sog_knots * np.cos(heading_rad)  # north component
    return vx, vy


def compute_cpa_tcpa(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
    """
    Given two vessels' positions (nm) and velocities (nm/hr), return
    (cpa_nm, tcpa_hr): closest approach distance and time until it occurs.

    If the two vessels' relative speed is below MIN_REL_SPEED_KNOTS (e.g.
    both stationary, or moving in near-identical fashion — like two ships
    moored side by side), this returns (inf, inf) so the pair is never
    flagged as a "collision risk" just for being near each other while
    not actually converging.
    """
    dx, dy = x2 - x1, y2 - y1
    dvx, dvy = vx2 - vx1, vy2 - vy1

    rel_speed = np.sqrt(dvx**2 + dvy**2)
    if rel_speed < MIN_REL_SPEED_KNOTS:
        return np.inf, np.inf

    rel_speed_sq = rel_speed**2
    tcpa = -(dx * dvx + dy * dvy) / rel_speed_sq

    cpa_x = dx + dvx * tcpa
    cpa_y = dy + dvy * tcpa
    cpa = np.sqrt(cpa_x**2 + cpa_y**2)

    return cpa, tcpa


def detect_collision_risks(
    ais_df: pd.DataFrame,
    cpa_threshold_nm: float = CPA_THRESHOLD_NM,
    tcpa_max_hr: float = TCPA_MAX_HR,
) -> pd.DataFrame:
    """
    For each timestamp snapshot, take each vessel's state and check every
    pair present at that moment for a risky encounter: CPA below
    cpa_threshold_nm, with TCPA between 0 and tcpa_max_hr, AND relative
    speed above MIN_REL_SPEED_KNOTS (enforced inside compute_cpa_tcpa).

    NOTE: this is O(n^2) per snapshot and will flag the SAME real-world
    encounter once per minute it persists. Call collapse_encounters() on
    the result to reduce repeated per-minute flags into single episodes.
    """
    if ais_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "MMSI_1", "MMSI_2", "VesselName_1", "VesselName_2",
            "cpa_nm", "tcpa_min",
        ])

    ref_lat = ais_df["LAT"].mean()
    ref_lon = ais_df["LON"].mean()

    df = ais_df.copy()
    df["x_nm"], df["y_nm"] = latlon_to_xy_nm(df["LAT"], df["LON"], ref_lat, ref_lon)
    df["vx"], df["vy"] = velocity_components(df["SOG"], df["COG"])
    df["snapshot"] = df["BaseDateTime"].dt.floor(SNAPSHOT_FLOOR)

    results = []
    for snapshot, group in df.groupby("snapshot"):
        vessels = group.drop_duplicates(subset="MMSI")
        n = len(vessels)
        if n < 2:
            continue

        records = vessels.to_dict("records")
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = records[i], records[j]
                cpa, tcpa = compute_cpa_tcpa(
                    v1["x_nm"], v1["y_nm"], v1["vx"], v1["vy"],
                    v2["x_nm"], v2["y_nm"], v2["vx"], v2["vy"],
                )
                if 0 <= tcpa <= tcpa_max_hr and cpa <= cpa_threshold_nm:
                    results.append({
                        "timestamp": snapshot,
                        "MMSI_1": v1["MMSI"], "MMSI_2": v2["MMSI"],
                        "VesselName_1": v1.get("VesselName", "Unknown"),
                        "VesselName_2": v2.get("VesselName", "Unknown"),
                        "lat_1": v1["LAT"], "lon_1": v1["LON"],
                        "lat_2": v2["LAT"], "lon_2": v2["LON"],
                        "cpa_nm": round(cpa, 2),
                        "tcpa_min": round(tcpa * 60, 1),
                    })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["risk_score"] = (
            (CPA_THRESHOLD_NM - result_df["cpa_nm"]).clip(lower=0)
            / max(CPA_THRESHOLD_NM, 1e-6)
        )
        result_df = result_df.sort_values("cpa_nm").reset_index(drop=True)
    return result_df


def collapse_encounters(result_df: pd.DataFrame, gap_minutes: int = EPISODE_GAP_MINUTES) -> pd.DataFrame:
    """
    Collapse consecutive per-minute flags for the same vessel pair into a
    single encounter episode, keeping the row with the minimum CPA as the
    representative snapshot. Two flags for the same pair count as the same
    episode if they're within `gap_minutes` of each other; a longer gap
    means the vessels separated and later re-approached — a new episode.
    """
    if result_df.empty:
        return result_df

    df = result_df.copy()
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["MMSI_1"], r["MMSI_2"]])), axis=1)
    df = df.sort_values(["pair_key", "timestamp"])

    df["gap"] = df.groupby("pair_key")["timestamp"].diff().dt.total_seconds().div(60)
    df["new_episode"] = df["gap"].isna() | (df["gap"] > gap_minutes)
    df["episode_id"] = df.groupby("pair_key")["new_episode"].cumsum()

    collapsed = (
        df.sort_values("cpa_nm")
        .groupby(["pair_key", "episode_id"], as_index=False)
        .first()
    )
    collapsed = collapsed.drop(columns=["pair_key", "gap", "new_episode", "episode_id"])
    return collapsed.sort_values("cpa_nm").reset_index(drop=True)