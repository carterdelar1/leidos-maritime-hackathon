"""
pipeline.py
Detects/predicts potential vessel-to-vessel collision risks from AIS data
using CPA (Closest Point of Approach) and TCPA (Time to Closest Point of
Approach) — Challenge 9: Ocean Vessel Collision Risk Predictor.

Covers the full MVS (project trajectories, identify converging pairs,
calculate CPA/TCPA, assign risk, rank encounters, visualize trajectories)
plus several stretch goals: vessel-size-adjusted thresholds, encounter
classification (head-on/crossing/overtaking), simplified COLREG give-way
reasoning, and rule-based plain-language explanations.

Usage:
    from pipeline import load_ais, predict_current_risks, detect_collision_risks, collapse_encounters

    ais = load_ais("data/AIS_June_Daily_2025")
    live_risks = predict_current_risks(ais)
    historical_risks = collapse_encounters(detect_collision_risks(ais))
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

CPA_THRESHOLD_NM = 0.3
TCPA_MAX_HR = 0.25
SNAPSHOT_FLOOR = "min"
MIN_REL_SPEED_KNOTS = 2.0
EPISODE_GAP_MINUTES = 20

# Vessel-size effect: add a buffer to the CPA threshold based on the
# combined length of both vessels, since bigger ships need more clearance
# to be genuinely "safe" at the same nominal distance.
SIZE_BUFFER_FACTOR = 0.5  # multiplier applied to combined length (nm)

# Map center/zoom used consistently across all map views so the view
# doesn't jump around between reruns with different numbers of points.
DEFAULT_MAP_CENTER = {"lat": 33.72, "lon": -118.20}
DEFAULT_MAP_ZOOM = 11

SECTORS = {
    "Anchorage":        {"lat_min": 33.60, "lat_max": 33.68, "lon_min": -118.30, "lon_max": -118.22},
    "Approach Channel": {"lat_min": 33.68, "lat_max": 33.74, "lon_min": -118.25, "lon_max": -118.18},
    "Inner Harbor":     {"lat_min": 33.74, "lat_max": 33.80, "lon_min": -118.22, "lon_max": -118.10},
}

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


def assign_sector(lat: float, lon: float) -> str:
    for name, bounds in SECTORS.items():
        if bounds["lat_min"] <= lat <= bounds["lat_max"] and bounds["lon_min"] <= lon <= bounds["lon_max"]:
            return name
    return "Unknown"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_zst_csv(path: str, chunksize: int = CHUNKSIZE) -> pd.DataFrame:
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
                    for col in ["LAT", "LON", "SOG", "COG", "Length", "Width"]:
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
    df = _load_ais_folder(path)

    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    df = df.sort_values(["MMSI", "BaseDateTime"]).drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["SOG", "COG"])

    if "Length" not in df.columns:
        df["Length"] = np.nan
    df["Length"] = df["Length"].fillna(df["Length"].median() if df["Length"].notna().any() else 100.0)

    df["sector"] = df.apply(lambda r: assign_sector(r["LAT"], r["LON"]), axis=1)

    return df


# ---------------------------------------------------------------------------
# Coordinate + trajectory math
# ---------------------------------------------------------------------------
def latlon_to_xy_nm(lat, lon, ref_lat, ref_lon):
    x = (lon - ref_lon) * 60 * np.cos(np.radians(ref_lat))
    y = (lat - ref_lat) * 60
    return x, y


def xy_nm_to_latlon(x, y, ref_lat, ref_lon):
    lon = ref_lon + x / (60 * np.cos(np.radians(ref_lat)))
    lat = ref_lat + y / 60
    return lat, lon


def velocity_components(sog_knots, cog_deg):
    heading_rad = np.radians(cog_deg)
    vx = sog_knots * np.sin(heading_rad)
    vy = sog_knots * np.cos(heading_rad)
    return vx, vy


def compute_cpa_tcpa(x1, y1, vx1, vy1, x2, y2, vx2, vy2):
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


def effective_cpa_threshold(length_1_m, length_2_m, base_threshold_nm):
    """
    Vessel-size stretch goal: bigger ships need more clearance to be
    genuinely 'safe' at the same nominal distance. Adds a buffer
    proportional to the combined vessel length (converted to nm).
    """
    combined_length_nm = (length_1_m + length_2_m) / 1852.0  # meters -> nm
    return base_threshold_nm + SIZE_BUFFER_FACTOR * combined_length_nm


# ---------------------------------------------------------------------------
# Encounter classification + simplified COLREG give-way reasoning
# ---------------------------------------------------------------------------
def relative_bearing(cog_from_deg, dx, dy):
    """Bearing FROM one vessel TO another, relative to the first vessel's
    own heading. Returns degrees in [-180, 180]; positive = target is to
    starboard (right), negative = target is to port (left)."""
    bearing_to_target = np.degrees(np.arctan2(dx, dy)) % 360
    rel = (bearing_to_target - cog_from_deg + 180) % 360 - 180
    return rel


def classify_encounter(cog1, cog2, dx, dy):
    """
    Classify the encounter type from relative course angle:
      - 'head-on': courses roughly opposite, each roughly ahead of the other
      - 'overtaking': courses roughly the same direction
      - 'crossing': anything else
    This is a simplified heuristic, not full COLREG geometry.
    """
    course_diff = abs((cog1 - cog2 + 180) % 360 - 180)  # 0-180

    if course_diff < 15:
        return "overtaking"
    elif course_diff > 165:
        return "head-on"
    else:
        return "crossing"


def determine_give_way(name1, name2, cog1, dx, dy, encounter_type):
    """
    SIMPLIFIED COLREG-style give-way determination — NOT a substitute for
    real navigational rules, but a reasonable rule-based approximation for
    demo/explanation purposes:
      - head-on: both vessels are expected to alter course to starboard
      - overtaking: the vessel coming from behind must keep clear
      - crossing: the vessel that has the other on her own starboard side
        is the give-way vessel (roughly mirrors real COLREG Rule 15)
    """
    if encounter_type == "head-on":
        return "Both vessels (standard head-on rule: alter course to starboard)"

    if encounter_type == "overtaking":
        # Vessel 2 is "ahead" in this dx/dy frame from vessel 1's perspective
        # if bearing to it is roughly along vessel 1's own course.
        rel = relative_bearing(cog1, dx, dy)
        return name1 if abs(rel) < 90 else name2

    # crossing
    rel = relative_bearing(cog1, dx, dy)
    return name1 if 0 < rel <= 112.5 else name2


def risk_level(cpa_nm, tcpa_min, threshold_nm):
    """Combine CPA and TCPA into a simple Low/Medium/High/Critical label."""
    closeness = max(0.0, (threshold_nm - cpa_nm) / max(threshold_nm, 1e-6))

    if closeness > 0.6 and tcpa_min < 5:
        return "Critical"
    elif closeness > 0.3 or tcpa_min < 8:
        return "High"
    elif closeness > 0.1 or tcpa_min < 15:
        return "Medium"
    else:
        return "Low"


def generate_encounter_explanation(row: dict) -> str:
    """Rule-based plain-language explanation of one flagged encounter."""
    return (
        f"{row['VesselName_1']} and {row['VesselName_2']} are on a "
        f"**{row['encounter_type']}** course, projected to close to "
        f"{row['cpa_nm']} nm in {row['tcpa_min']} minutes. "
        f"Risk level: **{row['risk_level']}**. "
        f"Under simplified give-way logic, **{row['give_way_vessel']}** "
        f"would be expected to maneuver to avoid the encounter."
    )


# ---------------------------------------------------------------------------
# Shared pairwise risk check
# ---------------------------------------------------------------------------
def _pairwise_risk_check(records, base_cpa_threshold_nm, tcpa_max_hr, sector_name=None):
    results = []
    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = records[i], records[j]

            threshold = effective_cpa_threshold(
                v1.get("Length", 100.0) or 100.0,
                v2.get("Length", 100.0) or 100.0,
                base_cpa_threshold_nm,
            )

            cpa, tcpa = compute_cpa_tcpa(
                v1["x_nm"], v1["y_nm"], v1["vx"], v1["vy"],
                v2["x_nm"], v2["y_nm"], v2["vx"], v2["vy"],
            )
            if 0 <= tcpa <= tcpa_max_hr and cpa <= threshold:
                dx, dy = v2["x_nm"] - v1["x_nm"], v2["y_nm"] - v1["y_nm"]
                encounter_type = classify_encounter(v1["COG"], v2["COG"], dx, dy)
                give_way = determine_give_way(
                    v1.get("VesselName", "Vessel 1"), v2.get("VesselName", "Vessel 2"),
                    v1["COG"], dx, dy, encounter_type,
                )

                row = {
                    "MMSI_1": v1["MMSI"], "MMSI_2": v2["MMSI"],
                    "VesselName_1": v1.get("VesselName", "Unknown"),
                    "VesselName_2": v2.get("VesselName", "Unknown"),
                    "lat_1": v1["LAT"], "lon_1": v1["LON"],
                    "lat_2": v2["LAT"], "lon_2": v2["LON"],
                    "cpa_nm": round(cpa, 2),
                    "tcpa_min": round(tcpa * 60, 1),
                    "threshold_nm": round(threshold, 2),
                    "encounter_type": encounter_type,
                    "give_way_vessel": give_way,
                }
                row["risk_level"] = risk_level(row["cpa_nm"], row["tcpa_min"], threshold)
                row["explanation"] = generate_encounter_explanation(row)
                if sector_name is not None:
                    row["sector"] = sector_name
                results.append(row)
    return results


# ---------------------------------------------------------------------------
# LIVE prediction
# ---------------------------------------------------------------------------
def predict_current_risks(ais_df, cpa_threshold_nm=CPA_THRESHOLD_NM, tcpa_max_hr=TCPA_MAX_HR, recency_window_min=10):
    if ais_df.empty:
        return pd.DataFrame()

    latest_overall = ais_df["BaseDateTime"].max()
    cutoff = latest_overall - pd.Timedelta(minutes=recency_window_min)

    # Only vessels whose latest ping falls within this recent window —
    # so "live" actually means "around the same real moment," not
    # "whenever this vessel last happened to report."
    recent = ais_df[ais_df["BaseDateTime"] >= cutoff]
    latest = recent.sort_values("BaseDateTime").groupby("MMSI").tail(1).copy()
    ...

# ---------------------------------------------------------------------------
# HISTORICAL scan
# ---------------------------------------------------------------------------
def detect_collision_risks(
    ais_df: pd.DataFrame,
    cpa_threshold_nm: float = CPA_THRESHOLD_NM,
    tcpa_max_hr: float = TCPA_MAX_HR,
) -> pd.DataFrame:
    if ais_df.empty:
        return pd.DataFrame()

    ref_lat = ais_df["LAT"].mean()
    ref_lon = ais_df["LON"].mean()

    df = ais_df.copy()
    df["x_nm"], df["y_nm"] = latlon_to_xy_nm(df["LAT"], df["LON"], ref_lat, ref_lon)
    df["vx"], df["vy"] = velocity_components(df["SOG"], df["COG"])
    df["snapshot"] = df["BaseDateTime"].dt.floor(SNAPSHOT_FLOOR)

    results = []
    for snapshot, group in df.groupby("snapshot"):
        for sector_name, sector_group in group.groupby("sector"):
            vessels = sector_group.drop_duplicates(subset="MMSI")
            if len(vessels) < 2:
                continue
            records = vessels.to_dict("records")
            pair_results = _pairwise_risk_check(records, cpa_threshold_nm, tcpa_max_hr, sector_name)
            for r in pair_results:
                r["timestamp"] = snapshot
            results.extend(pair_results)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        risk_order = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0}
        result_df["_risk_rank"] = result_df["risk_level"].map(risk_order)
        result_df = result_df.sort_values(["_risk_rank", "cpa_nm"], ascending=[False, True])
        result_df = result_df.drop(columns=["_risk_rank"]).reset_index(drop=True)
    return result_df


def collapse_encounters(result_df: pd.DataFrame, gap_minutes: int = EPISODE_GAP_MINUTES) -> pd.DataFrame:
    if result_df.empty or "timestamp" not in result_df.columns:
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


# ---------------------------------------------------------------------------
# Trajectory geometry for map visualization
# ---------------------------------------------------------------------------
def get_encounter_trajectory(
    ais_df: pd.DataFrame,
    mmsi_1,
    mmsi_2,
    ref_lat: float,
    ref_lon: float,
    recent_pings: int = 5,
) -> dict:
    v1_hist = ais_df[ais_df["MMSI"] == mmsi_1].sort_values("BaseDateTime").tail(recent_pings)
    v2_hist = ais_df[ais_df["MMSI"] == mmsi_2].sort_values("BaseDateTime").tail(recent_pings)

    if v1_hist.empty or v2_hist.empty:
        return {}

    v1_latest = v1_hist.iloc[-1]
    v2_latest = v2_hist.iloc[-1]

    x1, y1 = latlon_to_xy_nm(v1_latest["LAT"], v1_latest["LON"], ref_lat, ref_lon)
    x2, y2 = latlon_to_xy_nm(v2_latest["LAT"], v2_latest["LON"], ref_lat, ref_lon)
    vx1, vy1 = velocity_components(v1_latest["SOG"], v1_latest["COG"])
    vx2, vy2 = velocity_components(v2_latest["SOG"], v2_latest["COG"])

    cpa_nm, tcpa_hr = compute_cpa_tcpa(x1, y1, vx1, vy1, x2, y2, vx2, vy2)
    if not np.isfinite(tcpa_hr):
        tcpa_hr = 0.0

    p1x, p1y = x1 + vx1 * tcpa_hr, y1 + vy1 * tcpa_hr
    p2x, p2y = x2 + vx2 * tcpa_hr, y2 + vy2 * tcpa_hr

    p1_lat, p1_lon = xy_nm_to_latlon(p1x, p1y, ref_lat, ref_lon)
    p2_lat, p2_lon = xy_nm_to_latlon(p2x, p2y, ref_lat, ref_lon)

    cpa_x, cpa_y = (p1x + p2x) / 2, (p1y + p2y) / 2
    cpa_lat, cpa_lon = xy_nm_to_latlon(cpa_x, cpa_y, ref_lat, ref_lon)

    return {
        "track_1": list(zip(v1_hist["LAT"], v1_hist["LON"])),
        "track_2": list(zip(v2_hist["LAT"], v2_hist["LON"])),
        "projected_1": [(v1_latest["LAT"], v1_latest["LON"]), (p1_lat, p1_lon)],
        "projected_2": [(v2_latest["LAT"], v2_latest["LON"]), (p2_lat, p2_lon)],
        "cpa_point": (cpa_lat, cpa_lon),
        "cpa_nm": round(cpa_nm, 2) if np.isfinite(cpa_nm) else None,
        "tcpa_min": round(tcpa_hr * 60, 1),
    }