"""
make_sample_data.py
Generates a synthetic AIS CSV matching the Marine Cadastre schema so you can
test pipeline.py and app.py before your real data extract is ready.

Usage:
    python make_sample_data.py
    # writes data/ais_long_beach.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

PORT_LAT, PORT_LON = 33.75, -118.19
N_VESSELS = 25
HOURS = 24 * 14  # 14 days of data
POINTS_PER_VESSEL = 60  # AIS pings per vessel over the window

VESSEL_TYPES = {
    "Cargo": list(range(70, 80)),
    "Tanker": list(range(80, 90)),
    "Fishing": list(range(30, 39)),
    "Other": [0, 51, 52, 60],
}


def make_vessel_track(mmsi, vessel_type_code, vessel_name, start_time):
    """
    Simulate a vessel approaching the port, dwelling nearby, then departing.
    Produces a simple in -> near port -> out arc over POINTS_PER_VESSEL pings.
    """
    t = pd.date_range(start_time, periods=POINTS_PER_VESSEL, freq="20min")

    # distance profile: starts far, dips near the port mid-track, moves away again
    progress = np.linspace(0, 1, POINTS_PER_VESSEL)
    dist_deg = 0.9 * (1 - np.sin(progress * np.pi)) + 0.01  # ~0.01 to ~0.9 deg swing
    angle = np.random.uniform(0, 2 * np.pi)

    lat = PORT_LAT + dist_deg * np.cos(angle) * np.random.uniform(0.8, 1.2, POINTS_PER_VESSEL)
    lon = PORT_LON + dist_deg * np.sin(angle) * np.random.uniform(0.8, 1.2, POINTS_PER_VESSEL)

    sog = np.where(dist_deg < 0.05, np.random.uniform(0, 1, POINTS_PER_VESSEL),
                   np.random.uniform(5, 15, POINTS_PER_VESSEL))

    return pd.DataFrame({
        "MMSI": mmsi,
        "BaseDateTime": t,
        "LAT": lat,
        "LON": lon,
        "SOG": sog,
        "COG": np.random.uniform(0, 360, POINTS_PER_VESSEL),
        "VesselName": vessel_name,
        "VesselType": vessel_type_code,
        "Status": 0,
        "Length": np.random.uniform(100, 300),
        "Width": np.random.uniform(15, 45),
    })


def main():
    os.makedirs("data", exist_ok=True)
    frames = []
    start = pd.Timestamp("2026-09-01")

    class_names = list(VESSEL_TYPES.keys())
    for i in range(N_VESSELS):
        mmsi = 366000000 + i
        cls = class_names[i % len(class_names)]
        vt_code = np.random.choice(VESSEL_TYPES[cls])
        vessel_start = start + pd.Timedelta(hours=np.random.uniform(0, HOURS - 24))
        frames.append(make_vessel_track(mmsi, vt_code, f"Vessel_{i:03d}", vessel_start))

    df = pd.concat(frames, ignore_index=True).sort_values(["MMSI", "BaseDateTime"])
    out_path = "data/ais_long_beach.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows for {N_VESSELS} vessels to {out_path}")


if __name__ == "__main__":
    main()
