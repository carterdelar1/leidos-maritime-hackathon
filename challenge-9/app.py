"""
app.py
Ocean Vessel Collision Risk Predictor — Challenge 9

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    load_ais,
    predict_current_risks,
    detect_collision_risks,
    collapse_encounters,
    get_encounter_trajectory,
    CPA_THRESHOLD_NM,
    TCPA_MAX_HR,
    DEFAULT_MAP_CENTER,
    DEFAULT_MAP_ZOOM,
)

st.set_page_config(page_title="Collision Risk Predictor", layout="wide")

st.title("Ocean Vessel Collision Risk Predictor")
st.caption("Prototype for education/demonstration only — not a certified navigation or safety system.")

DATA_PATH = "data/AIS_June_Daily_2025"


@st.cache_data
def get_data(path):
    return load_ais(path)


try:
    ais_df = get_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find {DATA_PATH}. Update DATA_PATH in app.py to your AIS extract.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Prediction mode",
    ["Live prediction (current positions)", "Historical scan (past data)"],
    help=(
        "Live: projects trajectories forward from each vessel's most recent "
        "known position — predicts risk right now.\n\n"
        "Historical: scans every past snapshot in the loaded data to find "
        "and validate past near-miss episodes."
    ),
)

st.sidebar.header("Risk thresholds")
cpa_threshold = st.sidebar.slider("Base CPA threshold (nm)", 0.05, 2.0, CPA_THRESHOLD_NM, 0.05)
tcpa_max = st.sidebar.slider("Max TCPA (minutes)", 5, 60, int(TCPA_MAX_HR * 60), 5)
st.sidebar.caption("Actual threshold used per pair is widened based on combined vessel length (vessel-size effect).")

sector_options = ["All"] + sorted(ais_df["sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("Sector", sector_options)

st.sidebar.info(
    "Note: this dataset is static historical AIS (two fixed days), not a "
    "live feed. 'Continuous monitoring' would require a real-time AIS "
    "stream — out of scope for this prototype."
)


@st.cache_data
def get_live_risks(_ais_df, cpa_threshold_nm, tcpa_max_hr):
    return predict_current_risks(_ais_df, cpa_threshold_nm=cpa_threshold_nm, tcpa_max_hr=tcpa_max_hr)


@st.cache_data
def get_historical_risks(_ais_df, cpa_threshold_nm, tcpa_max_hr):
    raw = detect_collision_risks(_ais_df, cpa_threshold_nm=cpa_threshold_nm, tcpa_max_hr=tcpa_max_hr)
    return collapse_encounters(raw)


with st.spinner("Computing collision risks..."):
    if mode.startswith("Live"):
        risks = get_live_risks(ais_df, cpa_threshold, tcpa_max / 60)
        risk_label = "Live flagged risks (right now)"
    else:
        risks = get_historical_risks(ais_df, cpa_threshold, tcpa_max / 60)
        risk_label = "Flagged encounter episodes (historical)"

if selected_sector != "All" and not risks.empty and "sector" in risks.columns:
    risks = risks[risks["sector"] == selected_sector]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric(risk_label, len(risks))
col2.metric("Vessels in dataset", ais_df["MMSI"].nunique())
if not risks.empty and "risk_level" in risks.columns:
    col3.metric("Critical-risk encounters", int((risks["risk_level"] == "Critical").sum()))
else:
    col3.metric("Critical-risk encounters", 0)

st.divider()

# ---------------------------------------------------------------------------
# Ranked table
# ---------------------------------------------------------------------------
st.subheader("Ranked risk encounters")
if not risks.empty:
    display_cols = [
        "VesselName_1", "VesselName_2", "cpa_nm", "tcpa_min",
        "risk_level", "encounter_type", "give_way_vessel",
    ]
    if "sector" in risks.columns:
        display_cols.insert(0, "sector")
    if "timestamp" in risks.columns:
        display_cols.insert(0, "timestamp")

    st.dataframe(risks[display_cols], use_container_width=True)
else:
    st.info("No encounters met the current risk thresholds. Try loosening them in the sidebar.")

st.divider()

# ---------------------------------------------------------------------------
# Trajectory projection for a selected encounter
# ---------------------------------------------------------------------------
st.subheader("Trajectory projection")

if not risks.empty:
    risks_display = risks.reset_index(drop=True)
    options = [
        f"{i}: {row['VesselName_1']} vs {row['VesselName_2']} "
        f"({row['risk_level']}, CPA {row['cpa_nm']}nm, TCPA {row['tcpa_min']}min)"
        for i, row in risks_display.iterrows()
    ]
    selected = st.selectbox("Choose an encounter to visualize", options)
    selected_idx = int(selected.split(":")[0])
    row = risks_display.iloc[selected_idx]

    st.markdown(row["explanation"])

    ref_lat = ais_df["LAT"].mean()
    ref_lon = ais_df["LON"].mean()
    traj = get_encounter_trajectory(ais_df, row["MMSI_1"], row["MMSI_2"], ref_lat, ref_lon)

    if traj:
        fig_traj = go.Figure()

        t1_lat, t1_lon = zip(*traj["track_1"])
        t2_lat, t2_lon = zip(*traj["track_2"])
        fig_traj.add_trace(go.Scattermap(lat=list(t1_lat), lon=list(t1_lon), mode="lines+markers",
                                          line=dict(color="blue"), name=row["VesselName_1"]))
        fig_traj.add_trace(go.Scattermap(lat=list(t2_lat), lon=list(t2_lon), mode="lines+markers",
                                          line=dict(color="red"), name=row["VesselName_2"]))

        p1_lat, p1_lon = zip(*traj["projected_1"])
        p2_lat, p2_lon = zip(*traj["projected_2"])
        # Scattermap's line object has no `dash` property (unlike regular
        # Scatter traces on a Cartesian plot) — use a thinner, semi-
        # transparent line instead to visually distinguish the projected
        # path from the solid actual track.
        fig_traj.add_trace(go.Scattermap(lat=list(p1_lat), lon=list(p1_lon), mode="lines",
                                          line=dict(color="blue", width=2),
                                          opacity=0.5,
                                          name=f"{row['VesselName_1']} projected"))
        fig_traj.add_trace(go.Scattermap(lat=list(p2_lat), lon=list(p2_lon), mode="lines",
                                          line=dict(color="red", width=2),
                                          opacity=0.5,
                                          name=f"{row['VesselName_2']} projected"))

        cpa_lat, cpa_lon = traj["cpa_point"]
        fig_traj.add_trace(go.Scattermap(
            lat=[cpa_lat], lon=[cpa_lon], mode="markers",
            marker=dict(size=14, color="black"),
            name=f"CPA: {traj['cpa_nm']}nm in {traj['tcpa_min']}min",
        ))

        fig_traj.update_layout(
            map_style="open-street-map",
            map_center=DEFAULT_MAP_CENTER,
            map_zoom=DEFAULT_MAP_ZOOM,
            height=550,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.info("Not enough recent history for this vessel pair to plot a trajectory.")
else:
    st.info("No flagged encounters to visualize.")

st.divider()

# ---------------------------------------------------------------------------
# Overview encounter map (fixed center/zoom, consistent across reruns)
# ---------------------------------------------------------------------------
st.subheader("Encounter map (all flagged pairs)")

if not risks.empty:
    map_points = pd.concat([
        risks[["lat_1", "lon_1"]].rename(columns={"lat_1": "lat", "lon_1": "lon"})
            .assign(vessel=risks["VesselName_1"], role="Vessel 1"),
        risks[["lat_2", "lon_2"]].rename(columns={"lat_2": "lat", "lon_2": "lon"})
            .assign(vessel=risks["VesselName_2"], role="Vessel 2"),
    ])
    fig_map = px.scatter_map(
        map_points, lat="lat", lon="lon", color="role", hover_name="vessel",
        zoom=DEFAULT_MAP_ZOOM, center=DEFAULT_MAP_CENTER, height=500,
    )
else:
    fig_map = px.scatter_map(
        pd.DataFrame({"lat": [DEFAULT_MAP_CENTER["lat"]], "lon": [DEFAULT_MAP_CENTER["lon"]]}),
        lat="lat", lon="lon", zoom=DEFAULT_MAP_ZOOM, center=DEFAULT_MAP_CENTER, height=500,
    )

fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)