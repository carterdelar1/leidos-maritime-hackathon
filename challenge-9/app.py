"""
app.py
Ocean Vessel Collision Risk Predictor — Challenge 9

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (
    load_ais,
    detect_collision_risks,
    collapse_encounters,
    CPA_THRESHOLD_NM,
    TCPA_MAX_HR,
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

st.sidebar.header("Risk thresholds")
cpa_threshold = st.sidebar.slider("CPA threshold (nm)", 0.1, 3.0, CPA_THRESHOLD_NM, 0.1)
tcpa_max = st.sidebar.slider("Max TCPA (minutes)", 5, 60, int(TCPA_MAX_HR * 60), 5)


@st.cache_data
def get_risks(_ais_df, cpa_threshold_nm, tcpa_max_hr):
    raw = detect_collision_risks(_ais_df, cpa_threshold_nm=cpa_threshold_nm, tcpa_max_hr=tcpa_max_hr)
    return collapse_encounters(raw)


with st.spinner("Scanning for collision risks..."):
    risks = get_risks(ais_df, cpa_threshold, tcpa_max / 60)

col1, col2 = st.columns(2)
col1.metric("Flagged encounter episodes", len(risks))
col2.metric("Vessels in dataset", ais_df["MMSI"].nunique())

st.divider()

st.subheader("Ranked risk encounters")
if not risks.empty:
    st.dataframe(
        risks[["timestamp", "VesselName_1", "VesselName_2", "cpa_nm", "tcpa_min", "risk_score"]]
        .sort_values("cpa_nm"),
        use_container_width=True,
    )
else:
    st.info("No encounters met the current risk thresholds. Try loosening them in the sidebar.")

st.divider()

st.subheader("Encounter map")
if not risks.empty:
    map_points = pd.concat([
        risks[["lat_1", "lon_1"]].rename(columns={"lat_1": "lat", "lon_1": "lon"}).assign(vessel=risks["VesselName_1"]),
        risks[["lat_2", "lon_2"]].rename(columns={"lat_2": "lat", "lon_2": "lon"}).assign(vessel=risks["VesselName_2"]),
    ])
    fig_map = px.scatter_map(
        map_points,
        lat="lat",
        lon="lon",
        hover_name="vessel",
        zoom=10,
        height=500,
    )
    fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No encounters to map yet.")