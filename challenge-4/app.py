"""
app.py
Seaport Operations Dashboard — Port of Long Beach (Challenge 4)

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (
    PORT_NAME,
    PORT_LAT,
    PORT_LON,
    load_ais,
    compute_events,
    compute_kpis,
    generate_situation_report,
    compute_hourly_activity,
    compute_congestion,
    detect_dwell_anomalies,
)

st.set_page_config(page_title="Seaport Operations Dashboard", layout="wide")

st.title(f"{PORT_NAME} — operations dashboard")
st.caption("Prototype for education/demonstration only — not a certified operational system.")

# ---------------------------------------------------------------------------
# Load + process data (cached so filtering doesn't re-run the pipeline)
# ---------------------------------------------------------------------------
DATA_PATH = "data/AIS_June_Daily_2025"  # point this at your Marine Cadastre extract

MAP_POINT_CAP = 1000
TABLE_ROW_CAP = 500


@st.cache_data
def get_data(path):
    ais = load_ais(path)
    events = compute_events(ais)
    return ais, events


try:
    ais_df, events_df = get_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find {DATA_PATH}. Update DATA_PATH in app.py to your AIS extract.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

classes = ["All"] + sorted(events_df["vessel_class"].dropna().unique().tolist())
selected_class = st.sidebar.selectbox("Vessel class", classes)

min_date = events_df["timestamp"].min().date()
max_date = events_df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)

filtered = events_df.copy()
if selected_class != "All":
    filtered = filtered[filtered["vessel_class"] == selected_class]
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)
    ]

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
kpis = compute_kpis(filtered, ais_df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vessels in area now", kpis["vessels_in_area"])
col2.metric("Arrivals (period)", kpis["arrivals"])
col3.metric("Departures (period)", kpis["departures"])
col4.metric("Avg dwell time (hrs)", kpis["avg_dwell_hours"])

st.divider()

## etc

st.divider()

st.subheader("Congestion over time")
if not filtered.empty:
    congestion = compute_congestion(filtered, resample="h", threshold=5)
    if not congestion.empty:
        fig_congestion = px.line(
            congestion,
            x="timestamp",
            y="occupancy",
            markers=False,
            labels={"timestamp": "Time", "occupancy": "Vessels in area"},
        )
        fig_congestion.add_hline(y=5, line_dash="dash", line_color="red",
                                  annotation_text="Congestion threshold")
        st.plotly_chart(fig_congestion, use_container_width=True)

        congested_periods = congestion[congestion["congested"]]
        if not congested_periods.empty:
            st.warning(f"⚠️ {len(congested_periods)} hour(s) in this window exceeded the congestion threshold.")
        else:
            st.success("No congestion threshold breaches in this window.")
    else:
        st.info("Not enough data to compute congestion.")
else:
    st.info("No events in the selected filters.")

st.divider()

st.subheader("Anomaly alerts — unusually long dwell times")
if not filtered.empty:
    anomalies = detect_dwell_anomalies(filtered, z_thresh=2.0)
    if not anomalies.empty:
        st.dataframe(
            anomalies[["MMSI", "dwell_hours", "z_score"]].round(2),
            use_container_width=True,
        )
        st.caption(f"{len(anomalies)} vessel(s) flagged with dwell time significantly above average (z ≥ 2.0).")
    else:
        st.success("No dwell-time anomalies detected in this window.")
else:
    st.info("No events in the selected filters.")

## time

st.divider()

st.subheader("Arrivals vs departures by hour of day")
if not filtered.empty:
    hourly = compute_hourly_activity(filtered)
    fig_hourly = px.line(
        hourly,
        x="hour",
        y="count",
        color="event",
        markers=True,
        labels={"hour": "Hour of day", "count": "Event count"},
    )
    fig_hourly.update_xaxes(dtick=1, range=[-0.5, 23.5])
    st.plotly_chart(fig_hourly, use_container_width=True)
else:
    st.info("No events in the selected filters.")

# ---------------------------------------------------------------------------
# Situation report
# ---------------------------------------------------------------------------
st.subheader("Port Situation Report")
st.markdown(generate_situation_report(kpis, PORT_NAME))

st.divider()

# ---------------------------------------------------------------------------
# Trend + class mix
# ---------------------------------------------------------------------------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("Arrivals vs departures by day")
    if not filtered.empty:
        daily = (
            filtered.assign(day=filtered["timestamp"].dt.date)
            .groupby(["day", "event"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(daily, x="day", y="count", color="event", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No events in the selected filters.")

with right:
    st.subheader("Vessel class mix")
    if not filtered.empty:
        class_counts = filtered["vessel_class"].value_counts().reset_index()
        class_counts.columns = ["vessel_class", "count"]
        fig2 = px.pie(class_counts, names="vessel_class", values="count", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No events in the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Map (capped point count to keep rendering fast)
# ---------------------------------------------------------------------------
st.subheader("Vessel activity map")

if not filtered.empty:
    if len(filtered) > MAP_POINT_CAP:
        map_data = filtered.sample(MAP_POINT_CAP, random_state=1)
        st.caption(f"Showing a random sample of {MAP_POINT_CAP:,} of {len(filtered):,} events for map performance.")
    else:
        map_data = filtered

    fig_map = px.scatter_map(
        map_data,
        lat="lat",
        lon="lon",
        color="event",
        hover_name="VesselName",
        hover_data={"vessel_class": True, "timestamp": True, "lat": False, "lon": False},
        zoom=10,
        height=500,
        center={"lat": PORT_LAT, "lon": PORT_LON},
    )
    fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No events in the selected filters.")

st.divider()

# ---------------------------------------------------------------------------
# Drill-down table (capped rows so a huge filtered set doesn't stall rendering)
# ---------------------------------------------------------------------------
st.subheader("Vessel activity log")

table_data = filtered.sort_values("timestamp", ascending=False)
if len(table_data) > TABLE_ROW_CAP:
    st.caption(f"Showing the most recent {TABLE_ROW_CAP:,} of {len(table_data):,} events.")
    table_data = table_data.head(TABLE_ROW_CAP)

st.dataframe(
    table_data[["VesselName", "vessel_class", "event", "timestamp", "lat", "lon"]],
    use_container_width=True,
)