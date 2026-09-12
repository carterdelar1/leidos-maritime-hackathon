# Seaport Operations Dashboard — Port of Long Beach
**Challenge 4 (Easy) — St. Louis Maritime Geospatial Hackathon**

## Problem
A major port needs a clear, real-time picture of inbound and outbound vessel
activity. This dashboard turns raw AIS (Automatic Identification System)
vessel-position data into operational awareness: who's arriving, who's
departing, how long vessels are staying, and where activity is concentrated.

## What it does
- Loads AIS vessel-position pings from NOAA/BOEM Marine Cadastre data
- Detects **arrival** and **departure** events by tracking when each vessel
  crosses into or out of a defined radius around the Port of Long Beach
- Classifies vessels into readable categories (Cargo, Tanker, Fishing, Other)
  from their AIS VesselType codes
- Computes four operational KPIs:
  - Vessels currently in the monitored area
  - Arrivals in the selected period
  - Departures in the selected period
  - Average dwell time (hours between arrival and departure)
- Generates a plain-language **port situation report** summarizing current
  conditions
- Visualizes activity through:
  - Arrivals vs. departures by day (bar chart)
  - Arrivals vs. departures by hour of day (line chart, intraday pattern)
  - Vessel class mix (pie chart)
  - An interactive map of vessel activity
  - A filterable, sortable activity log table
- Supports filtering by vessel class and date range

## How it works
1. **`pipeline.py`** — all data loading and logic:
   - Decompresses and reads daily `.csv.zst` AIS extracts in chunks (to
     control memory usage), filtering each chunk to the Long Beach / San
     Pedro Bay bounding box before combining files
   - Computes distance from each AIS ping to the port center using the
     haversine formula; flags pings within 2.5 nautical miles as "in area"
   - Detects arrival/departure events as speed-qualified crossings of that
     boundary (a vessel must be moving faster than 2 knots for the crossing
     to count, to avoid flagging vessels idling near the edge)
   - Pairs each arrival with its next departure to compute dwell time
   - Also includes a loader for the NGA World Port Index (Pub 150) port
     reference data
2. **`app.py`** — the Streamlit UI: loads data via `pipeline.py` (cached so
   filtering doesn't re-run the full pipeline), renders KPIs, the situation
   report, charts, map, and activity log

## Data sources
- **AIS vessel tracks** — NOAA/BOEM Marine Cadastre
  (https://marinecadastre.gov/ais/), Zone 11, June 2025 daily extracts
- **Port geography reference** — NGA World Port Index (Pub 150)
  (https://msi.nga.mil/Publications/WPI)

## Running it
1. Install dependencies:
2. Place daily AIS `.csv.zst` files in `data/AIS_June_Daily_2025/`
3. Run: streamlit run app.py
4. Open the local URL Streamlit prints (typically `http://localhost:8501`)

## Known limitations / assumptions
- **Port boundary is a radius, not the real harbor geometry.** "In area" is
  defined as within 2.5 nautical miles of a single fixed point
  (33.75°N, -118.19°W), not the actual breakwater/berth footprint. This is a
  simplification appropriate for a prototype, not a certified navigational
  boundary.
- **Vessel classification is partial.** Only Cargo (AIS codes 70–79), Tanker
  (80–89), and Fishing (30–39) are currently mapped; all other AIS VesselType
  codes fall into "Other." This could be extended using the full AIS type
  code reference.
- **Map and table are sampled/capped for performance.** The map displays a
  random sample of up to 1,000 points and the activity log shows the most
  recent 500 rows when the filtered result set is larger, to keep the
  dashboard responsive.
- **This is a prototype for education/demonstration only.** It is not a
  certified navigation, maritime safety, or operational system.

## AI-assisted development
This project used AI assistance throughout — including debugging data-loading
and memory issues, writing/vectorizing the event-detection and KPI logic, and
generating the situation report and visualization code. All AI-suggested code
was reviewed, tested against real AIS data, and modified (e.g., the original
event-detection logic used a slow row-by-row loop; it was rewritten to a
vectorized pandas approach after profiling showed it was too slow on a full
day of AIS data).