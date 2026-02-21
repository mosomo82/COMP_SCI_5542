# CS 5542 — Week 5 Snowflake Integration Starter

This starter kit provides a minimal, reproducible **Data → Snowflake → Query → App → Logging** pipeline.

## Repo Layout
- `sql/`: schema, staging/loading, and query examples
- `scripts/`: connection + local CSV → stage → COPY loader
- `app/`: Streamlit dashboard connected to Snowflake
- `data/`: sample CSVs (replace with your project subset)
- `logs/`: pipeline usage logs
- `CONTRIBUTIONS.md`: individual accountability

## Week 5 Scope (≈50%)

| Item | Included this week | Deferred |
|---|---|---|
| Dataset(s) | All 14 trucking tables (customers, drivers, trucks, routes, loads, trips, fuel_purchases + trailers, facilities, delivery_events, maintenance_records, safety_incidents, driver/truck metrics) | — |
| Feature(s) | Schema + staging + COPY INTO, 5 analytical queries, 5 views, batch Python loader, 5-tab Streamlit dashboard, pipeline monitoring | — |

## End-to-End Flow
```mermaid
flowchart TD
    A[14 CSVs — Synthetic Logistics Data] --> B[Automated Ingestion Pipeline: scripts/run_pipeline.py]
    B --> C{Stage Selection}
    C -->|Local| D[@CS5542_STAGE Internal]
    C -->|S3| E[@CS5542_S3_STAGE External]
    D & E --> F[Snowflake: CS5542_WEEK5.PUBLIC]
    F --> G[14 Tables: Core + Extensions]
    G --> H[Views & Derived Analytics]
    H --> I[7-Tab Streamlit Dashboard]
    I --> J[pipeline_logs.csv]
```

## Setup
1) Create `.env` from `.env.example` and fill your Snowflake values.
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Snowflake SQL Setup
Run these scripts in a Snowflake Worksheet (in order):
1. `sql/01_create_schema.sql` — creates database + 14 tables
2. `sql/02_stage_and_load.sql` — warehouse, file format, internal stage, COPY INTO
3. `sql/04_views.sql` — 5 derived views for the dashboard
4. `sql/05_derived_analytics.sql` — 4 advanced derived analytics tables
5. `sql/06_s3_pipeline.sql` — S3 storage integration, external stage, COPY INTO from S3

> **Tip:** Steps 1–5 are fully automated by `scripts/run_pipeline.py` — see below.

## Load Data

### 🤖 Automated pipeline (recommended)
```bash
# Load all 14 tables from S3, build views + derived analytics, log the run:
py scripts/run_pipeline.py

# Use local CSVs instead of S3:
py scripts/run_pipeline.py --local

# Skip storage integration creation (already exists):
py scripts/run_pipeline.py --skip-s3-setup
```

### Manual batch (internal stage)
```bash
python scripts/load_local_csv_to_stage.py --batch
```

### Single table
```bash
python scripts/load_local_csv_to_stage.py data/customers.csv CUSTOMERS
python scripts/load_local_csv_to_stage.py data/drivers.csv DRIVERS
python scripts/load_local_csv_to_stage.py data/trucks.csv TRUCKS
python scripts/load_local_csv_to_stage.py data/routes.csv ROUTES
python scripts/load_local_csv_to_stage.py data/loads.csv LOADS
python scripts/load_local_csv_to_stage.py data/trips.csv TRIPS
python scripts/load_local_csv_to_stage.py data/fuel_purchases.csv FUEL_PURCHASES
```

## Analytical Queries
Run `sql/03_queries.sql` after loading data:
1. **Q1: Revenue by customer** — top customers by total completed-load revenue
2. **Q2: Driver fuel efficiency** — avg MPG per driver, ranked
3. **Q3: Route profitability** — revenue minus fuel cost per route (4-table join)
4. **Q4: Monthly revenue trend** — time-series analysis with DATE_TRUNC
5. **Q5: Truck fleet utilization** — filtered multi-join with aggregation

## Dashboard
```bash
streamlit run app/streamlit_app.py
```

| Tab | Description |
|---|---|
| 📊 Overview | KPI cards + monthly revenue line chart (date-range filter). |
| 🚛 Fleet & Drivers | Truck/driver performance (fuel-type multi-select, min-trips slider). |
| 🗺️ Routes | Route scorecard (margin threshold, min-loads filter). |
| ⛽ Fuel Spend | Fuel spend by state (state filter, top-N slider). |
| 📈 Monitoring | Performance stats, latency over time, and raw query logs. |
| 🔬 Analytics | Advanced derived tables (Driver rankings, Truck health, Route quality). |
| 🎯 Executive | Auto-loading KPIs, terminal heatmap, and live SQL explorer. |

## Extensions Completed
- **Extension 1: Full dataset ingestion** — Ingested all 14 trucking CSVs, including trailers, facilities, maintenance_records, and more.
- **Extension 2: Pipeline monitoring** — Dedicated `📈 Monitoring` tab with performance summary, latency charts, and query stats.
- **Extension 3: Advanced derived analytics** — `05_derived_analytics.sql` creates materialized tables for driver rankings, truck health, and route quality.
- **Extension 4: Automated S3 ingestion pipeline** — `scripts/run_pipeline.py` provides one-command orchestration for schema creation and S3 data loading.
- **Extension 5: Interactive executive dashboard** — `🎯 Executive` tab with auto-loading KPIs, heatmap, and a live SQL explorer.

## Demo Video Link
- 

## Notes / Bottlenecks
- 
