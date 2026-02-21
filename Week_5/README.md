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
flowchart LR
A[14 CSVs — Trucking Data] --> B[Snowflake Stage + COPY]
B --> C["7 Tables (4 dim + 3 fact)"]
C --> D[5 Views]
D --> E[4-Tab Streamlit Dashboard]
E --> F[Monitoring Logs]
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
2. `sql/02_stage_and_load.sql` — warehouse, file format, stage, COPY INTO
3. `sql/04_views.sql` — 5 derived views for the dashboard
4. `sql/05_derived_analytics.sql` — 4 advanced derived analytics tables

## Load Data

### Batch (all 7 tables at once — recommended)
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
| 📊 Overview | KPI cards + monthly revenue line chart (date-range filter) |
| 🚛 Fleet & Drivers | Truck/driver performance (fuel-type multi-select, min-trips slider) |
| 🗺️ Routes | Route scorecard (margin threshold, min-loads filter) |
| ⛽ Fuel Spend | Fuel spend by state (state filter, top-N slider) |

## Extensions Completed
- **Extension 1: Full dataset ingestion** — ingested all 14 trucking CSVs (added trailers, facilities, delivery_events, maintenance_records, safety_incidents, driver_monthly_metrics, truck_utilization_metrics)
- **Extension 2: Pipeline monitoring** — auto-logging with `perf_note`, latency charts, per-query stats, and performance summary in `📈 Monitoring` tab
- **Extension 3: Advanced derived analytics** — `05_derived_analytics.sql` creates 4 materialized tables using window functions (RANK, NTILE, PERCENT_RANK, LAG), CTEs, composite scoring, and cross-domain joins:
  - `DT_DRIVER_PERFORMANCE_RANKED` — driver rankings, safety scores, percentile bands
  - `DT_TRUCK_HEALTH_SCORECARD` — composite truck health score (MPG + downtime + incidents)
  - `DT_ROUTE_DELIVERY_QUALITY` — on-time %, detention burden, per-mile revenue with PERCENT_RANK
  - `DT_MONTHLY_OPERATIONS_SUMMARY` — MoM growth (LAG), 3-month rolling avg, net margin estimate

## Demo Video Link
- 

## Notes / Bottlenecks
- 
