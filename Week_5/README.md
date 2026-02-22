# 🚚 CS 5542 — Week 5: Logistics Operation Dashboard

*A minimal, reproducible Data Engineering pipeline: Data → Snowflake → Query → App → Logging*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![Snowflake](https://img.shields.io/badge/Snowflake-Data_Warehouse-29B5E8.svg?style=for-the-badge&logo=snowflake&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](#)

</div>

---

## 🔗 Live Deployment

> **Streamlit Cloud:** [🚀 Launch Lab5_Streamlit_Cloud](https://cs5542lab5.streamlit.app/)

---

## 🏗️ Pipeline Architecture & End-to-End Flow

![logistics_pipeline_architecture_1771661265221](https://github.com/user-attachments/assets/972aed5f-27d4-442e-bce0-eb888fb42cd1)


---

## 🗂️ Project Structure

Below is the complete directory structure for the Week 5 Logistics Dashboard pipeline. It is organized into distinct layers for data generation, SQL execution, Python orchestration, and front-end visualization.

```text
Week_5/
├── app/                        # Presentation Layer
│   └── streamlit_app.py        # Main 7-tab Streamlit dashboard application
├── data/                       # Synthetic Logistics Datasets (14 Core/Ext Tables)
│   ├── customers.csv           # 200 rows 
│   ├── drivers.csv             # 150 rows
│   ├── trucks.csv              # 100 rows
│   ├── routes.csv              # 80 rows
│   ├── loads.csv               # 500 rows
│   ├── trips.csv               # 600 rows
│   ├── fuel_purchases.csv      # 400 rows
│   ├── trailers.csv            # Extension table
│   ├── facilities.csv          # Extension table
│   ├── delivery_events.csv     # Extension table
│   ├── maintenance_records.csv # Extension table
│   ├── safety_incidents.csv    # Extension table
│   ├── driver_monthly_metrics.csv
│   └── truck_utilization_metrics.csv
├── logs/                       # Pipeline Monitoring
│   └── pipeline_logs.csv       # Tracks ingestion success, errors, and latencies
├── scripts/                    # Ingestion & Orchestration Scripts
│   ├── load_local_csv_to_stage.py # Local/Batch ingestion using internal staging
│   └── run_pipeline.py         # Automated master orchestrator (Local & S3 modes)
├── sql/                        # Data Warehouse Definition & Analytics
│   ├── 01_create_schema.sql    # DDL for Database, Schema, and 14 Tables
│   ├── 02_stage_and_load.sql   # Warehouse config, Internal Stage, COPY INTO
│   ├── 03_queries.sql          # 5 core analytical queries (Revenue, Efficiency, etc.)
│   ├── 04_views.sql            # 5 derived views for dashboard visualization
│   ├── 05_derived_analytics.sql# 4 advanced materialized analytics tables
│   └── 06_s3_pipeline.sql      # AWS Storage Integration & External Stage setup
├── .env.example                # Template for Snowflake & AWS credentials
├── CONTRIBUTIONS.md            # Individual team member accountability log
├── README.md                   # Primary documentation
└── requirements.txt            # Python package dependencies
```
---

## 🎯 Week 5 Scope (≈50%)

| Scope Item | Included this week | Deferred |
| --- | --- | --- |
| **Dataset(s)** | All 14 trucking tables *(customers, drivers, trucks, routes, loads, trips, fuel_purchases + trailers, facilities, delivery_events, maintenance_records, safety_incidents, driver/truck metrics)* | — |
| **Feature(s)** | Schema + staging + `COPY INTO`, 5 analytical queries, 5 views, batch Python loader, 5-tab Streamlit dashboard, pipeline monitoring | — |

---

## 🚀 Setup Instructions

**1. Configure Environment**
Copy the example environment file and fill in your Snowflake credentials.

```bash
cp .env.example .env

```

**2. Install Dependencies**

```bash
pip install -r requirements.txt

```

---

## ❄️ Snowflake SQL Setup

Run these scripts in a Snowflake Worksheet (in order).

> 💡 **Tip:** Steps 1–5 are fully automated by `scripts/run_pipeline.py` (see the Load Data section below).

1. `sql/01_create_schema.sql` — Creates database + 14 tables.
2. `sql/02_stage_and_load.sql` — Configures warehouse, file format, internal stage, and executes `COPY INTO`.
3. `sql/04_views.sql` — Generates 5 derived views for the dashboard.
4. `sql/05_derived_analytics.sql` — Builds 4 advanced derived analytics tables.
5. `sql/06_s3_pipeline.sql` — Sets up S3 storage integration, external stage, and runs `COPY INTO` from S3.

---

## 📥 Load Data

Choose your preferred ingestion method below:

### 🤖 Automated Pipeline (Recommended)

Load all 14 tables from S3, build views + derived analytics, and log the run in one command:

```bash
python scripts/run_pipeline.py

```

*Alternative pipeline flags:*

```bash
# Use local CSVs instead of S3
python scripts/run_pipeline.py --local

# Skip storage integration creation (if it already exists)
python scripts/run_pipeline.py --skip-s3-setup

```

### 📂 Manual Batch (Internal Stage)

```bash
python scripts/load_local_csv_to_stage.py --batch

```

### 📄 Single Table Loading

```bash
python scripts/load_local_csv_to_stage.py data/customers.csv CUSTOMERS
python scripts/load_local_csv_to_stage.py data/drivers.csv DRIVERS
# ... repeat for other tables as needed

```

---

## 🧠 Analytical Queries

Run `sql/03_queries.sql` after loading your data to test these core insights:

1. **Revenue by Customer:** Top customers by total completed-load revenue.
2. **Driver Fuel Efficiency:** Average MPG per driver, ranked.
3. **Route Profitability:** Revenue minus fuel cost per route (4-table join).
4. **Monthly Revenue Trend:** Time-series analysis utilizing `DATE_TRUNC`.
5. **Truck Fleet Utilization:** Filtered multi-join with aggregation.

---

## 📈 Streamlit Dashboard

Launch the interactive application locally:

```bash
streamlit run app/streamlit_app.py

```

### Dashboard Layout

| Tab | Features |
| --- | --- |
| 📊 **Overview** | KPI cards + monthly revenue line chart (date-range filter). |
| 🚛 **Fleet & Drivers** | Truck/driver performance (fuel-type multi-select, min-trips slider). |
| 🗺️ **Routes** | Route scorecard (margin threshold, min-loads filter). |
| ⛽ **Fuel Spend** | Fuel spend by state (state filter, top-N slider). |
| 📈 **Monitoring** | Performance stats, latency over time, and raw query logs. |
| 🔬 **Analytics** | Advanced derived tables (Driver rankings, Truck health, Route quality). |
| 🎯 **Executive** | Auto-loading KPIs, terminal heatmap, and a live SQL explorer. |

---

## ✨ Extensions Completed

* [x] **Extension 1: Full dataset ingestion** — Ingested all 14 trucking CSVs, including trailers, facilities, maintenance_records, and more.
* [x] **Extension 2: Pipeline monitoring** — Dedicated `📈 Monitoring` tab with performance summary, latency charts, and query stats.
* [x] **Extension 3: Advanced derived analytics** — `05_derived_analytics.sql` creates materialized tables for driver rankings, truck health, and route quality.
* [x] **Extension 4: Automated S3 ingestion pipeline** — `scripts/run_pipeline.py` provides one-command orchestration for schema creation and S3 data loading.
* [x] **Extension 5: Interactive executive dashboard** — `🎯 Executive` tab with auto-loading KPIs, heatmap, and a live SQL explorer.

---

## 📂 Data Source

All 14 CSV datasets in `data/` are sourced from the publicly available Kaggle dataset:

> **[Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database)**  
> Kaggle · *yogape* · Logistics / Transportation domain

The CSVs were downloaded as-is and ingested into Snowflake via the staging scripts in `scripts/`.

---

## 🎬 Demo & Notes

* **Demo Video Link:** *(Insert link here)*
  
* **Notes / Bottlenecks:**
  
    - **Security & IAM**: ACCOUNTADMIN is required to run the `STORAGE INTEGRATION` step once. AWS IAM roles must manually trust the Snowflake principal (found via `DESCRIBE`) for the S3 pipeline to work.
    - **Data Freshness**: The derived tables in `05_derived_analytics.sql` are "on-demand" materialized. They need a manual re-run (or a Snowflake Task) after new data loads to reflect the latest rankings and scores.
    - **Scaling Limits**: The Streamlit SQL Explorer caps results at 500 rows to prevent browser OOM. For massive datasets (>1M rows), analytics should be pushed to Snowflake views rather than processed in Pandas.
    - **Cold Starts**: The first query in a session may take 5–10s if the Snowflake warehouse is suspended. Subsequent cached queries (TTL: 2 min) are sub-second.
    - **S3 Connectivity**: The pipeline assumes CSVs are in the `/data/` prefix of the bucket. Metadata mismatches in S3 will cause `COPY INTO` failures logged in `pipeline_logs.csv`.
 
---

## 📄 License

This project is for academic use as part of CS 5542 at the University of Missouri–Kansas City.
