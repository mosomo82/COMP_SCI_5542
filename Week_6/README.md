# 🚚 CS 5542 — Week 6: Logistics Operation Dashboard

*A minimal, reproducible Data Engineering pipeline: Data → Snowflake → Query → App → Logging*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![Snowflake](https://img.shields.io/badge/Snowflake-Data_Warehouse-29B5E8.svg?style=for-the-badge&logo=snowflake&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](#)

</div>

---

## 🔗 Live Deployment

> **Streamlit Cloud:** [🚀 Launch Lab6_Streamlit_Cloud](https://cs5542lab5.streamlit.app/)

---

## 🏗️ Pipeline Architecture & End-to-End Flow

![logistics_pipeline_architecture_1771661265221](https://github.com/user-attachments/assets/972aed5f-27d4-442e-bce0-eb888fb42cd1)


---

## 🗂️ Project Structure

Below is the complete directory structure for the Week 6 Logistics Dashboard pipeline. It is organized into distinct layers for data generation, SQL execution, Python orchestration, and front-end visualization.

```text
Week_6/
├── app/                        # Presentation Layer
│   └── streamlit_app.py        # Main 8-tab Streamlit dashboard application
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
├── reports/                    # Reports delivered
│   ├── CONTRIBUTIONS_DE.md     # Contributions by Daniel Evans
│   ├── CONTRIBUTIONS_JV.md     # Contributions by Joel Vinas
│   └── CONTRIBUTIONS_TN.md     # Contributions by Tony Nguyen
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
├── CONTRIBUTIONS.md            # Collective Team Member accountability log
├── README.md                   # Primary documentation
└── requirements.txt            # Python package dependencies
```
---

## 🎯 Lab 6 Scope (Multi-Agent Analytics)

| Scope Item | Included this week | Deferred |
| --- | --- | --- |
| **AI Integration** | Gemini 2.5 Flash Agent with 9 specialized data tools | — |
| **Dataset(s)** | All 14 trucking tables (including safety, maintenance, fuel) | — |
| **Feature(s)** | 9 Python agent tools, Tool schemas, CLI agent loop, 9-tab Streamlit dashboard | — |

---

## 🤖 AI Data Analytics Agent

The dashboard now features an autonomous **AI Data Analytics Agent** powered by Google Gemini 2.5 Flash. The agent can reason across 9 specialized tools to answer complex operational questions.

### Agent Tools
1.  **`query_snowflake`**: Arbitrary read-only SQL queries.
2.  **`get_monthly_revenue`**: Aggregated revenue trends.
3.  **`get_fleet_performance`**: Truck metrics (trips, miles, MPG, revenue).
4.  **`get_pipeline_logs`**: System health and ingestion latency.
5.  **`get_safety_metrics`**: Driver incident analytics and claims.
6.  **`get_route_profitability`**: Margin analysis by route.
7.  **`get_delivery_performance`**: On-time rates and detention times.
8.  **`get_maintenance_health`**: Maintenance costs, labor, and downtime.
9.  **`get_fuel_spend_analysis`**: Regional fuel spend breakdown.

---

## 🚀 Setup Instructions

**1. Configure Environment**
Copy the example environment file and fill in your Snowflake and Gemini credentials.

```bash
cp .env.example .env
# Ensure GEMINI_API_KEY is populated for the Agent to function.
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 📈 Streamlit Dashboard

Launch the interactive application locally:

```bash
streamlit run app/streamlit_app.py
```

### Dashboard Layout

| Tab | Description |
|---|---|
| 📊 Overview | KPI cards + monthly revenue line chart. |
| 🚛 Fleet & Drivers | Truck/driver performance filtering. |
| 🗺️ Routes | Route scorecard and margin thresholds. |
| ⛽ Fuel Spend | Fuel spend analysis by state/city. |
| 📈 Monitoring | Performance stats and latency charts. |
| 🔬 Analytics | Advanced materialized analytics tables. |
| 🎯 Executive | Auto-loading KPIs and live SQL explorer. |
| ⚠️ Safety | Driver hazard analysis and claim cost charts. |
| 🤖 Agent Chat | **NEW:** Interactive AI chat powered by Gemini and 9 tools. |

## Extensions Completed
- **Extension 1: Full dataset ingestion** — Ingested all 14 trucking CSVs.
- **Extension 2: Pipeline monitoring** — Dedicated `📈 Monitoring` tab.
- **Extension 3: Advanced derived analytics** — Materialized tables for rankings.
- **Extension 4: Automated S3 ingestion pipeline** — One-command orchestration.
- **Extension 5: Interactive executive dashboard** — Auto-loading KPIs and heatmap.
- **Extension 6: Safety Incidents dashboard tab** — Dedicated incident analytics.
- **Extension 7: AI Data Analytics Agent** — Integrated Gemini 2.5 Flash with 9-tool automated function calling.

## Demo Video Link
- [📺 Watch the Project Demo on YouTube](https://youtu.be/aC4HItQJ1aM)

## Notes / Bottlenecks

### Infrastructure & Security
- **ACCOUNTADMIN Required**: The `STORAGE INTEGRATION` step in `06_s3_pipeline.sql` requires `ACCOUNTADMIN` privileges (one-time setup). AWS IAM roles must manually trust the Snowflake external ID found via `DESCRIBE INTEGRATION`.
- **Credential Management**: All Snowflake and AWS credentials are stored in `.env` (gitignored). The `.env.example` template is safe for public repositories. Never commit `.env` to GitHub.

### Data Ingestion
- **S3 Connectivity**: The pipeline assumes CSVs are under the `/data/` prefix of the S3 bucket. Metadata mismatches (e.g., wrong headers, encoding) will cause `COPY INTO` failures, which are logged in `pipeline_logs.csv`.
- **Data Freshness**: The derived tables in `05_derived_analytics.sql` are "on-demand" materialized. They require a manual re-run (or Snowflake Task) after new data loads to reflect updated rankings and scores.
- **Batch vs. Streaming**: The current pipeline uses batch `COPY INTO`. For real-time ingestion, Snowpipe with S3 event notifications would be needed.

### Dashboard Performance
- **Cold Starts**: The first query in a session may take 5–10 seconds if the Snowflake warehouse (`XSMALL`) is suspended. Subsequent cached queries (TTL: 2 min) are sub-second.
- **SQL Explorer Cap**: The Live SQL Explorer in the Executive tab caps results at 500 rows to prevent browser out-of-memory issues. Large-scale analytics should be pushed to Snowflake views rather than processed in Pandas.
- **Query Caching**: All `run_query()` calls are cached by Streamlit with a 120-second TTL. Adjusting filters without clicking "Run" again returns the cached result.

### Data Quality
- **Source Data**: The 14 CSV datasets in `data/` are sourced from the public Kaggle dataset [Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database) by *yogape*. No synthetic generation scripts are used.
- **Schema Validation**: Column types and constraints are enforced by the Snowflake DDL in `01_create_schema.sql`. CSV files with mismatched columns will fail at the `COPY INTO` stage.

### Logging & Monitoring
- **Query Audit Trail**: Every dashboard query is logged to `logs/pipeline_logs.csv` with timestamp, team, user, query name, latency, row count, and auto-generated performance notes.
- **Error Tracking**: Failed queries log the exception message in the `error` column. The Monitoring tab displays error rate as a KPI card.

### Deployment
- **Streamlit Cloud**: The app is deployed at [cs5542lab5.streamlit.app](https://cs5542lab5.streamlit.app/). Snowflake credentials must be configured in the Streamlit Cloud secrets manager for cloud deployment.
- **Local Development**: Run `streamlit run app/streamlit_app.py` with a valid `.env` file for local testing.

---
