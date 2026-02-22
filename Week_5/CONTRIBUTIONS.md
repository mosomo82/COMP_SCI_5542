# CONTRIBUTIONS.md — CS 5542 Lab 5

> **Team:** CS 5542 Week 5 · Logistics Operations Dashboard  
> **Repository:** [COMP_SCI_5542 / Week_5](https://github.com/)  
> **Deadline:** Feb. 23, 2026 — 12:00 PM

---

## Team Members & Responsibilities

### Member 1 — Tony (Lead Engineer)

**Implemented components:**

| Component | Files |
|---|---|
| Snowflake environment setup (database, schema, warehouse, roles) | `sql/01_create_schema.sql` |
| Internal staging + `COPY INTO` ingestion | `sql/02_stage_and_load.sql` |
| 5 core analytical queries (revenue, efficiency, profitability, trends, fleet) | `sql/03_queries.sql` |
| 5 derived views for dashboard | `sql/04_views.sql` |
| 4 advanced materialized analytics tables (driver rankings, truck health, route quality, monthly ops) | `sql/05_derived_analytics.sql` |
| AWS S3 external stage + `COPY INTO` from S3 | `sql/06_s3_pipeline.sql` |
| Automated master orchestrator pipeline (local + S3 modes) | `scripts/run_pipeline.py` |
| Manual batch CSV loader via internal stage | `scripts/load_local_csv_to_stage.py` |
| Snowflake connection helper module | `scripts/sf_connect.py` |
| 7-tab Streamlit dashboard (Overview, Fleet & Drivers, Routes, Fuel Spend, Monitoring, Analytics, Executive) | `app/streamlit_app.py` |
| Pipeline logging (`pipeline_logs.csv`) | `logs/pipeline_logs.csv` |
| Primary README documentation | `README.md` |

---

### Member 2 — Daniel

**Implemented components:**

| Component | Files |
|---|---|
| Data source attribution — cited original Kaggle dataset, removed incorrect `generate_data.py` reference from README | `README.md` |
| Credential template for safe GitHub commits | `.env.example` |
| ⚠️ Safety Incidents dashboard tab — new 8th tab querying `SAFETY_INCIDENTS` with KPI cards, incident-type breakdown chart, top-10 driver incident chart, parameterized filters | `app/streamlit_app.py` |
| Team contribution documentation | `CONTRIBUTIONS.md` |

---

## Extensions Implemented

| Extension | Implemented By | Files |
|---|---|---|
| Full dataset ingestion (14 tables including 5 extension tables) | Tony | `sql/01_create_schema.sql`, `sql/02_stage_and_load.sql` |
| Pipeline monitoring tab with latency charts | Tony | `app/streamlit_app.py` (Tab 5) |
| Advanced derived analytics (4 materialized tables) | Tony | `sql/05_derived_analytics.sql`, `app/streamlit_app.py` (Tab 6) |
| Automated S3 ingestion pipeline (one-command orchestration) | Tony | `scripts/run_pipeline.py`, `sql/06_s3_pipeline.sql` |
| Interactive Executive Dashboard (auto-load KPIs, heatmap, SQL explorer) | Tony | `app/streamlit_app.py` (Tab 7) |
| Safety Incidents dashboard tab (cross-dataset incident analytics) | Daniel | `app/streamlit_app.py` (Tab 8) |

---

## Commit Evidence

> Each member's contributions are verifiable by filtering GitHub commits by author:
>
> - **Tony's commits:** schema design, SQL scripts, ingestion scripts, core dashboard tabs
> - **Daniel's commits:** `CONTRIBUTIONS.md`, `.env.example`, README data source section, Safety Incidents tab

---

## Notes

- The 14 CSV datasets in `data/` are sourced from the public Kaggle dataset [Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database) by *yogape*.
- Snowflake credentials are stored in `.env` (gitignored). The `.env.example` template is safe for public repositories.
- The pipeline has been tested on XSMALL warehouse. Cold starts on a suspended warehouse may add 5–10 s to the first query.
