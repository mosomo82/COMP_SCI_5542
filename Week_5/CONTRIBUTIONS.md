# CONTRIBUTIONS.md — CS 5542 Lab 5

> **Team Name:** EVN 
> **Team Members:** Tony Nguyen (mosomo82), Daniel Evans (devans2718), Joe Vinas (jvinas)
> **Project Title:** Logistics Operation Dashboard — Snowflake Integration

> **Deadline:** Feb. 24, 2026 — 11:59 PM

---

## Team Members & Responsibilities

### Member 1: Tony Nguyen (mosomo82)
- Responsibilities:
  - Designed and implemented the Snowflake schema (`sql/01_create_schema.sql`) — 14 tables for the trucking logistics domain
  - Created the staging and loading SQL (`sql/02_stage_and_load.sql`) — warehouse config, file format, internal stage, and `COPY INTO`
  - Built the automated pipeline orchestrator (`scripts/run_pipeline.py`) — one-command S3 and local ingestion
  - Developed the Snowflake connection module (`scripts/sf_connect.py`) — secure credential handling with `.env`
  - Set up project environment (`.env.example`, `.gitignore`, `requirements.txt`)
- Secondary Support:
  - Assisted with dashboard feature integration and debugging
  - Managed security configurations and credential templates
- Evidence (PR/commits):
  - Commit `6458f57` — Initial Week 5 project setup
  - Commit `fd3643d` — Update `sf_connect.py` with authentication improvements
  - Commit `6f31f63` — Update `streamlit_app.py` with dashboard features
  - Commit `e33559c` — Update pipeline scripts
  - Commit `141aefd` — Update ingestion and schema scripts
  - Commit `f8e17a1` — Update pipeline orchestration
  - Commits `02bd4e2`, `fd2dbab` — Update `.gitignore` and README documentation
- Tested:
  - Verified all 14 tables are created successfully in Snowflake (`01_create_schema.sql`)
  - Tested `COPY INTO` staging pipeline — confirmed CSV data loads with correct row counts
  - Ran `scripts/run_pipeline.py` end-to-end (both `--local` and S3 modes) — validated logging output and error handling
  - Confirmed `sf_connect.py` establishes a valid Snowflake connection with environment variables

---

### Member 2: Daniel Evans (devans2718)
- Responsibilities:
  - Wrote core analytical queries (`sql/03_queries.sql`) — 5 queries for Revenue, Fuel Efficiency, Route Profitability, Monthly Trends, and Fleet Utilization
  - Created derived views (`sql/04_views.sql`) — 5 views powering dashboard visualizations
  - Built the Safety Incidents dashboard tab in the Streamlit app
  - Set up the initial project structure (README, CONTRIBUTIONS.md, `.env.example`)
  - Developed the Streamlit **Overview** and **Fleet & Drivers** tabs (KPI cards, monthly revenue chart, truck/driver performance)
- Secondary Support:
  - Coordinated overall project documentation and repository alignment
  - Assisted in initial environment setup and scaffolding
- Evidence (PR/commits):
  - Commit `d64aad7` — feat: Add initial project structure including README, CONTRIBUTIONS.md, `.env.example`, and a new Safety Incidents dashboard tab
  - Branch `daniel` — Dedicated feature branch for Safety Incidents tab and project scaffolding
  - Pull Request: `daniel` → `main` branch merge for Safety Incidents feature
- Tested:
  - Validated all 5 analytical queries in `03_queries.sql` against loaded Snowflake data — confirmed correct aggregations and joins
  - Tested 5 derived views in `04_views.sql` — verified `SELECT *` returns expected columns and row counts
  - Verified the Overview tab renders KPI cards and monthly revenue line chart with date-range filtering
  - Tested the Fleet & Drivers tab with fuel-type multi-select and min-trips slider — confirmed correct data filtering

---

### Member 3: Joe Vinas
- Responsibilities:
  - Built the advanced derived analytics (`sql/05_derived_analytics.sql`) — 4 materialized analytics tables (Driver Rankings, Truck Health, Route Quality, and more)
  - Designed the AWS S3 integration pipeline (`sql/06_s3_pipeline.sql`) — storage integration, external stage setup, and `COPY INTO` from S3
  - Developed the batch CSV loader (`scripts/load_local_csv_to_stage.py`) — single-table and `--batch` mode loading
  - Built the Streamlit **Routes**, **Fuel Spend**, **Monitoring**, and **Executive** tabs
  - Prepared and validated all 14 synthetic logistics datasets in `data/`
- Secondary Support:
  - Assisted in synthetic data ingestion and schema validation
  - Provided support for automated pipe orchestration testing
- Evidence (PR/commits):
  - Commit `b04fca2` — Add files via upload (data CSVs and scripts)
  - Commit `ccbcf95` — Add Week 5 requirements to `requirements.txt`
  - Commit `30b7b19` — Delete exposed `.env` file for security
  - Commit `a56842b` — Create `.gitignore` for project dependencies and artifacts
  - Commits `4fdcbaa`, `0a836ac`, `619fdb0`, `fa0b139` — Clean up directory structure and data files
  - Commits `486696f`, `e6508f4`, `23ea9cd` — Update README with results, improved clarity, and documentation
- Tested:
  - Verified all 4 derived analytics tables in `05_derived_analytics.sql` — confirmed correct materialized aggregations
  - Tested S3 integration pipeline (`06_s3_pipeline.sql`) — validated external stage and `COPY INTO` from S3 bucket
  - Ran `scripts/load_local_csv_to_stage.py --batch` — confirmed all 14 CSVs load successfully with correct row counts
  - Tested Routes tab with margin threshold and min-loads filters — verified scorecard rendering
  - Tested Fuel Spend tab with state filter and top-N slider — confirmed chart accuracy
  - Validated Monitoring tab — confirmed latency charts and performance stats display correctly
  - Tested Executive tab — auto-loading KPIs, terminal heatmap, and live SQL explorer all functional

## Team Components: 

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
| Data source attribution — cited original Kaggle dataset, removed incorrect `generate_data.py` reference from README | `README.md` |
| Credential template for safe GitHub commits | `.env.example` |
| ⚠️ Safety Incidents dashboard tab — new 8th tab querying `SAFETY_INCIDENTS` with KPI cards, incident-type breakdown chart, top-10 driver incident chart, parameterized filters | `app/streamlit_app.py` |
| Team contribution documentation | `CONTRIBUTIONS.md` |
| Full dataset ingestion (14 tables including 5 extension tables) | `sql/01_create_schema.sql`, `sql/02_stage_and_load.sql` |
| Pipeline monitoring tab with latency charts | `app/streamlit_app.py` (Tab 5) |
| Advanced derived analytics (4 materialized tables) | `sql/05_derived_analytics.sql`, `app/streamlit_app.py` (Tab 6) |
| Automated S3 ingestion pipeline (one-command orchestration) | `scripts/run_pipeline.py`, `sql/06_s3_pipeline.sql` |
| Interactive Executive Dashboard (auto-load KPIs, heatmap, SQL explorer) | `app/streamlit_app.py` (Tab 7) |
| Safety Incidents dashboard tab (cross-dataset incident analytics) | `app/streamlit_app.py` (Tab 8) | 

## Division of Labor Summary

| Name | Role | Primary Contributions | Secondary Support |
| :--- | :--- | :--- | :--- |
| **Tony Nguyen** | Infrastructure & Integration | Snowflake Schema, Automated Ingestion Pipeline, Environment Setup. | Dashboard Feature Integration, Security & Credentials management. |
| **Daniel Evans** | Analytics & UI | Analytical Queries, Dashboard Views, Overview & Safety Tabs. | Project Documentation, Initial Structure setup. |
| **Joe Vinas** | Advanced Features & S3 | Materialized Analytics, S3 Pipeline, Logistics & Executive Tabs. | Synthetic Data Ingestion & Validation. |

### Extensions Labor Breakdown

| Extension | Primary Contributor | Secondary Support |
| :--- | :--- | :--- |
| **Ext 1: Full dataset ingestion** | Tony Nguyen | Joe Vinas |
| **Ext 2: Pipeline monitoring** | Joe Vinas | Tony Nguyen |
| **Ext 3: Advanced derived analytics** | Joe Vinas | Daniel Evans |
| **Ext 4: Automated S3 ingestion pipeline** | Tony Nguyen | Daniel Evans |
| **Ext 5: Interactive executive dashboard** | Daniel Evans | Joe Vinas |
| **Ext 6: Safety Incidents dashboard tab** | Daniel Evans | Tony Nguyen |


## Technical Reflection: Tony Nguyen

Working on Lab 5 provided deep exposure to the full-stack data engineering lifecycle, from cloud infrastructure setup to automated Python orchestration.

- **Snowflake-Native Development:** I gained hands-on experience designing a 14-table schema for the logistics domain. I learned the critical value of separating storage from compute and how to optimize warehouse usage (using X-Small for cost-efficiency) while maintaining performance.
- **AWS S3 Integration:** Setting up the S3 bucket and IAM policies taught me the intricacies of cloud security. The most rewarding part was configuring the `STORAGE INTEGRATION` to allow Snowflake to securely access S3 without exposing long-term credentials.
- **Data Pipelines:** Building the `run_pipeline.py` orchestrator helped me understand how to handle metadata-driven ingestion. I learned that for large datasets, using `COPY INTO` with partitioned S3 prefixes is significantly more robust than row-by-row inserts, especially regarding cost and throughput limitations.
- **Challenges:** The primary challenge was the "plumbing"—ensuring the Python connector, Snowflake schema, and S3 external stages all aligned securely. Managing authentication across local and cloud environments required careful credential handling.

**Production-Scale Roadmap:**
To evolve this project into a production-grade system, I would implement:
1. **Snowpipe:** Transition from batch runs to continuous auto-ingestion using Snowpipe for real-time availability.
2. **dbt (Data Build Tool):** Integrate dbt to add automated data quality tests (null checks, referential integrity) post-ingestion.
3. **Advanced Orchestration:** Use Apache Airflow or Snowflake Tasks to schedule dependencies between ingestion and materialized analytics.
4. **Monitoring & Alerting:** Implement real-time notifications for pipeline failures or latency spikes.

## Notes

- The 14 CSV datasets in `data/` are sourced from the public Kaggle dataset [Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database) by *yogape*.
- Snowflake credentials are stored in `.env` (gitignored). The `.env.example` template is safe for public repositories.
- The pipeline has been tested on XSMALL warehouse. Cold starts on a suspended warehouse may add 5–10 s to the first query.