# CONTRIBUTIONS.md — CS 5542 Lab 5

> **Team Name:** EVN 
> **Team Members:** Tony Nguyen (mosomo82), Daniel Evans (devans2718), Joel Vinas (joelvinas)
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
  - Commit [`163b40b`](https://github.com/mosomo82/COMP_SCI_5542/commit/163b40b216b4ba1577fc308e1be81fae0b41f7b8) — Lab 5 Implementation (initial full project upload)
  - Commit [`b04fca2`](https://github.com/mosomo82/COMP_SCI_5542/commit/b04fca2c325ff68f94961c3681a7bbe0d7a45ec6) — Add files via upload (data CSVs and scripts)
  - Commit [`fd3643d`](https://github.com/mosomo82/COMP_SCI_5542/commit/fd3643d27ed95046a0286d434a129a475f116ed5) — Update `sf_connect.py` with authentication improvements
  - Commit [`6f31f63`](https://github.com/mosomo82/COMP_SCI_5542/commit/6f31f6320eba76e92d633cee3c2f662d0e503c6a) — Update `streamlit_app.py` with dashboard features
  - Commits [`519d5b4`](https://github.com/mosomo82/COMP_SCI_5542/commit/519d5b45b098472852f05f27c32667e123ca7cd3), [`7ee1f39`](https://github.com/mosomo82/COMP_SCI_5542/commit/7ee1f394e7a56da7135ff227c632cffbef285e40), [`9a1b022`](https://github.com/mosomo82/COMP_SCI_5542/commit/9a1b02220f873bebd6666191a9d08657ac913af4) — Update `streamlit_app.py` (iterative dashboard fixes)
  - Commit [`e33559c`](https://github.com/mosomo82/COMP_SCI_5542/commit/e33559c9711e4c336fe3b675e747e093583eb34c) — Update pipeline scripts
  - Commit [`141aefd`](https://github.com/mosomo82/COMP_SCI_5542/commit/141aefd1934ab418d6e39474f96ac83391865913) — Update ingestion and schema scripts
  - Commit [`02bd4e2`](https://github.com/mosomo82/COMP_SCI_5542/commit/02bd4e292bfe6374fc3215ae7e0419ab6d869f53) — Update `.gitignore`
  - Commits [`fd2dbab`](https://github.com/mosomo82/COMP_SCI_5542/commit/fd2dbabc076580c3b1c6cfaefe2cd5e8a5550b38), [`23ea9cd`](https://github.com/mosomo82/COMP_SCI_5542/commit/23ea9cd0ed9e0df8623bc526d818cf5743ffb906) — Update README documentation
    
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
  - Commit [`d64aad7`](https://github.com/mosomo82/COMP_SCI_5542/commit/d64aad7e839b2dcc102333a6c1294f8ec4a1ea81) — feat: Add initial project structure including README, CONTRIBUTIONS.md, `.env.example`, and a new Safety Incidents dashboard tab *(Feb 22)*
  - Commit [`f2c2ea1`](https://github.com/mosomo82/COMP_SCI_5542/commit/f2c2ea1f94e316c551dc200cdb453210cabe13ee) — feat: Add initial project structure including Safety Incidents dashboard tab *(branch: `daniel`)*
  - Commit [`f919986`](https://github.com/mosomo82/COMP_SCI_5542/commit/f9199865aa1565ca89a8e7acb3109f7d238f1a61) — Add SQL files for analytical queries and derived views for dashboard (`sql/03_queries.sql`, `sql/04_views.sql`)
  - Commit [`2af3d53`](https://github.com/mosomo82/COMP_SCI_5542/commit/2af3d533d51b4951456810ee1962b7c2b6ff110b) — Merge branch `daniel` (sync before PR)
  - [Pull Request #1](https://github.com/mosomo82/COMP_SCI_5542/pull/1): `mosomo82/daniel` → `main` — Safety Incidents tab & initial project structure; merged as commit [`6d80a0a`](https://github.com/mosomo82/COMP_SCI_5542/commit/6d80a0a6214d10f296e7eb68ab2627e49b8e1ec3)
  - [Pull Request #2](https://github.com/mosomo82/COMP_SCI_5542/pull/2): `mosomo82/add-sql-files-3-4` → `main` — Analytical queries & derived views SQL files; merged as commit [`f570678`](https://github.com/mosomo82/COMP_SCI_5542/commit/f570678a712cb5f846f0722215424bcbadbf8d4d)
- Tested:
  - Validated all 5 analytical queries in `03_queries.sql` against loaded Snowflake data — confirmed correct aggregations and joins
  - Tested 5 derived views in `04_views.sql` — verified `SELECT *` returns expected columns and row counts
  - Verified the Overview tab renders KPI cards and monthly revenue line chart with date-range filtering
  - Tested the Fleet & Drivers tab with fuel-type multi-select and min-trips slider — confirmed correct data filtering

---

### Member 3: Joel Vinas
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
  - Commit [`3b93508`](https://github.com/mosomo82/COMP_SCI_5542/commit/3b935089b181553227f9c1067a2eda777f0a039b) — Update Contributions.md to include JVinas
  - Commit [`bfee410`](https://github.com/mosomo82/COMP_SCI_5542/pull/4/commits/bfee4108d0833d5670908841a1cd7a3cbc10f811) — Updated to reintroduce sql for derived analytics & s3 pipeline
  - Commit [`b4eb440`](https://github.com/mosomo82/COMP_SCI_5542/pull/4/commits/b4eb440abc51369e448ab49d6df41c65fe8ed8c9) — Updated to reintroduce sql for derived analytics & s3 pipeline (final)
  - [Pull Request #3](https://github.com/mosomo82/COMP_SCI_5542/pull/3): `jvinas` → `main` — Update Contributions.md to include JVinas; merged as commit [`156336b`](https://github.com/mosomo82/COMP_SCI_5542/commit/156336b58b648ef76d2d1a9dd9e1b5ea4bd21559)
  - [Pull Request #4](https://github.com/mosomo82/COMP_SCI_5542/pull/4): `jvinas` → `main` — Merging SQL for derived analytics & S3 pipeline; merged as commit [`28561ac`](https://github.com/mosomo82/COMP_SCI_5542/commit/28561ac70a1f65481689aa7144dc97a797f39cb3)
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
| **Joel Vinas** | Advanced Features & S3 | Materialized Analytics, S3 Pipeline, Logistics & Executive Tabs. | Synthetic Data Ingestion & Validation. |

### Extensions Labor Breakdown

| Extension | Primary Contributor | Secondary Support |
| :--- | :--- | :--- |
| **Ext 1: Full dataset ingestion** | Tony Nguyen | Joel Vinas |
| **Ext 2: Pipeline monitoring** | Joel Vinas | Tony Nguyen |
| **Ext 3: Advanced derived analytics** | Joel Vinas | Daniel Evans |
| **Ext 4: Automated S3 ingestion pipeline** | Tony Nguyen | Daniel Evans |
| **Ext 5: Interactive executive dashboard** | Daniel Evans | Joel Vinas |
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