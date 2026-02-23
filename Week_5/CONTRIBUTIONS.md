# Week 5 Contributions (Required)

## Team Name: Team mosomo82
## Project Title: Logistics Operation Dashboard — Snowflake Integration

---

### Member 1: Tony (mosomo82)
- Responsibilities:
  - Designed and implemented the Snowflake schema (`sql/01_create_schema.sql`) — 14 tables for the trucking logistics domain
  - Created the staging and loading SQL (`sql/02_stage_and_load.sql`) — warehouse config, file format, internal stage, and `COPY INTO`
  - Built the automated pipeline orchestrator (`scripts/run_pipeline.py`) — one-command S3 and local ingestion
  - Developed the Snowflake connection module (`scripts/sf_connect.py`) — secure credential handling with `.env`
  - Set up project environment (`.env.example`, `.gitignore`, `requirements.txt`)
- Evidence (PR/commits):
  - Commit `6458f57` — Initial Week 5 project setup
  - Commit `fd3643d` — Update `sf_connect.py` with authentication improvements
  - Commit `6f31f63` — Update `streamlit_app.py` with dashboard features
  - Commit `e33559c` — Update pipeline scripts
  - Commit `141aefd` — Update ingestion and schema scripts
  - Commit `f8e17a1` — Update pipeline orchestration
  - Merge commits `9b03174`, `6756d9c` — Merge branch integrations
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
