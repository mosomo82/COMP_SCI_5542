# CONTRIBUTIONS.md — Team EVN

> **Team Name:** EVN 
> **Team Members:** Tony Nguyen (mosomo82), Daniel Evans (devans2718), Joel Vinas (joelvinas)
> **Project Title:** Logistics Operation Dashboard — Snowflake & AI Integration

---

## Lab 5: Snowflake Dashboard & Pipeline (Completed Feb 24, 2026)

### Member 1: Tony Nguyen (mosomo82)
- **Responsibilities:**
  - Designed and implemented the Snowflake schema (`sql/01_create_schema.sql`) — 14 tables for the trucking logistics domain.
  - Created the staging and loading SQL (`sql/02_stage_and_load.sql`) — warehouse config, file format, internal stage, and `COPY INTO`.
  - Built the automated pipeline orchestrator (`scripts/run_pipeline.py`) — one-command S3 and local ingestion.
  - Developed the Snowflake connection module (`scripts/sf_connect.py`) — secure credential handling with `.env`.
  - Set up project environment (`.env.example`, `.gitignore`, `requirements.txt`).
- **Evidence (Commits):** `163b40b`, `b04fca2`, `fd3643d`, `6f31f63`.

### Member 2: Daniel Evans (devans2718)
- **Responsibilities:**
  - Wrote core analytical queries (`sql/03_queries.sql`) and derived views (`sql/04_views.sql`).
  - Built the Streamlit **Overview**, **Fleet & Drivers**, and **Safety Incidents** tabs.
  - Set up the initial project structure and baseline documentation.
- **Evidence (Commits):** `d64aad7`, `f2c2ea1`, `f919986`, `6d80a0a`.

### Member 3: Joel Vinas (joelvinas)
- **Responsibilities:**
  - Built the advanced derived analytics (`sql/05_derived_analytics.sql`) and S3 integration pipeline (`sql/06_s3_pipeline.sql`).
  - Developed the batch CSV loader (`scripts/load_local_csv_to_stage.py`).
  - Built the Streamlit **Routes**, **Fuel Spend**, **Monitoring**, and **Executive** tabs.
  - Prepared all 14 synthetic logistics datasets in `data/`.
- **Evidence (Commits):** `3b93508`, `bfee410`, `28561ac`.

---

## Project Component Ownership

| Segment | Files |
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

---

# ADDENDUM: Lab 6 — Multi-Agent Analytics

> **Lab 6 Deadline:** March 3, 2026 — 11:59 PM

## Lab 6 Contributions (Final — March 3, 2026)

### Member 1: Tony Nguyen (mosomo82) — Backend & AI Lead
- **Responsibilities:**
  - **Phase 1 (Infrastructure):** Built the toolset foundation in `tools.py` (Tools 1–5) and the core agent logic in `agent.py`.
  - **Phase 1 (Documentation):** Authored the `task1_antigravity_report.md` regarding automated assistance from Antigravity.
  - **Phase 3 (Streamlit Integration):** Wired the AI Agent into the dashboard. Developed the `st.session_state` management system and dynamic tool registration logic that enables multi-member parallel development.
- **Evidence (Lab 6 Commits & PRs):**
  - **PR [#6](https://github.com/mosomo82/COMP_SCI_5542/pull/6)** — `tony/lab6-agent-integration` → `main` (merged at `86504e5`)
  - Commit [`4458fd7`](https://github.com/mosomo82/COMP_SCI_5542/commit/4458fd7) — feat(Tony): agent session setup with all 9 tools, chat lifecycle, tool logging.
  - Commit [`68fcf89`](https://github.com/mosomo82/COMP_SCI_5542/commit/68fcf89) — docs: add Lab 6 contribution addendum for Tony.
  - Commit [`4f14cd4`](https://github.com/mosomo82/COMP_SCI_5542/commit/4f14cd4) — Create implementation_plan.md.
  - Commit [`21ee827`](https://github.com/mosomo82/COMP_SCI_5542/commit/21ee827) — Update implementation_plan.md.
  - Commit [`a02710b`](https://github.com/mosomo82/COMP_SCI_5542/commit/a02710b) — Update implementation_plan.md.
- **Verification:**
  - Confirmed agent binds to all 9 active tools and maintains multi-turn conversation history.
  - Validated tool-usage transparency via the sidebar and message expanders.

### Member 2: Daniel Evans (devans2718) — Tools & Evaluation Lead
- **Responsibilities:**
  - **Phase 2 (Tool Development):** Implemented `get_route_profitability` (Tool 6) and `get_delivery_performance` (Tool 7) in `tools.py` and `tool_schemas.py`.
  - **Phase 3 (Chat UI):** Built the Streamlit chat input loop, message history, display loop, and loading spinner.
  - **Phase 4 (Evaluation):** Authored the `task4_evaluation_report.md` (5-scenario evaluation) and `eval_scenarios.py` harness. Authored `CONTRIBUTIONS_DE.md`.
- **Evidence (Lab 6 Commits & PRs):**
  - **PR [#7](https://github.com/mosomo82/COMP_SCI_5542/pull/7)** — `daniel-lab6` → `main` (merged at `3a4e860`)
  - **PR [#5](https://github.com/mosomo82/COMP_SCI_5542/pull/5)** — `fix/week5-requirements` → `main` (merged at `6acf44b`)
  - Commit [`54dccd5`](https://github.com/mosomo82/COMP_SCI_5542/commit/54dccd5) — feat(daniel): add get_route_profitability and get_delivery_performance tools.
  - Commit [`923e986`](https://github.com/mosomo82/COMP_SCI_5542/commit/923e986) — docs: update implementation_plan.md to reflect Phase 2 completion.
  - Commit [`06d1b71`](https://github.com/mosomo82/COMP_SCI_5542/commit/06d1b71) — docs: Add individual contribution report for Lab 6.
  - Commit [`70771e1`](https://github.com/mosomo82/COMP_SCI_5542/commit/70771e1) — fix: add missing Week_5/requirements.txt for Streamlit Cloud.
  - Commit [`f2d7f83`](https://github.com/mosomo82/COMP_SCI_5542/commit/f2d7f83) — remove duplicated lines in streamlit_app.py.

### Member 3: Joel Vinas (joelvinas) — Tools & Documentation Lead
- **Responsibilities:**
  - **Phase 2 (Tool Development):** Implemented `get_maintenance_health` (Tool 8) and `get_fuel_spend_analysis` (Tool 9) in `tools.py` and `tool_schemas.py`.
  - **Phase 3 (Logging & Formatting):** Enhanced AI response formatting and tool execution logging.
  - **Phase 4 (Documentation):** Updated README, implementation plan, and screenshots for final submission. Authored `CONTRIBUTIONS_JV.md`.
- **Evidence (Lab 6 Commits & PRs):**
  - **PR [#8](https://github.com/mosomo82/COMP_SCI_5542/pull/8)** — `jvinas_20260303` → `main` (merged at `5febcc2`)
  - **PR [#9](https://github.com/mosomo82/COMP_SCI_5542/pull/9)** — `jvinas_20260303` → `main` (merged at `1e1e54f`)
  - Commit [`2c01629`](https://github.com/mosomo82/COMP_SCI_5542/commit/2c01629) — Updated code per implementation plan: agent.py, test_agent.py, test_tools.py, tool_schemas.py, tools.py. Added CONTRIBUTIONS_JV.
  - Commit [`0e801b1`](https://github.com/mosomo82/COMP_SCI_5542/commit/0e801b1) — Updated agent.py, test_agent.py, tool_schemas.py and implementation_plan.md.
  - Commit [`6c4fe75`](https://github.com/mosomo82/COMP_SCI_5542/commit/6c4fe75) — Updated implementation_plan.md, readme.md and task4_evaluation_report per implementation plan.
  - Commit [`eb13dc6`](https://github.com/mosomo82/COMP_SCI_5542/commit/eb13dc6) — Manually updated README.md to reference Week 6 and Contribution documents.
  - Commit [`cc996b3`](https://github.com/mosomo82/COMP_SCI_5542/commit/cc996b3) — Updated CONTRIBUTIONS_JV.md to reflect changes for Lab 6.
  - Commit [`36ee326`](https://github.com/mosomo82/COMP_SCI_5542/commit/36ee326) — updated contributions_jv.md to include GitHub link.

---

## Lab 6 Division of Labor Summary

| Member | Roles | Tools Owned | Primary Deliverables |
|---|---|---|---|
| **Tony Nguyen** | AI Infrastructure & Integration | Tools 1–5 (Core) | `agent.py`, `tools.py`, `task1_antigravity_report.md` |
| **Daniel Evans** | Tools, Chat UI & Evaluation | Tools 6–7 (Analytical) | `task4_evaluation_report.md`, `eval_scenarios.py`, Demo Video |
| **Joel Vinas** | Tools, Documentation & Logging | Tools 8–9 (Operational) | `README.md`, Screenshots, Implementation Plan updates |

## Lab 6 Project Component Ownership

| Segment | Owner(s) |
|---|---|
| Snowflake DB & Staging | Tony Nguyen |
| Data Ingestion Pipelines | Tony Nguyen / Joel Vinas |
| Core Analytics & Views | Daniel Evans |
| Dashboard UI & Monitoring | Joel Vinas / Daniel Evans |
| Advanced Analytics (Derived) | Joel Vinas |
| AI Agent Architecture (`agent.py`) | Tony Nguyen |
| Tools 1–5 (`query_snowflake`, `get_monthly_revenue`, `get_fleet_performance`, `get_pipeline_logs`, `get_safety_metrics`) | Tony Nguyen |
| Tools 6–7 (`get_route_profitability`, `get_delivery_performance`) | Daniel Evans |
| Tools 8–9 (`get_maintenance_health`, `get_fuel_spend_analysis`) | Joel Vinas |
| Streamlit Agent Chat Tab (Session Wiring) | Tony Nguyen |
| Streamlit Agent Chat Tab (Chat UI) | Daniel Evans |
| Streamlit Agent Chat Tab (Tool Logging) | Joel Vinas |
| Evaluation Harness & Report | Daniel Evans |
| Individual Reports | All Members |

---
