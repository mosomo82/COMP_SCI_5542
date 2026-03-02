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

## Lab 6: Multi-Agent Analytics (In Progress - March 2, 2026)

### Member 1: Tony Nguyen (mosomo82) — Backend & AI Lead
- **Responsibilities:**
  - **Core Infrastructure (Phase 1):** Built the toolset foundation in `tools.py` (Tools 1-5) and the core agent logic in `agent.py`.
  - **System Integration:** Authored the `task1_antigravity_report.md` regarding automated assistance from Antigravity.
  - **Streamlit Session Integration (Phase 3):** Wired the AI Agent into the dashboard. Developed the robust `st.session_state` management system and dynamic tool registration logic that enables multi-member parallel development.
- **Evidence (Lab 6 Commits):**
  - Commit [`68fcf89`](https://github.com/mosomo82/COMP_SCI_5542/commit/68fcf89) — docs: add Lab 6 contribution addendum for Tony.
  - Commit [`4458fd7`](https://github.com/mosomo82/COMP_SCI_5542/commit/4458fd7) — feat(Tony): agent session setup with all 9 tools, chat lifecycle, tool logging.
- **Verification:**
  - Confirmed agent binds to all active tools and maintains multi-turn conversation history.
  - Validated tool-usage transparency via the sidebar and message expanders.

### Member 2: Daniel Evans (devans2718)
- **Responsibilities:**
  - **Tool Development (Phase 2):** Implementing `get_route_profitability` and `get_delivery_performance`.
  - **UI/UX Polish:** Working on chat message formatting and history visualization.
  - **Evaluation (Phase 4):** Writing the `task4_evaluation_report.md` and recording the demo video.
- **Evidence (Lab 6 Commits):**
  - Commit [`70771e1`](https://github.com/mosomo82/COMP_SCI_5542/commit/70771e1) — fix: add missing Week_5/requirements.txt for Streamlit Cloud.
  - Commit [`f2d7f83`](https://github.com/mosomo82/COMP_SCI_5542/commit/f2d7f83) — remove duplicated lines in streamlit_app.py.

### Member 3: Joel Vinas (joelvinas)
- **Responsibilities:**
  - **Tool Development (Phase 2):** Implementing `get_maintenance_health` and `get_fuel_spend_analysis`.
  - **Logging & Formatting:** Enhancing the AI response formatting and tool execution logging.
  - **Final Documentation:** Updating README and screenshots for the final submission.
- **Evidence (Lab 6 Commits):**
  - Commit [`28561ac`](https://github.com/mosomo82/COMP_SCI_5542/commit/28561ac) — Merge pull request #4 from mosomo82/jvinas.
  - Commit [`b4eb440`](https://github.com/mosomo82/COMP_SCI_5542/commit/b4eb440) — Updated to reintroduce sql for derived analytics & s3 pipeline.

---

## Division of Labor Summary (Lab 6)

| Member | Roles | Tools | Deliverables |
|---|---|---|---|
| **Tony** | AI Infrastructure & Integration | Tools 1-5 (Core) | `agent.py`, `tools.py`, `task1_report.md` |
| **Daniel** | UI & Evaluation | Tools 6-7 (Analytical) | `task4_report.md`, Demo Video |
| **Joel** | Documentation & Logging | Tools 8-9 (Operational) | `README.md`, Screenshots |

---

## Project Component Ownership

| Segment | Primary Owner |
|---|---|
| Snowflake DB & Staging | Tony Nguyen |
| Data Ingestion Pipelines | Tony Nguyen / Joel Vinas |
| Core Analytics & Views | Daniel Evans |
| Dashboard UI & Monitoring | Joel Vinas / Daniel Evans |
| Advanced Analytics (Derived) | Joel Vinas |
| AI Agent & Tool Binding | Tony Nguyen |
| AI Tool Development | All Members (Team) |
