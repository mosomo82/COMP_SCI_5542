# CONTRIBUTIONS.md — CS 5542 Lab 5

> **Team Name:** EVN 
> **Team Members:** Tony Nguyen (mosomo82), Daniel Evans (devans2718), Joel Vinas (joelvinas)
> **Project Title:** Logistics Operation Dashboard — Snowflake Integration

> **Deadline:** Feb. 24, 2026 — 11:59 PM

---

## Team Members & Responsibilities

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

## Division of Labor Summary

| Name | Role | Primary Contributions | Secondary Support |
| :--- | :--- | :--- | :--- |
| **Tony Nguyen** | Infrastructure & Integration | Snowflake Schema, Automated Ingestion Pipeline, Environment Setup. | Dashboard Feature Integration, Security & Credentials management. |
| **Daniel Evans** | Analytics & UI | Analytical Queries, Dashboard Views, Overview & Safety Tabs. | Project Documentation, Initial Structure setup. |
| **Joel Vinas** | Advanced Features & S3 | Materialized Analytics, S3 Pipeline, Logistics & Executive Tabs. | Synthetic Data Ingestion & Validation. |

## Technical Reflection: Joel Vinas
Working with this team on combining the previous work into a cohesive Lab 5 helped me to better understand how we intend to push the components we've built so far into an integrated final system.

- **Data Analysis** We found a number of datasets related to the logistical operations/issues. Ultimately, this was a poor use of time as my insights didn't provide much momentum towards the larger system.
- **AWS S3 Integration:** I've set up S3 integrations before, but never on an autonomous, unattended system via Python. By collaborating with the team, I was able to better understand how to complete the implementation.
- **Challenges:** I found my lack of skill in writing code to be an incredible pitfall. Although I was able to get LLMs to assist me with writing the code, each prompt response created new errors which required follow-up prompts to resolve.

---

# ADDENDUM: Lab 6 — Multi-Agent Analytics (Completed March 3, 2026)

### Member 3: Joel Vinas (joelvinas / JVinas_Work)
- **Responsibilities:**
  - **Tool Development (Phase 2):** Designed and implemented the operational tools `get_maintenance_health` and `get_fuel_spend_analysis` in `tools.py`.
  - **Tool Schema Definitions:** Authored the JSON schemas for the new tools in `tool_schemas.py`, ensuring proper integration with Gemini's function calling.
  - **AI Integration UI (Phase 3):** Developed the "🔧 Tool Usage" execution logs in the Streamlit `🤖 AI Assistant` tab, providing transparency for the agent's multi-step reasoning.
  - **Agent Formatting:** Enhanced the markdown formatting for AI responses, including KPI tables and performance notes.
  - **Final Documentation (Phase 4):** Led the overhaul of the project `README.md` to reflect the 9-tool agentic state and updated the final Lab 6 screenshots.
- **Evidence (Lab 6 Commits):**
  - Commit [`28561ac`](https://github.com/mosomo82/COMP_SCI_5542/commit/28561ac) — Merge pull request #4 from mosomo82/jvinas.
  - Commit [`b4eb440`](https://github.com/mosomo82/COMP_SCI_5542/commit/b4eb440) — Updated to reintroduce sql for derived analytics & s3 pipeline.
  - Commit [`bfee410`](https://github.com/mosomo82/COMP_SCI_5542/commit/bfee410) — Updated to reintroduce sql for derived analytics & s3 pipeline (Lab 6 prep).
- **Verification:**
  - Verified `get_maintenance_health` returns accurate cost breakdowns for trucks.
  - Tested `get_fuel_spend_analysis` with state-level filtering across Snowflake datasets.
  - Confirmed the Streamlit "Tool Usage" expanders correctly display tool arguments and raw response data during agent sessions.
  - Validated that the final `README.md` and `implementation_plan.md` correctly reflect the 9-tool synchronized state.

### Technical Reflection: Lab 6
During Lab 6, I focused on making the AI's internal processes visible to the end-user. Building the tool execution logs in Streamlit was particularly rewarding as it bridged the gap between raw Python logic and a user-friendly analytical experience. Implementing the maintenance and fuel tools also allowed me to apply the complex SQL joining patterns I developed in Lab 5 to a new, agentic context.