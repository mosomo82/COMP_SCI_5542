# 🚚 CS 5542 — Lab 6: Multi-Agent Analytics for Trucking Logistics

*An AI-powered analytics platform: Data → Snowflake → Gemini Agent → Streamlit Dashboard*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](#)
[![Snowflake](https://img.shields.io/badge/Snowflake-Data_Warehouse-29B5E8.svg?style=for-the-badge&logo=snowflake&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Gemini](https://img.shields.io/badge/Gemini_2.5-AI_Agent-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)](#)

---

## 🔗 Live Deployment

> **Streamlit Cloud:** [🚀 Launch Lab6_Streamlit_Cloud](https://cs5542logisticsai.streamlit.app/)

---

## 🏗️ Pipeline Architecture & End-to-End Flow

![logistics_pipeline_architecture_1771661265221](https://github.com/user-attachments/assets/972aed5f-27d4-442e-bce0-eb888fb42cd1)

---

## 🎯 Lab 6 Scope — Multi-Agent Analytics

Lab 6 extends the Lab 5 Snowflake dashboard with a **Gemini 2.5 Flash AI Agent** capable of autonomous data retrieval, multi-step reasoning, and natural-language synthesis across the entire logistics domain.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 1** | AI Agent infrastructure (`agent.py`, Tools 1–5, `sf_connect.py`) | ✅ Complete |
| **Phase 2** | Extended tool suite (Tools 6–9: routes, delivery, maintenance, fuel) | ✅ Complete |
| **Phase 3** | Streamlit integration (Agent Chat tab, session management, tool logging) | ✅ Complete |
| **Phase 4** | Evaluation report (`eval_scenarios.py`, `task4_evaluation_report.md`) | ✅ Complete |

---

## 🤖 AI Data Analytics Agent

The dashboard features an autonomous **AI Data Analytics Agent** powered by Google Gemini 2.5 Flash. The agent reasons across 9 specialized tools to answer complex operational questions using automatic function calling.

### Agent Tools

| # | Tool | Description | Data Source |
|---|------|-------------|-------------|
| 1 | `query_snowflake` | Arbitrary read-only SQL queries | Any Snowflake table |
| 2 | `get_monthly_revenue` | Aggregated revenue trends | `V_MONTHLY_REVENUE` |
| 3 | `get_fleet_performance` | Truck metrics (trips, miles, MPG, revenue) | `V_TRIP_PERFORMANCE` |
| 4 | `get_pipeline_logs` | System health and ingestion latency | `logs/pipeline_logs.csv` |
| 5 | `get_safety_metrics` | Driver incident analytics and claims | `SAFETY_INCIDENTS` + `DRIVERS` |
| 6 | `get_route_profitability` | Margin analysis by route | `V_ROUTE_SCORECARD` |
| 7 | `get_delivery_performance` | On-time rates and detention times | `DELIVERY_EVENTS` |
| 8 | `get_maintenance_health` | Maintenance costs, labor, and downtime | `MAINTENANCE_RECORDS` + `TRUCKS` |
| 9 | `get_fuel_spend_analysis` | Regional fuel spend breakdown | `FUEL_PURCHASES` |

### Agent Capabilities

- **Single-tool queries:** "Show me the monthly revenue from January to June 2024."
- **Multi-tool reasoning:** "Which top diesel trucks have safety incidents on record?"
- **Cross-domain synthesis:** "Compare our most profitable routes against delivery reliability and recommend corrective actions."

### Running the Agent (CLI)

```bash
py agent.py
```

---

## 🗂️ Project Structure

```text
Week_6/
├── agent.py                    # Gemini 2.5 Flash agent with 9-tool automatic function calling
├── tools.py                    # 9 specialized data tools (Snowflake queries + CSV parsing)
├── tool_schemas.py             # OpenAI-format JSON schemas for all tools
├── eval_scenarios.py           # Task 4 evaluation harness (5 scenarios, retry logic)
├── eval_results.json           # Raw evaluation results from eval_scenarios.py
├── task4_evaluation_report.md  # Evaluation report (5 scenarios, accuracy, latency)
├── task1_antigravity_report.md # AI assistance documentation
├── test_agent.py               # Agent tool-binding smoke test
├── test_tools.py               # Individual tool integration tests
├── app/
│   └── streamlit_app.py        # 9-tab Streamlit dashboard with AI Agent Chat
├── data/                       # 14 trucking logistics CSVs
│   ├── customers.csv
│   ├── drivers.csv
│   ├── trucks.csv
│   ├── routes.csv
│   ├── loads.csv
│   ├── trips.csv
│   ├── fuel_purchases.csv
│   ├── trailers.csv
│   ├── facilities.csv
│   ├── delivery_events.csv
│   ├── maintenance_records.csv
│   ├── safety_incidents.csv
│   ├── driver_monthly_metrics.csv
│   └── truck_utilization_metrics.csv
├── logs/
│   └── pipeline_logs.csv       # Query audit trail (timestamp, latency, errors)
├── reports/
│   ├── CONTRIBUTIONS_TN.md     # Tony Nguyen — individual report
│   ├── CONTRIBUTIONS_DE.md     # Daniel Evans — individual report
│   └── CONTRIBUTIONS_JV.md     # Joel Vinas — individual report
├── scripts/
│   ├── sf_connect.py           # Centralized Snowflake connection module
│   └── run_pipeline.py         # Automated ingestion orchestrator (Local & S3)
├── sql/
│   ├── 01_create_schema.sql    # DDL for 14 tables
│   ├── 02_stage_and_load.sql   # Internal staging + COPY INTO
│   ├── 03_queries.sql          # 5 core analytical queries
│   ├── 04_views.sql            # 5 derived views for dashboard
│   ├── 05_derived_analytics.sql# 4 materialized analytics tables
│   └── 06_s3_pipeline.sql      # AWS S3 external stage setup
├── .env.example                # Credential template (Snowflake + Gemini + AWS)
├── CONTRIBUTIONS.md            # Team contribution log
├── README.md                   # This file
└── requirements.txt            # Python dependencies
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

### 1. Configure Environment

```bash
cp .env.example .env
```

Required variables in `.env`:
```
GEMINI_API_KEY=your_gemini_api_key
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=CS5542_WEEK5
SNOWFLAKE_SCHEMA=PUBLIC
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Snowflake Setup (first time only)

Execute the SQL files in order in Snowflake:
```
sql/01_create_schema.sql → 02_stage_and_load.sql → 03_queries.sql → 04_views.sql → 05_derived_analytics.sql
```

Or use the automated orchestrator:
```bash
py scripts/run_pipeline.py
```

---

## 📈 Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Dashboard Layout

| Tab | Description |
|-----|-------------|
| 📊 Overview | KPI cards + monthly revenue line chart |
| 🚛 Fleet & Drivers | Truck/driver performance with filtering |
| 🗺️ Routes | Route scorecard and margin thresholds |
| ⛽ Fuel Spend | Fuel spend analysis by state/city |
| 📈 Monitoring | Pipeline performance stats and latency charts |
| 🔬 Analytics | Advanced materialized analytics tables |
| 🎯 Executive | Auto-loading KPIs, heatmap, and live SQL explorer |
| ⚠️ Safety | Driver hazard analysis and claim cost charts |
| 🤖 Agent Chat | **Interactive AI chat** powered by Gemini 2.5 Flash with 9 tools |

---

## 🧪 Agent Evaluation (Task 4)

### Running the Evaluation

```bash
py eval_scenarios.py --json
```

### Results Summary

| Scenario | Complexity | Tools Expected | Result | Latency |
|----------|-----------|----------------|--------|---------|
| S1 | Simple | `get_monthly_revenue` | ✅ PASS | 7.1 s |
| S2 | Medium | `get_fleet_performance`, `get_safety_metrics` | ⚠️ Partial | 16.4 s |
| S3 | Complex | `get_route_profitability`, `get_delivery_performance` | ✅ PASS | 27.7 s |
| S4 | Medium | `get_maintenance_health` | ✅ PASS | 8.5 s |
| S5 | Complex | `get_fuel_spend_analysis`, `get_maintenance_health` | ✅ PASS | 19.2 s |

**Pass rate:** 4/5 (80%) · **Tool accuracy:** 6/7 (86%)

See [`task4_evaluation_report.md`](task4_evaluation_report.md) for full analysis.

---

## 🏆 Extensions Completed

| # | Extension | Description |
|---|-----------|-------------|
| 1 | Full dataset ingestion | Ingested all 14 trucking CSVs into Snowflake |
| 2 | Pipeline monitoring | Dedicated Monitoring tab with latency charts |
| 3 | Advanced derived analytics | 4 materialized tables for rankings and scores |
| 4 | Automated S3 ingestion pipeline | One-command orchestration via `run_pipeline.py` |
| 5 | Interactive executive dashboard | Auto-loading KPIs, heatmap, live SQL explorer |
| 6 | Safety Incidents dashboard tab | Incident analytics with KPI cards and charts |
| 7 | AI Data Analytics Agent | Gemini 2.5 Flash with 9-tool automatic function calling |

---

## 📺 Demo Video

- [Watch the Project Demo on YouTube](https://youtu.be/GfhWJHOUlaU)

---

## 👥 Team

| Member | GitHub | Role (Lab 6) |
|--------|--------|--------------|
| **Tony Nguyen** | `mosomo82` | AI Infrastructure & Integration (Tools 1–5, Agent, Session Wiring) |
| **Daniel Evans** | `devans2718` | Chat UI & Evaluation (Tools 6–7, Eval Report) |
| **Joel Vinas** | `joelvinas` | Documentation & Logging (Tools 8–9, README, Screenshots) |

See individual contribution reports in [`reports/`](reports/).

---

## ⚠️ Notes & Bottlenecks

### AI Agent
- **API Rate Limits:** The Gemini API has per-minute request quotas. The evaluation harness includes 15-second cooldowns and retry logic to handle 429 errors.
- **Response Latency:** Agent responses range from 7–28 seconds depending on the number of tool calls and synthesis complexity.
- **Tool Selection Accuracy:** The agent occasionally skips expected tool calls when it believes the data is insufficient for cross-referencing (observed in S2 evaluation).

### Infrastructure & Security
- **ACCOUNTADMIN Required:** The `STORAGE INTEGRATION` step in `06_s3_pipeline.sql` requires `ACCOUNTADMIN` privileges (one-time setup).
- **Credential Management:** All Snowflake, AWS, and Gemini credentials are stored in `.env` (gitignored). Never commit `.env` to GitHub.

### Data Ingestion
- **S3 Connectivity:** The pipeline assumes CSVs are under the `/data/` prefix of the S3 bucket.
- **Data Freshness:** Derived tables in `05_derived_analytics.sql` require manual re-run after new data loads.
- **Batch vs. Streaming:** Current pipeline uses batch `COPY INTO`. For real-time ingestion, Snowpipe with S3 event notifications would be needed.

### Dashboard Performance
- **Cold Starts:** First query may take 5–10 s if the Snowflake warehouse is suspended. Cached queries (TTL: 2 min) are sub-second.
- **SQL Explorer:** The Executive tab caps results at 500 rows to prevent browser OOM.
- **Query Caching:** All `run_query()` calls are cached by Streamlit with a 120-second TTL.

### Data Quality
- **Source Data:** The 14 CSV datasets are sourced from the public Kaggle dataset [Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database) by *yogape*.
- **Schema Validation:** Column types and constraints are enforced by Snowflake DDL in `01_create_schema.sql`.

### Logging & Monitoring
- **Query Audit Trail:** Every dashboard query is logged to `logs/pipeline_logs.csv` with timestamp, team, query name, latency, row count, and performance notes.
- **Error Tracking:** Failed queries log the exception message. The Monitoring tab displays error rate as a KPI card.

### Deployment

- **Local Development:** Run `streamlit run app/streamlit_app.py` with a valid `.env` file.

---
