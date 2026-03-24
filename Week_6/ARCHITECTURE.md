# System Architecture: Multi-Agent Trucking Logistics Analytics Platform

> **CS 5542 — Big Data and Analytics**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Lab 9 Documentation Expansion (addressing Lab 6 feedback)

---

## 1. High-Level Overview

The platform is an end-to-end AI analytics system that ingests raw trucking logistics data, stores and queries it through Snowflake, exposes it to a Gemini 2.5 Flash AI agent via nine specialized tools, and presents results through a deployed nine-tab Streamlit dashboard.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                             │
│                                                                         │
│   14 CSV Files (Kaggle)                                                 │
│   customers · drivers · trucks · routes · loads · trips                 │
│   fuel_purchases · trailers · facilities · delivery_events              │
│   maintenance_records · safety_incidents · driver_monthly_metrics       │
│   truck_utilization_metrics                                             │
│         │                                                               │
│         ▼                                                               │
│   run_pipeline.py  ──(batch COPY INTO)──▶  Snowflake (CS5542_WEEK5)    │
│   [or S3 external stage via 06_s3_pipeline.sql]                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       SNOWFLAKE DATA LAYER                              │
│                                                                         │
│   14 base tables  ──▶  5 derived views  ──▶  4 materialized analytics  │
│                                                                         │
│   Views: V_MONTHLY_REVENUE, V_TRIP_PERFORMANCE, V_ROUTE_SCORECARD,     │
│           V_DRIVER_SCORECARD, V_FLEET_UTILIZATION                      │
│                                                                         │
│   Analytics tables: route rankings, driver rankings,                   │
│                      safety scoring, utilization bands                  │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
┌───────────────────────────┐      ┌─────────────────────────────────────┐
│    DIRECT QUERY LAYER     │      │          AI AGENT LAYER             │
│                           │      │                                     │
│  sf_connect.py            │      │  agent.py                           │
│  Centralized connection   │      │  Gemini 2.5 Flash                   │
│  module with:             │      │  Automatic function calling          │
│  · keep-alive ping        │      │  9 specialized tools                │
│  · startup assertions     │      │  Multi-step reasoning               │
│  · 120s query cache TTL   │      │  Natural language synthesis         │
│  · structured logging     │      │                                     │
│  · retry decorator        │      │  tools.py  ──▶  tool_schemas.py     │
└───────────────────────────┘      └─────────────────────────────────────┘
                    │                            │
                    └─────────────┬──────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT PRESENTATION LAYER                         │
│                                                                         │
│  app/streamlit_app.py  (9 tabs)                                         │
│  ├── 📊 Overview          KPI cards + monthly revenue chart             │
│  ├── 🚛 Fleet & Drivers   Truck/driver performance with filtering       │
│  ├── 🗺️  Routes           Route scorecard and margin thresholds         │
│  ├── ⛽ Fuel Spend        Spend by state/city                           │
│  ├── 📈 Monitoring        Pipeline latency, error rate, p50/p90/p99     │
│  ├── 🔬 Analytics         Materialized analytics tables                 │
│  ├── 🎯 Executive         Auto-loading KPIs, heatmap, live SQL explorer │
│  ├── ⚠️  Safety           Incident analytics and claim cost charts      │
│  └── 🤖 Agent Chat        Gemini agent with tool-call trace expander    │
│                                                                         │
│  Sidebar: project description, data-freshness, quick-link navigation    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT LAYER                                  │
│                                                                         │
│  Streamlit Community Cloud  ──  https://cs5542logisticsai.streamlit.app │
│  Secrets: injected via Streamlit Secrets (Snowflake + Gemini + AWS)     │
│  Health check: FastAPI /health sidecar (Lab 9)                          │
│  Uptime monitoring: UptimeRobot (Lab 9)                                 │
│  CI/CD: GitHub Actions (eval harness + smoke test on every push)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 Data Ingestion Layer

| Component | File | Purpose |
|---|---|---|
| Batch orchestrator | `scripts/run_pipeline.py` | Runs SQL files in order; loads all 14 CSVs into Snowflake |
| S3 pipeline | `sql/06_s3_pipeline.sql` | External stage for AWS S3 ingestion (requires ACCOUNTADMIN) |
| Schema DDL | `sql/01_create_schema.sql` | Creates 14 typed tables with constraints |
| Load scripts | `sql/02_stage_and_load.sql` | Internal staging + `COPY INTO` for each table |
| Derived views | `sql/04_views.sql` | 5 analytical views aggregating base tables |
| Analytics tables | `sql/05_derived_analytics.sql` | 4 materialized tables for rankings and scoring |

All 14 source CSVs are from the [Logistics Operations Database](https://www.kaggle.com/datasets/yogape/logistics-operations-database) (Kaggle, `yogape`). Schema types and constraints are enforced at the Snowflake DDL level.

### 2.2 Snowflake Data Layer

The warehouse `CS5542_WEEK5` organizes data in three tiers:

- **Base tables (14):** Direct CSV mirrors with enforced column types.
- **Derived views (5):** Read-only SQL views that join and aggregate base tables. These are the primary data source for most dashboard tabs and agent tools.
- **Materialized analytics tables (4):** Pre-computed rankings, scoring bands, and route/driver leaderboards used by the Executive and Analytics tabs.

Query caching is handled by Streamlit's `@st.cache_data` decorator with a 120-second TTL on all `run_query()` calls, preventing redundant Snowflake round trips.

### 2.3 AI Agent Layer

The agent is implemented in `agent.py` using the Google Gemini 2.5 Flash API with automatic function calling enabled.

**Agent lifecycle per query:**
1. User submits a natural-language question in the Agent Chat tab.
2. `agent.py` constructs a prompt containing the user message, conversation history, and the nine tool schemas from `tool_schemas.py`.
3. Gemini selects and calls one or more tools automatically. The Python SDK dispatches the call to the corresponding function in `tools.py`.
4. Tool results are returned to Gemini, which continues reasoning (potentially calling additional tools) until it has enough information to synthesize a final answer.
5. The final text response is streamed back to Streamlit and appended to the chat history. The tool-call trace (tool name, inputs, latency) is written to a collapsible expander (Lab 9).

**Nine analytical tools:**

| # | Tool | Primary Data Source |
|---|---|---|
| 1 | `query_snowflake` | Any Snowflake table (read-only SQL) |
| 2 | `get_monthly_revenue` | `V_MONTHLY_REVENUE` |
| 3 | `get_fleet_performance` | `V_TRIP_PERFORMANCE` |
| 4 | `get_pipeline_logs` | `logs/pipeline_logs.csv` |
| 5 | `get_safety_metrics` | `SAFETY_INCIDENTS` + `DRIVERS` |
| 6 | `get_route_profitability` | `V_ROUTE_SCORECARD` |
| 7 | `get_delivery_performance` | `DELIVERY_EVENTS` |
| 8 | `get_maintenance_health` | `MAINTENANCE_RECORDS` + `TRUCKS` |
| 9 | `get_fuel_spend_analysis` | `FUEL_PURCHASES` |

### 2.4 Logging and Observability (Lab 9)

| Signal | Mechanism | Location |
|---|---|---|
| Query audit trail | CSV append on every `run_query()` call | `logs/pipeline_logs.csv` |
| Application logs | Python `logging` module (INFO/DEBUG) | stdout + Streamlit Cloud log viewer |
| Agent tool-call trace | Streamlit expander in Chat tab | In-browser, per query |
| Snowflake startup assertions | Checked in `sf_connect.py` at import time | Application startup |
| /health endpoint | FastAPI sidecar (`health_server.py`) | `GET /health` → JSON status |
| Uptime monitoring | UptimeRobot polling /health every 5 min | External SaaS |
| CI evaluation | GitHub Actions on every push to `main` | `.github/workflows/ci.yml` |

### 2.5 ReMindRAG Integration (Lab 7)

Lab 7 reproduced and extended the ReMindRAG research system (LLM-guided knowledge graph traversal for efficient RAG). The core library lives in a separate repository (`ReMindRAG_Week7`) and is validated on every CI run via the smoke test script `tests/smoke_test.py`. Integration with the main platform is ongoing in Lab 9, with the RAG subsystem targeted as an additional retrieval backend for the Agent Chat tab.

---

## 3. Data Flow Summary

```
User Query (Streamlit Chat)
        │
        ▼
  agent.py  ──builds prompt──▶  Gemini 2.5 Flash API
                                        │
                          ┌─────────────┘
                          │  selects tool(s) automatically
                          ▼
                    tools.py  ──queries──▶  Snowflake / CSV
                          │
                          └──returns data──▶  Gemini (synthesis)
                                                    │
                                        ──response──▶  Streamlit Chat UI
                                                    │
                                        ──trace─────▶  Expander (tool calls)
                                                    │
                                        ──appends───▶  pipeline_logs.csv
```

---

## 4. Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `SNOWFLAKE_ACCOUNT` | Yes | Snowflake account identifier |
| `SNOWFLAKE_USER` | Yes | Snowflake username |
| `SNOWFLAKE_PASSWORD` | Yes | Snowflake password |
| `SNOWFLAKE_WAREHOUSE` | Yes | Compute warehouse name |
| `SNOWFLAKE_DATABASE` | Yes | `CS5542_WEEK5` |
| `SNOWFLAKE_SCHEMA` | Yes | `PUBLIC` |
| `AWS_ACCESS_KEY_ID` | S3 only | Required for S3 external stage |
| `AWS_SECRET_ACCESS_KEY` | S3 only | Required for S3 external stage |
| `LOG_LEVEL` | No | Set to `DEBUG` for verbose agent traces (default: `INFO`) |

All credentials are stored in `.env` locally (gitignored) and injected via Streamlit Secrets in the cloud deployment. Never commit `.env` to the repository.

---

## 5. Known Limitations and Future Work

| Area | Current Limitation | Proposed Improvement |
|---|---|---|
| Ingestion | Batch `COPY INTO` only; no real-time streaming | Snowpipe + S3 event notifications |
| Agent latency | 7–28 s per query depending on tool call count | Response streaming / partial results |
| Tool selection | Agent occasionally skips expected tools on ambiguous queries | Fine-tuned tool descriptions or few-shot examples in system prompt |
| Data freshness | Derived tables require manual re-run after new data loads | Scheduled Snowflake tasks |
| RAG integration | ReMindRAG is a separate subsystem not yet wired into Agent Chat | Full integration planned post-Lab 9 |
| SQL Explorer cap | Executive tab limits results to 500 rows | Pagination or export-to-CSV |
