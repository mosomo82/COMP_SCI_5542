# ARCHITECTURE.md — HyperLogistics: Smart Supply Chain Optimization System

> **CS 5542 — Big Data and Analytics | Phase 2 / Lab 9**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Live App: [cs5542hyperlogistics.streamlit.app](https://cs5542hyperlogistics.streamlit.app/)
> Repository: [CS5542_SmartSC_Optimization_System](https://github.com/mosomo82/CS5542_SmartSC_Optimization_System)

---

## 1. System Overview

HyperLogistics is a Snowflake-native neuro-symbolic supply chain resilience system that closes the "prediction-action gap" in middle-mile logistics. When a real-time disruption occurs (weather alert, accident blackspot, bridge restriction), the system autonomously generates a constraint-compliant rerouting justification grounded in live data — before a dispatcher has to ask.

The system integrates three research components developed across Labs 6–8 into a single deployable platform:

- **Lab 6:** Gemini 2.5 Flash multi-agent analytics layer (9-tool Snowflake query agent)
- **Lab 7:** ReMindRAG LLM-guided knowledge graph traversal (low-cost explainable RAG)
- **Lab 8:** Phi-2 + QLoRA domain-adapted safety validation agent (CPP Consensus Planning Protocol)

---

## 2. Full System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  EXTERNAL DATA SOURCES                                                       ║
║                                                                              ║
║  US Accidents (7.7M)    NOAA GSOD (multi-TB)    NBI Bridges (600K+)         ║
║  DataCo Logistics       Logistics Ops DB (14 tables)                        ║
║       │                       │                        │                    ║
║  Snowpipe (stream)    External Table (S3)       Internal Stage (batch)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 1 — DATA PERCEPTION  (Medallion Architecture)                        ║
║                                                                              ║
║  BRONZE (raw)                                                                ║
║  BRONZE.TRAFFIC_INCIDENTS · BRONZE.NOAA_GSOD · BRONZE.BRIDGE_INVENTORY      ║
║  BRONZE.RAW_LOGISTICS (DataCo) · BRONZE.LOGISTICS (14-table ops DB)         ║
║           │  preprocess_*.py                                                 ║
║           ▼                                                                  ║
║  SILVER (analytics-ready)                                                    ║
║  SILVER.RISK_HEATMAP_VIEW · SILVER.WEATHER_ALERTS                           ║
║  SILVER.BRIDGE_INVENTORY_GEO · SILVER.LOGISTICS_VECTORIZED                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
╔═══════════════════════════════╗  ╔════════════════════════════════════════╗
║  LAYER 2 — INTELLIGENCE       ║  ║  LAYER 2 — FORECASTING                 ║
║                               ║  ║                                        ║
║  ReMindRAG (Lab 7)            ║  ║  SRSNet (NeurIPS 2025)                 ║
║  LLM-guided KG traversal      ║  ║  Adaptive 4–8h risk propagation        ║
║  Retrieves disruption history ║  ║  forecast across middle-mile patches   ║
║  + constraint precedents      ║  ║  Runs via Snowpark Python              ║
║  from SILVER.LOGISTICS_       ║  ║                                        ║
║  VECTORIZED (768D embeddings) ║  ║  Snowflake Cortex (Gemini)             ║
║                               ║  ║  Secure on-platform LLM inference      ║
╚═══════════════════════════════╝  ╚════════════════════════════════════════╝
                    │                                │
                    └───────────────┬────────────────┘
                                    │  route candidates + retrieved context
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 3 — VALIDATION & SAFETY  (Consensus Planning Protocol)               ║
║                                                                              ║
║  STEP 3A — DETERMINISTIC SPATIAL SQL GATE  (cannot be overridden)           ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  SELECT MIN(weight_limit_tons), MIN(vertical_clearance_mt)           │   ║
║  │  FROM   SILVER.BRIDGE_INVENTORY_GEO                                  │   ║
║  │  WHERE  ST_INTERSECTS(bridge_geom, route_corridor_geom)              │   ║
║  │                                                                      │   ║
║  │  IF vehicle_weight > MIN(weight_limit_tons)                          │   ║
║  │     OR vehicle_height > MIN(vertical_clearance_mt)  → HARD VETO     │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                │                           │                                ║
║         HARD VETO                        PASS                               ║
║       (return immediately)                 │                                ║
║                                            ▼                                ║
║  STEP 3B — DOMAIN-ADAPTED LLM  (Phi-2 + PEFT, Lab 8)                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  Input: disruption context + route + [RETRIEVED CONSTRAINTS] block  │   ║
║  │  Bridge limits injected verbatim from Step 3A — model never recalls │   ║
║  │  physical limits from memory                                         │   ║
║  │  Output: 4-step CoT justification (APPROVE or VETO + reasoning)     │   ║
║  │  Disruption → Route → Constraint → Decision                         │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — APPLICATION                                                       ║
║                                                                              ║
║  src/app/dashboard.py  (Streamlit in Snowflake)                              ║
║  ├── Risk Heatmap          Interactive map of accident blackspots            ║
║  ├── Weather Alerts        Live NOAA disruption feed                        ║
║  ├── Route Comparison      Side-by-side reroute evaluation with CPP score   ║
║  ├── Ask the Agent         Natural language dispatcher queries (Gemini)      ║
║  ├── Reasoning Path        ReMindRAG source citations per justification      ║
║  └── Analytics (Lab 6)     9-tab trucking analytics agent (Gemini + tools)  ║
║                                                                              ║
║  Live: https://cs5542hyperlogistics.streamlit.app/                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Data Layer — Medallion Architecture

### 3.1 Bronze Layer (Raw Ingestion)

| Table | Source | Ingestion Method | Size |
|---|---|---|---|
| `BRONZE.TRAFFIC_INCIDENTS` | US Accidents (Kaggle) | Snowpipe (S3 event-driven) | 7.7M rows × 47 cols |
| `BRONZE.NOAA_GSOD` | NOAA GSOD (AWS Open Data) | External Table (direct S3) | Multi-terabyte |
| `BRONZE.BRIDGE_INVENTORY` | NBI US DOT | Internal Stage (batch annual) | 600K+ records |
| `BRONZE.RAW_LOGISTICS` | DataCo Supply Chain | Snowpipe | 180K rows |
| `BRONZE.LOGISTICS` | Logistics Ops DB (14 CSVs) | Internal Stage | 85K+ records |

### 3.2 Silver Layer (Analytics-Ready)

| Table / View | Source | Key Transformation | Primary Consumer |
|---|---|---|---|
| `SILVER.RISK_HEATMAP_VIEW` | `BRONZE.TRAFFIC_INCIDENTS` | Geospatial clustering by freight corridor; severity scoring | Layer 4 Risk Heatmap |
| `SILVER.WEATHER_ALERTS` | `BRONZE.NOAA_GSOD` | Rule-based alert extraction (PRCP>50mm, SNWD>10cm, VISIB<1mi); semantic description | Layer 2 SRSNet + Layer 3 CPP |
| `SILVER.BRIDGE_INVENTORY_GEO` | `BRONZE.BRIDGE_INVENTORY` | Lat/lon → Snowflake GEOGRAPHY via `ST_GEOGFROMTEXT`; constraint field extraction | Layer 3 Spatial SQL gate |
| `SILVER.LOGISTICS_VECTORIZED` | `BRONZE.LOGISTICS` | 768D vector embeddings via Cortex; chunked for RAG retrieval | Layer 2 ReMindRAG |

### 3.3 Ingestion Automation (Tony Nguyen — Phase 2)

| Script | Purpose |
|---|---|
| `src/ingestion/setup_s3_automation.py` | Creates Snowflake external stages, storage integrations, Snowpipes |
| `src/sql/02_setup_s3_automation.sql` | SQL DDL for S3 stages, pipes, and event notifications |
| `src/run_pipeline.py` | Master orchestrator — runs all ingestion and preprocessing in sequence |
| `src/verify_pipeline.py` | Post-ingestion verification — row counts, schema checks, null audits |
| `upload_to_s3.py` | Uploads local datasets to S3 bucket for Snowpipe pickup |
| `src/utils/snowflake_conn.py` | Centralized Snowflake connection with keep-alive and retry decorator |

---

## 4. Intelligence Layer

### 4.1 ReMindRAG (Lab 7 Integration)

ReMindRAG provides the retrieval backbone for the CPP agent. When a dispatcher submits a rerouting query, ReMindRAG traverses the knowledge graph built from `SILVER.LOGISTICS_VECTORIZED` to surface:

- Prior disruptions of the same type on the same corridor
- Previously approved or vetoed routing decisions with outcomes
- Constraint records for bridges on the candidate route

The traversal memorizes paths — similar future queries reuse the cached traversal, reducing API calls by 37–42%.

**Key parameters (from `src/config/config.yaml`):**

| Parameter | Value | Effect |
|---|---|---|
| `seed` | 42 | Deterministic retrieval across runs |
| `edge_weight_coefficient` | 0.1 | Reliance on edge embeddings for strong links |
| `strong_connection_threshold` | 0.55 | KG traversal depth control |
| `max_jumps` | 10 | Max nodes expanded per query |

### 4.2 SRSNet (NeurIPS 2025 Integration)

SRSNet applies selective representation spaces to adaptively patch time-series weather and accident data into 4–8 hour forecast windows aligned to middle-mile transit times. Outputs a risk score per route segment that gates which route candidates enter the CPP.

### 4.3 Snowflake Cortex (Gemini)

All LLM inference runs inside Snowflake via Cortex, keeping data within the Snowflake trust boundary. Embeddings use Google Gemini (768D). The Lab 6 Gemini agent (9 analytical tools) also runs through Cortex for the analytics dashboard tabs.

---

## 5. Validation Layer — CPP Details

### 5.1 Three-Agent Negotiation

The Consensus Planning Protocol uses three specialized agents that must agree before a route is dispatched:

| Agent | Optimizes For | Override Capability |
|---|---|---|
| Context Agent | Historical disruption similarity (ReMindRAG) | Can suggest — cannot approve alone |
| Efficiency Agent | SRSNet risk score + distance delta | Can suggest — cannot approve alone |
| Compliance Agent | DOT bridge constraints (Spatial SQL) | **HARD VETO — cannot be overridden** |

A route is only dispatched when all three agents reach consensus. The Compliance Agent's Spatial SQL gate (Step 3A) is deterministic and executes before the LLM is invoked — no LLM reasoning can produce an APPROVE if a bridge limit is violated.

### 5.2 Evidence Injection (Lab 9 Fix)

Bridge constraints retrieved in Step 3A are injected verbatim into the Step 3B prompt:

```
[RETRIEVED CONSTRAINTS]
Bridge ID: NBI-IL-4821 | Route: I-55 SB | Weight limit: 40 tons | Height: 14.2 ft
Binding limit (most restrictive): 40 tons / 14.2 ft
```

The model is explicitly instructed not to recall physical limits from parametric memory. This reduces hallucination rate from ~40% (Lab 8 baseline) to 8% (Lab 9 with injection).

---

## 6. Application Layer

### 6.1 Dashboard Components

| Tab | Data Source | Description |
|---|---|---|
| Risk Heatmap | `SILVER.RISK_HEATMAP_VIEW` | Interactive geospatial map of accident blackspots along I-70, I-80, I-55, I-10 |
| Weather Alerts | `SILVER.WEATHER_ALERTS` | Live disruption feed: severity, affected route segments, alert type |
| Route Comparison | CPP output | Side-by-side original vs rerouted: distance delta, risk score, compliance status |
| Ask the Agent | Gemini + ReMindRAG | Natural language dispatcher queries with reasoning path citations |
| Analytics (Lab 6) | 9-tool Gemini agent | Fleet performance, revenue, safety, fuel, route profitability |

### 6.2 Lab 9 UI Enhancements

Applied from Lab 6 improvements to the unified dashboard:

- Persistent sidebar with data-freshness timestamp and quick navigation
- Collapsible reasoning path expander showing ReMindRAG traversal steps and CPP agent decisions
- Reset Session button clearing conversation state
- Structured logging (INFO/DEBUG) replacing print statements throughout `dashboard.py`
- Exponential-backoff retry on all Cortex and external API calls

---

## 7. Deployment

| Component | Platform | URL |
|---|---|---|
| Streamlit Dashboard | Streamlit in Snowflake | https://cs5542hyperlogistics.streamlit.app/ |
| Data Warehouse | Snowflake Enterprise (AWS) | `SMART_SUPPLY_CHAIN_DB` |
| Data Lake | AWS S3 | S3 bucket with Snowpipe event triggers |
| CI/CD | GitHub Actions | `.github/workflows/ci.yml` |

### Environment Variables (`.env.template`)

```
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=SMART_SUPPLY_CHAIN_DB
SNOWFLAKE_SCHEMA=SILVER
S3_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
GEMINI_API_KEY=
LOG_LEVEL=INFO
```

---

## 8. Repository Structure

```
CS5542_SmartSC_Optimization_System/
├── docs/
│   ├── Figures/                   # Architecture diagrams, screenshots
│   ├── snowflake_setup.md         # Database setup guide
│   ├── s3_automation_guide.md     # Full S3 + Snowpipe setup
│   └── detailed_documentation.md  # Component-level docs
├── notebooks/                     # Colab training notebooks (SRSNet, PEFT)
├── reports/                       # Phase 2 PDF reports
├── reproducibility/               # Reproduce scripts and guides
├── src/
│   ├── agents/
│   │   ├── cpp_agent.py           # Consensus Planning Protocol (3-agent)
│   │   ├── context_agent.py       # ReMindRAG retrieval agent
│   │   ├── efficiency_agent.py    # SRSNet risk scoring agent
│   │   └── compliance_agent.py    # Spatial SQL hard gate (Step 3A)
│   ├── app/
│   │   └── dashboard.py           # Unified Streamlit dashboard
│   ├── config/
│   │   └── config.yaml            # Hyperparameters, seeds, paths
│   ├── ingestion/
│   │   ├── ingest_accidents.py    # US Accidents → BRONZE
│   │   ├── ingest_bridges.py      # Bridge Inventory → BRONZE
│   │   ├── ingest_dataco.py       # DataCo → BRONZE
│   │   ├── ingest_logistics.py    # Logistics Ops → BRONZE
│   │   └── setup_s3_automation.py # S3 + Snowpipe automation
│   ├── preprocessing/
│   │   ├── preprocess_accidents.py
│   │   ├── preprocess_bridges.py
│   │   ├── preprocess_weather.py
│   │   ├── preprocess_dataco.py
│   │   └── preprocess_logistics.py
│   ├── sql/
│   │   ├── 00_create_database.sql
│   │   ├── 01_setup_noaa.sql
│   │   └── 02_setup_s3_automation.sql
│   ├── utils/
│   │   └── snowflake_conn.py      # Connection with retry + keep-alive
│   ├── run_pipeline.py            # Master orchestrator
│   └── verify_pipeline.py        # Post-ingestion verification
├── tests/
│   ├── evaluate_system.py         # 50-scenario golden dataset eval
│   ├── test_cpp_gate.py           # CPP Spatial SQL unit tests
│   └── test_pipeline.py           # End-to-end pipeline smoke test
├── ARCHITECTURE.md                # This file
├── EVALUATION.md                  # Evaluation methodology and results
├── CONTRIBUTIONS.md               # Team contribution log
├── requirements.txt               # Pinned Python dependencies
└── .env.template                  # Credential template
```

---

## 9. Known Limitations and Future Work

| Area | Current Limitation | Planned Improvement |
|---|---|---|
| Real-time ingestion | Snowpipe batch latency ~minutes | Snowpipe Streaming for sub-second updates |
| Agent latency | CPP adds 8–15s per query | Response streaming / partial results in UI |
| PEFT model | Runs on Colab/local GPU only | Deploy adapter via Snowflake Container Services |
| Benchmark coverage | 50-scenario golden dataset | Expand to 200+ with more disruption types |
| RAG memory | ReMindRAG path cache resets per session | Persistent cache in Snowflake KV store |
