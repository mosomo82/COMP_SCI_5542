# Lab 9 Development Report

## Application and Deployment Enhancement

**CS 5542 Big Data and Analytics**
Team: Tony Nguyen, Daniel Evans, Joel Vinas
Spring 2026

## 1. PROJECT OVERVIEW

Our team has built and iteratively improved an AI-powered logistics platform across Labs 5–8 and integrated all components into the Phase 2 project: HyperLogistics, a Snowflake-native neuro-symbolic supply chain resilience system. Lab 6 delivered a Multi-Agent Trucking Analytics Dashboard. Lab 7 reproduced and extended ReMindRAG. Lab 8 introduced the Consensus Planning Protocol (CPP) with a domain-adapted Phi-2 safety validation agent. Lab 9 directly addresses instructor feedback from Labs 6 and 7, fixes critical Lab 8 evaluation failures, and integrates all three components into the unified Phase 2 system for Research-A-Thon demonstration.

**Phase 2 GitHub:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System
**Phase 2 Live App:** https://cs5542hyperlogistics.streamlit.app/
**Lab 6 Live App:** https://cs5542logisticsai.streamlit.app/
**Lab 8 Live App:** https://cs5542lab8.streamlit.app/
**Lab 6–9 Repo:** https://github.com/mosomo82/COMP_SCI_5542
**Lab 7 Repo:** https://github.com/mosomo82/ReMindRAG_Week7

## 2. PRIOR LAB FEEDBACK ADDRESSED

**Lab 6 Feedback**

- "Minor improvements could include expanding the documentation of system architecture and evaluation details."
  **Response:** `ARCHITECTURE.md` and `EVALUATION.md` authored for `Week_9/`. Section 3.5 below.

**Lab 7 Feedback**

- "Minor improvements could include expanding the benchmarking results and adding additional automated verification for reproducibility experiments."
  **Response:** `BENCHMARKING.md` and `REPRO_VERIFICATION.md` authored (suite expanded 17–26 tests with CI). Section 3.6 below.

**Lab 8 Proactive Fixes**
Three critical issues identified in the Lab 8 evaluation are resolved in Lab 9: the Q13–Q14 monotonicity failure (CPP Spatial SQL hard gate), hallucinated bridge constraints (evidence injection), and a contradictory accuracy number in the report (corrected). Section 3.7 below.

## 3. APPLICATION ENHANCEMENTS (LAB 9)

### 3.1 Lab 6 UI and Stability

- Persistent sidebar, Agent Chat tool-call trace expander, Reset Session button, SQL Explorer error UX.
- Snowflake keep-alive ping: cold-start latency ~8s → <2s. Exponential-backoff retry on all Gemini API calls.
- Python logging module replacing print statements. FastAPI `/health` endpoint + UptimeRobot uptime monitoring.
- Agent eval suite expanded 5–10 scenarios with GitHub Actions CI (70% pass-rate gate).

### 3.2 Lab 7 Benchmarking and Verification

- `BENCHMARKING.md`: constrained LooGLE run (F1=0.496, consistent with paper's 0.49), HotpotQA subset (F1=0.58 vs paper's 0.61), efficiency verification (37–42% API call reduction on repeated queries).
- `REPRO_VERIFICATION.md`: 9 new automated tests across `test_benchmark_smoke.py` and `test_repro_variance.py`. Suite expanded 17–26 tests at zero API cost, fully integrated into CI.

### 3.3 Lab 8 Critical Fixes

- **CPP Spatial SQL hard gate:** bridge constraint check before any LLM invocation. Monotonicity invariant (APPROVE/VETO on bridge violation) is now deterministic SQL — all 13 metamorphic tests pass.
- **Evidence injection:** `[RETRIEVED CONSTRAINTS]` block injected verbatim into prompt. Model instructed not to recall limits from memory. Hallucination rate drops from ~40% to 8%.
- **Accuracy discrepancy corrected:** baseline was 40% on both platforms (not 60% as stated in Section 5 intro). `EVALUATION.md` documents correct values.
- **CoT-formatted training data:** `generate_dataset.py` updated for full 4-step chains. Dataset expanded 100–300+ pairs across 5 disruption types. PEFT CoT quality: 67% → 83%.
- **Evaluation expanded** 15–50 queries with 5-dimension rubric (0–10). Metamorphic suite expanded 3–13 pairs. GitHub Actions CI added.

### 3.4 Lab 6 Documentation

- `ARCHITECTURE.md` (Week_6/): 4-layer system, agent lifecycle, 9 tools, observability stack.
- `EVALUATION.md` (Week_6/): 10-scenario methodology, S2 root-cause, CI integration, traceability table.

### 3.5 Lab 7 Documentation

- `BENCHMARKING.md`: paper-vs-repro comparison, constrained runs, variance table, JSON output schema.
- `REPRO_VERIFICATION.md`: 26-test suite design, CI YAML, manual verification table.

### 3.6 Phase 2 Integration (CS5542_SmartSC_Optimization_System)

All Lab 6–8 components are integrated into the Phase 2 unified system for Research-A-Thon demonstration:

- **CPP agent pipeline (`src/agents/`):** `compliance_agent.py` implements the Spatial SQL hard gate; `context_agent.py` wires ReMindRAG into the retrieval layer; evidence injection is applied at the prompt level.
- **Unified Streamlit dashboard (`src/app/dashboard.py`):** merges Lab 6 analytics tabs (fleet, routes, safety, fuel) with the Lab 8 CPP dispatcher interface. Persistent sidebar, reasoning path expander, and retry decorator applied throughout.
- **Data pipeline (`src/run_pipeline.py`):** logging module replaces print statements; `verify_pipeline.py` adds row-count and schema assertions for all 4 SILVER tables.
- **CI/CD (`.github/workflows/ci.yml`):** pipeline smoke test + CPP unit tests + system eval (mock mode) + ReMindRAG 26-test suite — all run automatically on every push with a 70% pass-rate gate.
- **`ARCHITECTURE.md` + `EVALUATION.md` at Phase 2 repo root:** full system-level reference documentation covering the 4-layer architecture, CPP detail, RAG benchmarks, and pipeline evaluation results.

## 4. EXTENSION OF PHASE-2 PROTOTYPE

| Lab | Capability           | Baseline                    | Lab 9 Enhancement                             |
| :-- | :------------------- | :-------------------------- | :-------------------------------------------- |
| 6   | Architecture docs    | README only                 | `ARCHITECTURE.md` + `EVALUATION.md` (Week_9/) |
| 6   | UI & monitoring      | 9 raw tabs, manual eval     | Sidebar, tool trace, 10 scen., CI             |
| 6   | Logging & stability  | Print stmts, ~8s cold-start | Logging, retry, <2s, `/health`                |
| 7   | Benchmarking         | Metrics unverified          | LooGLE F1=0.496, HotpotQA F1=0.58             |
| 7   | Repro. verification  | 17 structural tests         | 26 tests + CI (zero API cost)                 |
| 8   | Monotonicity failure | FAIL both platforms         | PASS — Spatial SQL hard gate                  |
| 8   | Hallucination (PEFT) | ~40% rate                   | 8% — evidence injection                       |
| 8   | Evaluation coverage  | 15 queries, binary scorer   | 50 queries, 5-dim rubric (0–10)               |
| P2  | CPP integration      | Architecture described only | `compliance_agent.py` + wired CPP             |
| P2  | Unified dashboard    | Separate lab apps           | Single app: CPP + Lab 6 analytics             |
| P2  | System CI            | None                        | GitHub Actions: pipeline+eval+RAG             |
| P2  | System docs          | README only                 | `ARCHITECTURE.md` + `EVALUATION.md` (root)    |

## 5. REPOSITORY AND DEPLOYMENT LINKS

| Resource                    | URL                                                                                      | Last Commit |
| :-------------------------- | :--------------------------------------------------------------------------------------- | :---------- |
| Phase 2 Live App            | https://cs5542hyperlogistics.streamlit.app/                                              | N/A         |
| Phase 2 GitHub              | https://github.com/mosomo82/CS5542_SmartSC_Optimization_System                           | `35422ab`   |
| Lab 6 Live App              | https://cs5542logisticsai.streamlit.app/                                                 | N/A         |
| Lab 6–9 Main Repo           | https://github.com/mosomo82/COMP_SCI_5542                                                | `f5b11dd`   |
| Lab 7 ReMindRAG Repo        | https://github.com/mosomo82/ReMindRAG_Week7                                              | `a0e7603`   |
| Phase 2 ARCHITECTURE.md     | https://github.com/mosomo82/CS5542_SmartSC_Optimization_System/blob/main/ARCHITECTURE.md | `35422ab`   |
| Phase 2 EVALUATION.md       | https://github.com/mosomo82/CS5542_SmartSC_Optimization_System/blob/main/EVALUATION.md   | `35422ab`   |
| Lab 8 ARCHITECTURE.md       | https://github.com/mosomo82/COMP_SCI_5542/blob/main/Week_8/ARCHITECTURE.md               | `f5b11dd`   |
| Lab 8 EVALUATION.md         | https://github.com/mosomo82/COMP_SCI_5542/blob/main/Week_8/EVALUATION.md                 | `f5b11dd`   |
| Lab 6 ARCHITECTURE.md       | https://github.com/mosomo82/COMP_SCI_5542/blob/main/Week_6/ARCHITECTURE.md               | `f5b11dd`   |
| Lab 6 EVALUATION.md         | https://github.com/mosomo82/COMP_SCI_5542/blob/main/Week_6/EVALUATION.md                 | `f5b11dd`   |
| Lab 7 BENCHMARKING.md       | https://github.com/mosomo82/ReMindRAG_Week7/blob/main/BENCHMARKING.md                    | `a0e7603`   |
| Lab 7 REPRO_VERIFICATION.md | https://github.com/mosomo82/ReMindRAG_Week7/blob/main/REPRO_VERIFICATION.md              | `a0e7603`   |

## 6. SUMMARY

Lab 9 closes all identified gaps across Labs 6, 7, and 8 while completing the Phase 2 integration of HyperLogistics as a unified, demonstrable system. Lab 6 and 7 instructor feedback is addressed by six new documentation artifacts. Lab 8's three critical failures are resolved by deterministic SQL, evidence injection, and expanded evaluation. The Phase 2 repo gains a fully wired CPP pipeline, a unified Streamlit dashboard, end-to-end CI, and system-level architecture and evaluation documentation — positioning HyperLogistics for the Research-A-Thon competition.
