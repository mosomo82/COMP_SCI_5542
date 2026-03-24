# Individual Contribution Report — Lab 9 Phase 5

## Team Lead / Data & Back-End Engineer: Daniel Evans

### 1. Contributions Overview

I focused on the Phase 4 unified dashboard integration, Phase 2 pipeline benchmarking, and collaborative documentation for the Research-A-Thon submission.

### 2. Phase 4 Dashboard (Repo 1) — 100% Complete

I developed the unified Streamlit dashboard for the `CS5542_SmartSC_Optimization_System` repository, merging analytics features from Lab 6 with the new CPP safety protocol from Lab 8.

- **Persistent Sidebar:** Implemented a global navigation sidebar with project description, live data-freshness timestamps, and a 'Reset Session' button for better UX.
- **Reasoning Path Expander:** Created a collapsible UI component that displays the underlying ReMindRAG traversal sources and the CPP Compliance Agent's decision (PASS/VETO).
- **Operational Analytics:** Successfully ported and integrated the Fleet, Routes, Fuel, and Safety analytics tabs from Lab 6 into the unified dashboard.
- **Stability & Resilience:** Applied exponential-backoff retry decorators to all Snowflake Cortex and external API calls to prevent dashboard failures during rate-limiting events.

### 3. Phase 2 Benchmarking (Repo 3) — 100% Complete

I executed the pipeline evaluation runs for the `ReMindRAG_Week7` reproducibility project and committed the results.

- **LooGLE Evaluation:** Ran 5-title subset runs with an Avg F1 of 0.7125.
- **HotpotQA Evaluation:** Ran 50-question subset tests with an Avg F1/Accuracy of 0.7600.
- **Results Logging:** Recorded all scores in the project root `eval_results.json` for CI/CD integration.

### 4. Commits & Evidence

| Repository                           | Commits / Artifacts                                     |
| :----------------------------------- | :------------------------------------------------------ |
| `CS5542_SmartSC_Optimization_System` | `src/app/dashboard.py`, `CONTRIBUTIONS.md`, `README.md` |
| `ReMindRAG_Week7`                    | `eval_results.json`, `tests/test_repro_variance.py`     |
| `COMP_SCI_5542`                      | `Week_9/reports/CONTRIBUTION_DE.md`                     |

---

**Verified by:** Daniel Evans
**Date:** 2026-03-24
