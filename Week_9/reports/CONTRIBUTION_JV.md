# Individual Contribution Statement — Joel Vinas (Lab 9)

**GitHub:** `joelvinas`  
**Role:** Tests & CI (Phase 4)  
**Contribution:** 33%  
**Lab9 Repo:** https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_9

---

## Lab 9 Scope

Joel owned **Phase 4** (Phase 2 Project Integration) tests and CI pipeline integration, which included creating three test files and a unified CI workflow to ensure robustness and correctness.

---

## Phase 4 — Project Phase 2 Integration Tests (`CS5542_SmartSC_Optimization_System`)

**Primary commit:** `fd9adde` — _Lab9: Phase 4 Tests and CI Implementation_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

### `tests/test_cpp_gate.py` (NEW)

- Created unit tests for the Spatial SQL hard-gate implemented.
- Mocked the Snowflake `ST_INTERSECTS` call to ensure zero real API cost.
- Test cases implemented:
  - Overweight vehicle (GVWR > MIN weight limit) -> `HARD VETO`
  - Compliant vehicle -> `PASS`
  - Height violation (Height > MIN vertical clearance) -> `HARD VETO`
- Guards against the Q13→Q14 monotonicity failure.

### `tests/test_pipeline.py` (NEW)

- Created an end-to-end smoke test for the integration pipeline.
- Test cases include:
  - Snowflake connectivity evaluation.
  - Ensuring the 4 SILVER tables have `row_count > 0`.
  - Checking the Streamlit dashboard HTTP endpoint for `200 OK` response.

### `tests/evaluate_system.py` (NEW)

- Expanded system evaluation to exactly 50 queries across 5 disruption categories:
  - Weather / Road Closure (10)
  - Overweight / Bridge Compliance (10)
  - Driver Hours / Availability (10)
  - Fuel / Cost Optimization (10)
  - Multi-hop / Combined Disruptions (10)
- Incorporated a 5-dimension rubric (0-10 scale):
  - Decision Accuracy
  - Disruption Grounding
  - Constraint Citation
  - CoT Completeness
  - Jargon Accuracy
- Integrated `--mock` mode to run deterministic fixture responses safely in CI runs.

### `.github/workflows/ci.yml` (NEW)

- Established a unified GitHub Actions CI workflow triggered on push and pull requests to `main`.
- Incorporated the following jobs:
  - Pipeline Smoke (`test_pipeline.py`)
  - CPP Unit Tests (`test_cpp_gate.py`)
  - System Eval with mock mode (`evaluate_system.py --mock`)
  - Pass-Rate gate aggregating the success of all jobs (≥ 70%).

---

## Commit Summary

| Repo | Commit | Description |
|------|--------|-------------|
| CS5542_SmartSC | `fd9adde` | **Lab9: Phase 4 Tests and CI Implementation** |

---

**33%** — Tests & CI pipeline design, spatial gate mock tests, pipeline smoke checks, expanded 50-query system evaluation script, and GitHub Actions CI workflow composition.
