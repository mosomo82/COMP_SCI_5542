# Lab 9 Implementation Plan — Joel Vinas (joelvinas)

> **Team:** EVN — Tony Nguyen · Daniel Evans · Joel Vinas  
> **Repo:** `CS5542_SmartSC_Optimization_System`  
> **Role:** Tests & CI (Phase 4) — *run in parallel with Tony (pipeline/CPP) and Daniel (dashboard)*  
> **Source:** `lab9_full_execution_plan_v2.html` — tasks t46–t49

---

## Overview

According to the parallelization guide in the full execution plan:

> *"Joel owns Phase 4 tests/CI."*

Joel's scope lives entirely in **Phase 4 (Phase 2 Project Integration)** and covers four tasks:
creating three test files and a unified CI workflow. These can all be developed in parallel with
teammates' work and should be completed before Phase 5 (the group report) begins.

---

## Phase 4 — Joel's Tasks

### 1. `tests/test_cpp_gate.py` — CPP Spatial SQL Gate Unit Tests
**Tag:** `test`

Create unit tests for the Spatial SQL hard-gate (implemented by Tony in `compliance_agent.py`).

**Test cases to implement:**

| Scenario | Input | Expected Result |
|---|---|---|
| Overweight vehicle | Vehicle GVWR > `MIN(weight_limit_tons)` on route | `HARD VETO` |
| Compliant vehicle | Vehicle GVWR within all route limits | `PASS` |
| Height violation | Vehicle height > `MIN(vertical_clearance_mt)` on route | `HARD VETO` |

**Notes:**
- Mock the Snowflake `ST_INTERSECTS` call; no live DB required for unit tests.
- Ensure tests run with zero real API cost.
- These tests guard the fix for the **Q13→Q14 monotonicity failure** identified in the Lab 8 evaluation.

---

### 2. `tests/test_pipeline.py` — End-to-End Pipeline Smoke Test
**Tag:** `test`

Create a smoke test that exercises the full pipeline integration at a high level.

**Test cases to implement:**

| Check | Pass Criterion |
|---|---|
| Snowflake connectivity | Connection established without error |
| SILVER table row counts | All 4 SILVER tables have `row_count > 0` |
| Dashboard HTTP response | Streamlit app returns HTTP `200` |

**Notes:**
- This test is the primary integration health check.
- Keep it fast and dependency-light; use environment-based credentials (already validated by Tony's startup assertions).
- Complement Tony's `verify_pipeline.py` row-count assertions — the smoke test is the lightweight CI-friendly version.

---

### 3. `tests/evaluate_system.py` — Expanded System Evaluation
**Tag:** `test`

Expand the existing `evaluate_system.py` from **15 queries** to **50 queries** with a full 5-dimension rubric and a **mock mode** for CI execution.

**Rubric Dimensions (0–10 scale, pass ≥ 7):**

| # | Dimension | What it Checks |
|---|---|---|
| 1 | Decision Accuracy | Is the final route/compliance decision correct? |
| 2 | Disruption Grounding | Is the answer anchored to the disruption described? |
| 3 | Constraint Citation | Are bridge/weight/height limits explicitly cited? |
| 4 | CoT Completeness | Does the response cover all 4 CoT steps (Disruption→Route→Constraint→Decision)? |
| 5 | Jargon Accuracy | Are logistics/supply-chain terms used correctly? |

**Query distribution across 5 disruption categories:**

| Category | Query Count |
|---|---|
| Weather / Road Closure | 10 |
| Overweight / Bridge Compliance | 10 |
| Driver Hours / Availability | 10 |
| Fuel / Cost Optimization | 10 |
| Multi-hop / Combined Disruptions | 10 |

**Mock mode requirements:**
- `--mock` flag returns deterministic fixture responses (no LLM API calls).
- Mock mode must be the default when `CI=true` env var is set.
- Ensures CI never burns API budget.

---

### 4. `.github/workflows/ci.yml` — Unified CI Workflow (New File)
**Tag:** `test`

Create a single, unified GitHub Actions CI workflow for the `CS5542_SmartSC_Optimization_System` repo.

**Workflow must include all of the following jobs/steps:**

| Step | What Runs | Pass Criterion |
|---|---|---|
| Pipeline Smoke | `tests/test_pipeline.py` | All 3 checks pass |
| CPP Unit Tests | `tests/test_cpp_gate.py` | All gate scenarios pass |
| System Eval (mock) | `tests/evaluate_system.py --mock` | Pass rate ≥ 70% |
| ReMindRAG Suite | ReMindRAG 26-test suite (`test_benchmark_smoke.py` + `test_repro_variance.py`) | All 26 tests pass |
| Pass-Rate Gate | Aggregate across all suites | ≥ 70% overall |

**CI Trigger:** Run on every `push` and `pull_request` to `main`.

**Notes:**
- No live Snowflake or LLM credentials should be required for the CI run — use mock mode and environment secrets only where unavoidable.
- The CI badge from this workflow will be referenced in Phase 5's `README.md` update (t53).

---

## Dependencies & Coordination

| My Task | Depends On |
|---|---|
| `test_cpp_gate.py` | Tony's `compliance_agent.py` interface/signature |
| `test_pipeline.py` | Tony's `run_pipeline.py` + Snowflake SILVER schema being finalized |
| `evaluate_system.py` | Tony's `cpp_agent.py` and context/prompt pipeline being wired |
| `ci.yml` | All three test files above + ReMindRAG 26-test suite from Phase 2 |

> **Tip:** Stub the compliance agent interface early so `test_cpp_gate.py` can be written and passing before Tony finalizes the implementation.

---

## Phase 5 — Report Tasks (Shared, All Members)

These tasks begin **only after all CI badges are green**.

| Task | Description | Tag |
|---|---|---|
| Pre-flight: CI badges | Verify green badges on all three repos | `test` |
| Pre-flight: Live apps | Confirm `cs5542logisticsai.streamlit.app`, `cs5542lab8.streamlit.app`, `cs5542hyperlogistics.streamlit.app` are live | `test` |
| Screenshots | Capture Lab 6 sidebar + tool trace, Lab 8 CPP evidence, Phase 2 reasoning path expander | `doc` |
| Group Report §1 | Project overview — all three labs + Phase 2 in one paragraph | `doc` |
| Group Report §2 | Prior feedback — quote Lab 6 + Lab 7 instructor comments with responses | `doc` |
| Group Report §3 | Enhancements — all four areas (Labs 6/7/8 + Phase 2) | `doc` |
| Group Report (final) | Insert real GitHub commit links + actual benchmark numbers; regenerate PDF | `doc` |
| Individual Contribution | Fill in real commit hashes/links from all three repos; generate PDF | `doc` |
| Canvas Submission | Submit both PDFs | `doc` |

---

## Completion Checklist

- [x] `tests/test_cpp_gate.py` — 3 test cases (overweight, compliant, height violation)
- [x] `tests/test_pipeline.py` — 3 smoke checks (connectivity, row counts, HTTP 200)
- [x] `tests/evaluate_system.py` — expanded to 50 queries, 5-dimension rubric, `--mock` mode
- [x] `.github/workflows/ci.yml` — unified CI (pipeline smoke + CPP unit + eval mock + ReMindRAG + ≥70% gate)
- [ ] All CI badges green before Phase 5 begins
- [ ] Individual Contribution Report updated with real commit hashes
- [ ] Both PDFs submitted to Canvas
