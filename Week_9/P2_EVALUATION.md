# EVALUATION.md — System Evaluation: HyperLogistics

> **CS 5542 — Big Data and Analytics | Phase 2 / Lab 9**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Repository: [CS5542_SmartSC_Optimization_System](https://github.com/mosomo82/CS5542_SmartSC_Optimization_System)

---

## 1. Evaluation Overview

HyperLogistics is evaluated across three dimensions that correspond to its three core subsystems:

1. **CPP Safety Validation** — Does the system correctly APPROVE or VETO rerouting decisions? Does it enforce physical constraints deterministically?
2. **RAG Retrieval Quality** — Does ReMindRAG surface relevant disruption history and constraint precedents for each query?
3. **End-to-End Pipeline** — Does the full system run reproducibly from ingestion through Streamlit dashboard response?

All evaluation is implemented in `tests/evaluate_system.py` and runs automatically via GitHub Actions CI on every push to `main`.

---

## 2. CPP Safety Validation Evaluation

### 2.1 Golden Dataset

The evaluation uses a 50-scenario golden dataset (`tests/data/evaluation_queries.json`) covering five disruption categories:

| Category | Scenarios | Description |
|---|---|---|
| Weather disruption | 10 | Severe NOAA alerts (ice, flood, high wind) on freight corridors |
| Accident blackspot | 10 | US Accidents severity-flagged incidents with lane/shoulder impact |
| Bridge constraint — weight | 10 | Vehicle GVW at or above bridge tonnage limit on proposed route |
| Bridge constraint — height | 10 | Oversized load at or above bridge vertical clearance |
| Compound disruption | 10 | Two simultaneous disruption types requiring multi-constraint reasoning |

### 2.2 Rubric Scoring (5-Dimension, 0–10)

Each scenario is scored on five dimensions. Pass threshold: ≥7/10.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Decision correctness | Wrong (APPROVE when should VETO) | Correct with caveat | Correct and confident |
| Disruption grounding | No disruption cited | Type cited only | Specific alert/event cited with severity |
| Constraint citation | No bridge limit referenced | Type mentioned (weight/height) | Bridge ID + exact limit value cited |
| CoT completeness | No reasoning chain | Partial chain (≤2 steps) | Full 4-step chain (Disruption→Route→Constraint→Decision) |
| Jargon accuracy | Domain terms absent or misused | Generic language | Correct use of LTL, deadhead, bobtail, heavy haul, etc. |

### 2.3 Results by Strategy (50-Query Expanded Evaluation)

| Strategy | Accuracy | Mean Rubric Score | Monotonicity PASS | Hallucination Rate |
|---|---|---|---|---|
| Baseline (Phi-2, no adaptation) | 38% | 3.1 / 10 | FAIL | 67% |
| Few-Shot | 66% | 6.2 / 10 | FAIL | 22% |
| SC-CoT | 90% | 8.6 / 10 | PASS | 4% |
| ReAct | 86% | 8.1 / 10 | PASS | 6% |
| PEFT + evidence injection (Lab 9) | 88% | 8.3 / 10 | PASS | 8% |

**Best strategy for production:** SC-CoT — highest rubric score across all dimensions, zero monotonicity failures, lowest hallucination rate.

**Best strategy for explainability:** ReAct — interleaved Thought/Action/Observation steps produce the clearest audit trail for dispatcher trust.

### 2.4 Metamorphic Test Suite (13 Pairs)

| Pair | Type | Description | Status |
|---|---|---|---|
| Q11 vs Q12 | Invariance | Rephrased query → same decision | PASS |
| Q13 vs Q14 | Monotonicity (original) | Add bridge violation → APPROVE flips to VETO | PASS (after CPP gate) |
| Q15 | Symmetry | Swap origin/destination → decision preserved | PASS |
| Q16 vs Q17 | Monotonicity (weight) | Increment GVW past limit → APPROVE flips to VETO | PASS |
| Q18 vs Q19 | Monotonicity (height) | Add oversized cargo → APPROVE flips to VETO | PASS |
| Q20 vs Q21 | Invariance (jargon) | "LTL" vs "less-than-truckload" → same decision | PASS |
| Q22 vs Q23 | Monotonicity (compound) | Add second disruption → cannot become more permissive | PASS |
| Q48 vs Q49 | Near-duplicate | Near-identical query on different route → decision varies | PASS |
| Q50 | Regression | Original Q13→Q14 failure re-run after CPP fix | PASS |

All 13 metamorphic tests pass after the Lab 9 CPP Spatial SQL hard gate is applied.

### 2.5 Key Fixes Applied (Lab 9)

| Lab 8 Failure | Root Cause | Lab 9 Fix | Result |
|---|---|---|---|
| Q13→Q14 FAIL (monotonicity) | LLM performed constraint check instead of SQL | CPP Step 3A Spatial SQL hard gate before LLM invocation | All monotonicity tests PASS |
| ~40% hallucination rate (RTX 3060) | Bridge limits not injected — model recalled from memory | [RETRIEVED CONSTRAINTS] block injected verbatim into prompt | 8% hallucination rate |
| CoT quality regression (PEFT 67%) | Training data lacked full 4-step reasoning chains | Expanded dataset (300+ pairs) with CoT-formatted outputs | 83% CoT quality |
| Accuracy discrepancy in report | Baseline stated as 60% in intro, 40% in table | Corrected to 40% baseline on both platforms | Consistent reporting |

---

## 3. RAG Retrieval Evaluation (ReMindRAG)

### 3.1 Retrieval Metrics (5-Title LooGLE Subset, `--seed 42`)

ReMindRAG is evaluated on the LooGLE `longdep_qa` benchmark — multi-hop, long-range-dependency QA designed to test knowledge graph traversal quality.

| Title Index | Questions | Correct | F1 | Notes |
|---|---|---|---|---|
| 0 | 8 | 4 | 0.50 | Multi-hop; 2 partial-credit answers |
| 1 | 6 | 3 | 0.49 | Long-range dependency; 1 missed entity |
| 2 | 10 | 6 | 0.54 | Shortest doc; strong performance |
| 3 | 7 | 3 | 0.44 | Dense technical domain |
| 4 | 9 | 5 | 0.51 | Mixed question types |
| **Average** | **8.0** | **4.2** | **0.496** | Consistent with paper's reported 0.49 |

### 3.2 Efficiency Verification

The paper's core claim is that path memorization reduces API calls for repeated/similar queries:

| Query Type | Avg API Calls (Naive RAG) | Avg API Calls (ReMindRAG) | Reduction |
|---|---|---|---|
| Novel queries | 1.0 | 1.9 | — (KG traversal overhead) |
| Repeated queries | 1.0 | 1.1 | −42% vs first run |
| Near-duplicate queries | 1.0 | 1.2 | −37% vs first run |

For a system handling repeated route queries (e.g., recurring I-70 corridor disruptions), this efficiency gain is practically significant — a dispatcher asking about the same corridor twice in a shift pays ~40% fewer API calls on the second query.

### 3.3 Automated Verification Tests (26 Tests)

The ReMindRAG automated test suite runs in CI with zero API cost:

| Suite | Tests | Scope |
|---|---|---|
| `test_reproducibility.py` (Lab 7) | 17 | Structural fixes: encoding, directories, path handling, determinism |
| `test_benchmark_smoke.py` (Lab 9) | 5 | Pipeline output quality, chunk retrieval, answer schema |
| `test_repro_variance.py` (Lab 9) | 4 | Cross-run determinism with real embeddings (all-MiniLM-L6-v2) |
| **Total** | **26** | All pass in ~115 seconds at zero API cost |

---

## 4. End-to-End Pipeline Evaluation

### 4.1 Pipeline Smoke Test (`tests/test_pipeline.py`)

Verifies the full pipeline from ingestion through dashboard response on every CI push:

| Step | Test | Expected |
|---|---|---|
| Snowflake connectivity | `test_snowflake_connection` | Returns `SMART_SUPPLY_CHAIN_DB` context |
| Bronze ingestion | `test_bronze_row_counts` | BRONZE tables non-empty after `run_pipeline.py` |
| Silver transformation | `test_silver_schema` | SILVER tables match expected column schema |
| CPP gate | `test_cpp_veto_on_overweight` | Returns HARD VETO for 45-ton vehicle on 40-ton bridge |
| CPP gate | `test_cpp_pass_on_compliant` | Returns PASS for compliant vehicle + weather-only disruption |
| ReMindRAG retrieval | `test_remindrag_returns_chunks` | ≥1 chunk returned for sample rerouting query |
| Dashboard response | `test_dashboard_loads` | Streamlit app returns HTTP 200 |

### 4.2 Data Quality Metrics

| Table | Row Count Check | Null Rate | Schema Valid |
|---|---|---|---|
| `SILVER.RISK_HEATMAP_VIEW` | ≥100K rows | <0.5% nulls on GPS columns | PASS |
| `SILVER.WEATHER_ALERTS` | ≥1K rows | <1% nulls on alert_type | PASS |
| `SILVER.BRIDGE_INVENTORY_GEO` | ≥600K rows | 0% nulls on weight_limit/clearance | PASS |
| `SILVER.LOGISTICS_VECTORIZED` | ≥80K rows | 0% null on embedding column | PASS |

### 4.3 Latency Benchmarks

| Operation | p50 | p90 | p99 |
|---|---|---|---|
| CPP Step 3A (Spatial SQL) | 0.3 s | 0.6 s | 1.1 s |
| CPP Step 3B (PEFT + injection) | 4.2 s | 8.5 s | 14.3 s |
| ReMindRAG retrieval (novel query) | 3.8 s | 7.1 s | 12.4 s |
| ReMindRAG retrieval (cached path) | 0.9 s | 1.4 s | 2.1 s |
| Full CPP pipeline (Step 3A + 3B) | 4.5 s | 9.1 s | 15.4 s |
| Snowflake keep-alive first query | 1.8 s | 2.3 s | 3.1 s |

---

## 5. CI Integration

```yaml
# .github/workflows/ci.yml
name: HyperLogistics CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install -r requirements.txt

      # Pipeline smoke test
      - name: Pipeline smoke test
        run: python -m pytest tests/test_pipeline.py -v

      # CPP unit tests (mock Spatial SQL — no Snowflake needed)
      - name: CPP gate unit tests
        run: python -m pytest tests/test_cpp_gate.py -v

      # System evaluation (mock mode — no GPU/API needed)
      - name: System evaluation (mock)
        run: python tests/evaluate_system.py --mode mock

      # ReMindRAG 26-test suite
      - name: ReMindRAG reproducibility suite
        run: |
          python -m pytest tests/test_reproducibility.py -v
          python -m pytest tests/test_benchmark_smoke.py -v
          python -m pytest tests/test_repro_variance.py -v

      # Assert pass rates
      - name: Assert quality gates
        run: |
          python -c "
          import json
          r = json.load(open('eval_results.json'))
          assert r['pass_rate'] >= 0.70, f'CPP pass rate {r[\"pass_rate\"]:.1%} below 70%'
          print(f'PASS: {r[\"pass_rate\"]:.1%} CPP pass rate')
          "

      - uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
```

---

## 6. Evaluation Summary

| Dimension | Metric | Result |
|---|---|---|
| CPP accuracy (50 queries) | Decision correctness | 88–90% (SC-CoT / PEFT+injection) |
| CPP monotonicity | Add violation → VETO | 13/13 PASS (after CPP gate) |
| CPP hallucination rate | Fabricated bridge limits | 8% (down from 40%) |
| CPP CoT quality | 4-step chain completeness | 83% (PEFT) / 100% (SC-CoT) |
| RAG retrieval (LooGLE) | F1 score (5-title subset) | 0.496 (consistent with paper's 0.49) |
| RAG efficiency | API call reduction (repeated) | 37–42% reduction |
| Reproducibility tests | Automated suite | 26/26 PASS (~115 s, zero API cost) |
| Pipeline smoke test | End-to-end health | 7/7 PASS |
| CI integration | Automated on every push | GitHub Actions (≥70% gate) |
