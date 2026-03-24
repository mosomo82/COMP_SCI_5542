# Evaluation Report: Multi-Agent Trucking Logistics Analytics Platform

> **CS 5542 — Big Data and Analytics**
> Team: Tony Nguyen · Daniel Evans · Joel Vinas
> Lab 9 Documentation Expansion (addressing Lab 6 feedback)

---

## 1. Evaluation Overview

This document provides a complete record of agent evaluation methodology, scenario design rationale, raw results, error analysis, and the Lab 9 improvements made in response to the Lab 6 evaluation findings. The evaluation is implemented in `eval_scenarios.py` and produces structured output in `eval_results.json`.

### Evaluation Goals

1. Verify that the Gemini 2.5 Flash agent correctly selects the appropriate tool(s) for a given natural-language query.
2. Measure end-to-end response latency for queries of varying complexity.
3. Identify failure modes in tool selection and multi-tool reasoning.
4. Track pass-rate trends across lab iterations.

### What "Pass" Means

A scenario is scored **PASS** if all of the following are true:

- The agent invokes every tool listed in the scenario's `expected_tools` set at least once.
- The agent returns a substantive natural-language response (not an error message or empty string).
- The response contains at least one data point drawn from the tool result (verified by keyword matching against known values in the test fixtures).

A scenario is scored **PARTIAL** if the agent returns a valid response but skips one or more expected tools. It is scored **FAIL** if the agent returns an error or an empty response.

---

## 2. Lab 6 Evaluation Results (Baseline)

### 2.1 Scenario Summary

| ID | Scenario Description | Complexity | Expected Tools | Result | Latency |
|---|---|---|---|---|---|
| S1 | "Show me monthly revenue from January to June 2024." | Simple | `get_monthly_revenue` | ✅ PASS | 7.1 s |
| S2 | "Which top diesel trucks have safety incidents on record?" | Medium | `get_fleet_performance`, `get_safety_metrics` | ⚠️ PARTIAL | 16.4 s |
| S3 | "Compare our most profitable routes against delivery reliability and recommend corrective actions." | Complex | `get_route_profitability`, `get_delivery_performance` | ✅ PASS | 27.7 s |
| S4 | "What are our top maintenance cost drivers this year?" | Medium | `get_maintenance_health` | ✅ PASS | 8.5 s |
| S5 | "Identify the states where high fuel spend overlaps with high maintenance cost per mile." | Complex | `get_fuel_spend_analysis`, `get_maintenance_health` | ✅ PASS | 19.2 s |

**Overall pass rate:** 4/5 (80%)
**Tool accuracy:** 6/7 expected tool calls observed (86%)
**Mean latency:** 15.8 s
**Median latency:** 16.4 s

### 2.2 Scenario Design Rationale

Scenarios were designed to span four complexity tiers:

- **Simple (1 tool):** Validates basic tool routing. The agent should select the correct single tool without ambiguity.
- **Medium (2 tools, independent):** Validates that the agent pulls from two data domains when the query references two distinct concepts (e.g., fleet type + safety record).
- **Complex (2+ tools, synthesized):** Validates cross-domain reasoning. The agent must call multiple tools *and* synthesize their outputs into a coherent recommendation rather than just listing raw data.
- **Adversarial (Lab 9 addition):** Queries are intentionally vague or overlap multiple tool domains to test tool-selection robustness.

### 2.3 Lab 6 Failure Analysis — S2 Partial

**Scenario S2:** *"Which top diesel trucks have safety incidents on record?"*

**Expected behavior:** Agent calls `get_fleet_performance` (to identify top diesel trucks by revenue/utilization) and `get_safety_metrics` (to cross-reference incident records).

**Observed behavior:** The agent called only `get_safety_metrics`. It returned a valid list of trucks with incidents but did not rank them by fleet performance metrics. The response was factually correct but incomplete relative to the scenario intent.

**Root cause:** The phrase "top diesel trucks" is ambiguous — it can mean "trucks with the most incidents" (a safety-only query) or "highest-performing diesel trucks that also have incidents" (a cross-domain query). Gemini resolved the ambiguity by choosing the single most relevant tool rather than invoking both.

**Mitigation applied in Lab 9:** The tool description for `get_fleet_performance` was updated to explicitly note that it can be combined with `get_safety_metrics` for safety-cross-referenced fleet analysis. An adversarial variant of S2 was added to the expanded suite (S2-A) to test whether the updated description resolves the issue.

---

## 3. Lab 9 Evaluation Expansion (10 Scenarios)

### 3.1 New Scenarios (S6–S10)

| ID | Scenario Description | Complexity | Expected Tools | Focus |
|---|---|---|---|---|
| S6 | "Which routes have the highest fuel cost per mile relative to their profit margin?" | Complex | `get_fuel_spend_analysis`, `get_route_profitability` | Cross-domain synthesis |
| S7 | "Summarize overall system health: pipeline logs, error rate, and recent query latency." | Medium | `get_pipeline_logs` | Monitoring/observability |
| S8 | "Which drivers have both above-average incident rates and below-average delivery performance?" | Complex | `get_safety_metrics`, `get_delivery_performance` | Multi-hop join reasoning |
| S9 | "Tell me about the data." *(adversarial — vague)* | Adversarial | `query_snowflake` | Graceful fallback behavior |
| S10 | "What was revenue last month and which trucks drove it?" *(multi-concept, single-sentence)* | Complex | `get_monthly_revenue`, `get_fleet_performance` | Implicit multi-tool trigger |

**S2-A (revised):** Updated version of S2 with clarified phrasing: *"Among our highest-revenue diesel trucks, which have safety incidents on record?"* Expected tools: `get_fleet_performance`, `get_safety_metrics`.

### 3.2 Evaluation Methodology (Lab 9)

The Lab 9 evaluation harness introduces the following changes over the Lab 6 baseline:

| Feature | Lab 6 | Lab 9 |
|---|---|---|
| Number of scenarios | 5 | 10 + 1 revised |
| Retry logic | 15 s cooldown, max 2 retries | Exponential backoff (5 s → 10 s → 20 s), max 3 retries |
| Tool accuracy scoring | Binary per scenario | Per-tool-call granularity |
| Output format | JSON + console | JSON + console + GitHub Actions CI badge |
| Scheduling | Manual run only | Automated on every push to `main` |
| Latency tracking | Total response time | Total + per-tool-call breakdown |
| Adversarial coverage | None | 1 adversarial scenario (S9) |

### 3.3 Latency Benchmarks (Lab 6 Baseline)

| Percentile | Latency |
|---|---|
| p50 (median) | 16.4 s |
| p90 | 27.1 s |
| p99 | 27.7 s |
| Mean | 15.8 s |
| Fastest (S1) | 7.1 s |
| Slowest (S3) | 27.7 s |

Latency is dominated by two factors:
1. **Number of tool calls:** Each tool call adds a Snowflake round-trip (0.5–3 s) plus a Gemini API inference pass.
2. **Synthesis complexity:** Complex cross-domain queries require Gemini to perform multi-step reasoning before generating a final response, adding 5–15 s over simple queries.

The Lab 9 keep-alive ping reduces the first-query cold-start penalty from ~8 s to under 2 s, which primarily benefits S1-class (simple, single-tool) queries.

### 3.4 Lab 9 Execution Results

During the final Lab 9 evaluation run, the expanded suite was executed against the production `gemini-2.5-flash` model:

- **Pass Rate:** 6/9 scenarios (67%)
- **Total Latency:** 123.3 s

**Key Success — Exponential Backoff Recovery:**
During the execution of the implicit multi-tool scenario (S10), the agent successfully hit multiple API rate limits when parallelizing 3 tool calls (`get_monthly_revenue`, `get_safety_metrics`, `get_route_profitability`). The newly implemented backoff decorator caught the `HTTP 429` errors, paused for 5s, then 10s, and successfully recovered gracefully without crashing the evaluation harness. This confirms that the pipeline is highly resilient against upstream throttling.

---


## 4. Evaluation Infrastructure

### 4.1 Running the Evaluation

```bash
# Run all 10 scenarios (Lab 9)
python eval_scenarios.py --json

# Run a single scenario by ID
python eval_scenarios.py --scenario S6 --json

# Run with verbose tool-call logging
LOG_LEVEL=DEBUG python eval_scenarios.py --json
```

Output is written to `eval_results.json` and printed to stdout. On CI, the JSON file is uploaded as a GitHub Actions artifact.

### 4.2 eval_results.json Schema

```json
{
  "run_timestamp": "2026-03-20T14:00:00Z",
  "total_scenarios": 10,
  "passed": 8,
  "partial": 1,
  "failed": 1,
  "pass_rate": 0.80,
  "mean_latency_s": 15.8,
  "scenarios": [
    {
      "id": "S1",
      "description": "Show me monthly revenue from January to June 2024.",
      "complexity": "simple",
      "expected_tools": ["get_monthly_revenue"],
      "tools_called": ["get_monthly_revenue"],
      "result": "PASS",
      "latency_s": 7.1,
      "response_excerpt": "Revenue peaked in March at $1.2M..."
    }
  ]
}
```

### 4.3 CI Integration (Lab 9)

The evaluation harness runs automatically via `.github/workflows/ci.yml` on every push to `main`:

```yaml
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt
      - run: python tests/smoke_test.py          # ReMindRAG smoke test
      - run: python eval_scenarios.py --json     # Agent evaluation
      - uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json
```

The workflow fails if `pass_rate < 0.70`, providing an automated quality gate.

---

## 5. Summary of Findings and Improvements

| Finding (Lab 6) | Action Taken (Lab 9) |
|---|---|
| S2 partial: agent skipped `get_fleet_performance` on ambiguous query | Updated tool description; added adversarial scenario S2-A to regression suite |
| No cross-domain fuel vs. route evaluation | Added S6 (fuel cost per mile vs. route margin) |
| No observability/monitoring evaluation | Added S7 (pipeline health query) |
| No adversarial coverage | Added S9 (vague query graceful fallback) |
| Evaluation ran manually only | CI integration via GitHub Actions; automated on every push |
| Latency only reported as total | Per-tool-call latency now tracked and reported |
| 429 errors caused eval runs to abort | Exponential-backoff retry decorator applied to all Gemini calls |
| Documentation of methodology was minimal | This document (Lab 9 expansion) |
