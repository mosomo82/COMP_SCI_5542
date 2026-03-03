# Task 4 — Agent Evaluation Report

> **Date:** 2026-03-03  
> **Model:** `gemini-2.5-flash` (Pro tier)  
> **Tools available:** 7 (`query_snowflake`, `get_monthly_revenue`, `get_fleet_performance`, `get_pipeline_logs`, `get_safety_metrics`, `get_route_profitability`, `get_delivery_performance`)  
> **Evaluation harness:** [`eval_scenarios.py`](eval_scenarios.py) — manual agentic loop with per-step instrumentation, 15 s inter-scenario cooldown, exponential-backoff retry on 429 errors  
> **Raw results:** [`eval_results.json`](eval_results.json)

---

## 1. Evaluation Scenarios & User Queries

### S1 — Simple (Single Tool)

| Field | Value |
|-------|-------|
| **Query** | *"Show me the monthly revenue from January 2024 to June 2024."* |
| **Complexity** | Low — requires one data retrieval call |
| **Expected Tool(s)** | `get_monthly_revenue` |
| **Expected Steps** | 1 |

### S2 — Medium (Multiple Tools)

| Field | Value |
|-------|-------|
| **Query** | *"Which are our top 5 performing diesel trucks by revenue, and do any of those drivers have safety incidents on record? Provide both the fleet data and the safety data."* |
| **Complexity** | Medium — requires cross-referencing fleet and safety data |
| **Expected Tool(s)** | `get_fleet_performance`, `get_safety_metrics` |
| **Expected Steps** | 2–3 |

### S3 — Complex (Reasoning & Synthesis)

| Field | Value |
|-------|-------|
| **Query** | *"I need a strategic analysis: compare our top 10 most profitable routes against delivery performance in the same cities. Are our most profitable routes also the most reliable in terms of on-time delivery? Identify any routes where high profit coincides with poor on-time rates, and recommend corrective actions."* |
| **Complexity** | High — requires two data pulls, cross-city matching, gap identification, and strategic recommendations |
| **Expected Tool(s)** | `get_route_profitability`, `get_delivery_performance` |
| **Expected Steps** | 2–4 |

---

## 2. Tools Used

| Scenario | Expected Tools | Actually Called | Match? |
|----------|---------------|----------------|--------|
| **S1** | `get_monthly_revenue` | `get_monthly_revenue` | ✅ Exact match |
| **S2** | `get_fleet_performance`, `get_safety_metrics` | `get_fleet_performance` only | ❌ Missing `get_safety_metrics` |
| **S3** | `get_route_profitability`, `get_delivery_performance` | `get_route_profitability`, `get_delivery_performance` | ✅ Exact match |

**Tool Selection Accuracy:** 4 / 5 expected tool calls were made correctly (**80%**).

---

## 3. Number of Reasoning Steps

| Scenario | Expected Steps | Actual Steps | Assessment |
|----------|---------------|-------------|------------|
| **S1** | 1 | **1** | ✅ Optimal — single call, no unnecessary follow-ups |
| **S2** | 2–3 | **1** | ⚠️ Under-stepped — model stopped after one tool call |
| **S3** | 2–4 | **2** | ✅ Both tools called in parallel in a single round |

**Observation:** The model parallelized the two S3 tool calls into a single round rather than calling them sequentially — an efficient strategy. In S2, it made a pragmatic (but incorrect per our expectations) decision to skip the second tool.

---

## 4. Accuracy Assessment

### S1 — Simple ✅ PASS

The agent correctly called `get_monthly_revenue` with `start_month='2024-01-01'` and `end_month='2024-06-01'` and produced:

| Month | Revenue |
|-------|---------|
| Jan 2024 | $7,386,201.05 |
| Feb 2024 | $6,970,220.94 |
| Mar 2024 | $7,453,599.00 |
| Apr 2024 | $7,122,948.73 |
| May 2024 | $7,569,586.95 |
| Jun 2024 | $7,489,030.25 |

**Verdict:** Fully accurate. Data matches the Snowflake view. Response is professional and well-structured.

### S2 — Medium ⚠️ PARTIAL FAIL

The agent correctly retrieved the top 5 diesel trucks:

| Rank | Truck ID | Revenue | Make | Year |
|------|----------|---------|------|------|
| 1 | TRK00039 | $109,876.64 | International | 2015 |
| 2 | TRK00086 | $96,868.67 | Freightliner | 2015 |
| 3 | TRK00055 | $92,376.97 | Peterbilt | 2017 |
| 4 | TRK00024 | $85,325.56 | Volvo | 2015 |
| 5 | TRK00026 | $81,133.97 | International | 2021 |

However, the model **did not call `get_safety_metrics`**. It stated:

> *"The fleet performance data does not include driver IDs or names, making it impossible to directly check their safety records using the available tools."*

**Analysis:** The `V_TRIP_PERFORMANCE` view does return `DRIVER_NAME`, but the model either did not inspect the returned data carefully enough or lacks confidence in matching by name rather than ID. This is a **context-comprehension weakness** — the model had the data it needed but failed to recognise it.

**Verdict:** Fleet data accurate; safety cross-reference skipped due to agent reasoning error.

### S3 — Complex ✅ PASS

The agent called both tools and produced a rich strategic synthesis:

1. **Route profitability data** — top 10 routes with gross profit and margins
2. **Cross-city delivery matching** — on-time delivery % for each destination city
3. **Key finding:** All top-10 profitable routes deliver to cities with < 46% on-time rates
4. **Recommendations:** route/schedule optimization, last-mile efficiency, driver incentive programs, enhanced monitoring

Sample output:

| Route | Gross Profit | Destination On-Time % |
|-------|--------------|-----------------------|
| Seattle → Charlotte | $25,372,032.88 | 44.2% |
| Charlotte → Portland | $24,905,890.97 | 45.5% |
| Philadelphia → Seattle | $23,890,335.41 | 44.0% |

**Verdict:** Fully accurate. Correct tools, strong synthesis, actionable recommendations provided.

---

## 5. Latency Observations

| Scenario | Total Latency | Breakdown |
|----------|--------------|-----------|
| **S1** | **7.11 s** | ~3.5 s Snowflake query + ~3.6 s model inference |
| **S2** | **16.38 s** | ~3.0 s Snowflake query + ~13.4 s model inference (longer response generation) |
| **S3** | **27.70 s** | ~7.0 s (2 Snowflake queries) + ~20.7 s model inference (complex synthesis) |
| **Total** | **51.19 s** | Across all 3 scenarios (+ 30 s cooldown between scenarios) |

### Key Latency Observations

1. **Snowflake queries** take 2.9–4.6 s each, consistent with pipeline logs showing "Moderate latency (2–5 s)" for ad-hoc queries.
2. **Model inference scales with answer complexity:** S1 (short answer) ≈ 3.6 s, S2 (medium) ≈ 13.4 s, S3 (synthesis with tables & recommendations) ≈ 20.7 s.
3. **Parallel tool calls** in S3 are more efficient than sequential multi-step calls — both data fetches happened in one round.
4. **Inter-scenario cooldown** (15 s) successfully prevents 429 rate-limit errors.

---

## 6. Failure Cases and Analysis

### Failure 1: S2 — Model did not call `get_safety_metrics`

| Attribute | Detail |
|-----------|--------|
| **Type** | Agent reasoning error |
| **Severity** | Medium |
| **Root Cause** | The model incorrectly claimed that driver names were absent from fleet results, even though `DRIVER_NAME` was returned. It prematurely decided it could not cross-reference. |
| **Impact** | User received incomplete data — fleet metrics without the requested safety cross-reference. |
| **Mitigations** | (1) Strengthen system prompt: *"If the user asks about multiple data domains, you MUST call a tool for each domain."* (2) Add a `driver_name` parameter to `get_safety_metrics` to make cross-referencing more explicit. (3) Add a `query_snowflake` fallback prompt so the agent can self-correct by writing a custom join query. |

### Summary of Failure Modes

| # | Scenario | Failure Type | Recoverable? | Fix Complexity |
|---|----------|-------------|--------------|----------------|
| 1 | S2 | Reasoning / tool selection | Yes (prompt engineering) | Low |

---

## 7. Overall Summary

| Metric | S1 (Simple) | S2 (Medium) | S3 (Complex) |
|--------|-------------|-------------|--------------|
| **Result** | ✅ PASS | ⚠️ PARTIAL | ✅ PASS |
| **Tool Accuracy** | 1/1 correct | 1/2 correct | 2/2 correct |
| **Reasoning Steps** | 1 (optimal) | 1 (insufficient) | 2 (optimal) |
| **Latency** | 7.11 s | 16.38 s | 27.70 s |
| **Error** | None | Reasoning skip | None |

**Overall Pass Rate:** 2/3 scenarios fully passed.  
**Tool Selection Accuracy:** 4/5 expected tools were correctly called (80%).  
**Agent Reasoning:** Correct in S1 and S3; S2 shows a multi-step reasoning weakness where the model stops after one tool when it should continue.

### Recommendations

1. **Prompt hardening** — Add explicit multi-tool instructions to the system prompt to prevent S2-style early termination.
2. **Tool schema enrichment** — Add `driver_name` parameter to safety tools to guide the model toward cross-referencing.
3. **Retry logic** — The evaluation harness now includes exponential backoff for 429 errors (already implemented).
4. **Inter-scenario cooldown** — 15-second pauses between scenarios prevent rate-limit failures (already implemented).
