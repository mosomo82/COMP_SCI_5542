# Individual Contribution Statement — Tony Nguyen (Lab 9)

**GitHub:** `mosomo82` / `mtuan`  
**Role:** Data Automation, CPP Agent & Architecture Lead  
**Contribution:** 34%
** Lab9 Repo:** https://github.com/mosomo82/COMP_SCI_5542/tree/main/Week_9

---

## Lab 9 Scope

Tony owned **Phase 1** (Week 6 fixes), and **Phase 4** (Phase 2 CPP agents + pipeline).

---

## Phase 1 — Week 6 Enhancements (`COMP_SCI_5542/Week_6`)

**Primary commit:** `50f2cd4` — _feat(lab9): Implement Week 6 UI enhancements, keep-alive ping, retry hooks, and eval expansion_  
**Secondary commit:** `f9546cc` — _chore(lab9): Finalize evaluation reports and eval_scenarios_  
**Repo:** https://github.com/mosomo82/COMP_SCI_5542

### `streamlit_app.py`

| Enhancement | Detail |
|-------------|--------|
| Persistent sidebar | Project description, data-freshness timestamp, quick-link nav to all 9 tabs |
| Agent Chat redesign | Message history, loading spinner, collapsible tool-call trace expander (name, args, latency) |
| SQL Explorer UX | Input validation + friendly error messages in Executive tab |
| Reset Session button | Clears `st.session_state` and Gemini chat history |

### `sf_connect.py`

| Fix | Detail |
|-----|--------|
| Keep-alive ping | `SELECT 1` on session creation — cold-start latency ~8 s → <2 s |
| Startup assertions | Hard crash at import if any required env var is missing — fails fast before dashboard renders |

### `agent.py`

| Enhancement | Detail |
|-------------|--------|
| `logging` module | Replaced all `print()` with Python logging (INFO default, DEBUG via `LOG_LEVEL`) |
| 6 TOOL SELECTION RULES | Added to `SYSTEM_PROMPT` — fixes S2, S6, S8 multi-tool routing failures |

### `tools.py`

| Fix | Detail |
|-----|--------|
| `get_fleet_performance` | Added `COMBINE WITH get_safety_metrics` for incidents/violations queries |
| `get_fuel_spend_analysis` | Added keyword triggers + `COMBINE WITH get_maintenance_health` |
| `get_maintenance_health` | Added keyword triggers + `COMBINE WITH get_fuel_spend_analysis` |

### `eval_scenarios.py`

| Enhancement | Detail |
|-------------|--------|
| Cooldown increase | `COOLDOWN_SECONDS` 15 → 30 s to prevent cascading 429 rate-limit errors |
| Scenario expansion | 5 → 10 scenarios: S6 (fuel vs maintenance), S7 (monitoring), S8 (driver multi-hop), S9 (adversarial), S10 (implicit multi-tool) |
| S2-A regression fix | Revised S2 phrasing to unified single question; added to regression suite |

### `health_server.py` (NEW)

- FastAPI `/health` endpoint reporting Snowflake connectivity + Gemini API key validity
- Wired to UptimeRobot for continuous uptime monitoring

### `.github/workflows/ci.yml`

- GitHub Actions CI added: runs `eval_scenarios.py` on every push
- Pipeline fails if pass rate < 70%

### Documentation

| File | Location |
|------|----------|
| `Week6_ARCHITECTURE.md` | `COMP_SCI_5542/Week_9/` |
| `Week6_EVALUATION.md` | `COMP_SCI_5542/Week_9/` |

---

## Phase 4 — Project Phase 2 Integration (`CS5542_SmartSC_Optimization_System`)

**Primary commit:** `2caf35d` — _Lab9: update project with enhancement and fixes_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

### `src/utils/snowflake_conn.py`

- Keep-alive ping + `retry_snowflake` exponential-backoff decorator (5 s → 10 s → 20 s)
- Startup assertions — hard crash at import if any env var is missing
- `logging` module throughout (zero `print()`)

### `src/agents/compliance_agent.py` (NEW)

CPP Step 3A Spatial SQL Hard Gate:

```sql
SELECT BRIDGE_ID,
       VERT_CLR_OVER_MT_053  AS clearance_mt,
       OPERATING_RATING_064  AS weight_limit_tons
FROM HYPERLOGISTICS_DB.SILVER.BRIDGE_INVENTORY_GEO
WHERE ST_INTERSECTS(LOCATION, TO_GEOGRAPHY(route_wkt))
  AND (OPERATING_RATING_064 < vehicle_weight
    OR VERT_CLR_OVER_MT_053 < vehicle_height)
```

Returns `ComplianceResult` — **HARD VETO** before any LLM call if any bridge fails.  
Fixes Q13→Q14 monotonicity failure from Lab 8 evaluation.

### `src/agents/cpp_agent.py` — Pipeline Orchestrator with Compliance Gate (NEW)

**Commit:** `2caf35d` — _Lab9: update project with enhancement and fixes_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

- Wires `compliance_agent` as Gate 1 (always first, before any LLM)  
- Gate 2 (Snowflake Cortex) only reached on PASS  
- Returns `CPPDecision` dataclass tracking `verdict`, `compliance`, `response_text`, `llm_called`

---

### `src/agents/context_agent.py` — ReMindRAG Retrieval + Evidence Injection (NEW)

**Commit:** `2caf35d` — _Lab9: update project with enhancement and fixes_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

- Two-path retrieval from `SILVER.LOGISTICS_VECTORIZED`:  
  - **Primary:** `VECTOR_COSINE_SIMILARITY` + `EMBED_TEXT_768` (seed=42 tiebreaker)  
  - **Fallback:** `TEXT_CONTENT ILIKE '%keyword%'` when EMBEDDING not populated  
- `build_constrained_prompt()` injects `[RETRIEVED CONSTRAINTS]` block into LLM prompt  
- Reduces hallucination rate from ~40% → 8% (evidence-grounded generation)

---

### `src/run_pipeline.py` — Logging Refactor

**Commit:** `2caf35d` — _Lab9: update project with enhancement and fixes_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

- All `print()` → `logging` (INFO/WARNING/DEBUG)
- Respects `LOG_LEVEL` env var — zero `print()` calls (AST-verified)

### `src/verify_pipeline.py` — Logging + Schema Assertions

**Commit:** `2caf35d` — _Lab9: update project with enhancement and fixes_  
**Repo:** https://github.com/mosomo82/CS5542_SmartSC_Optimization_System

- All `print()` → `logging`
- `_assert_silver_schema()` — queries `INFORMATION_SCHEMA.COLUMNS`, raises `AssertionError` for missing columns
- 4 SILVER tables validated: `RISK_HEATMAP_VIEW`, `BRIDGE_INVENTORY_GEO`, `CLEANED_LOGISTICS`, `LOGISTICS_VECTORIZED`
- `sys.exit(1)` on failure for CI integration

--- 

## Deliverables — COMP_SCI_5542/Week_6

**Commit:** `da4c5d8` — _Merge pull request #14 from mosomo82/lab9-enhancements_  
**Repo:** https://github.com/mosomo82/COMP_SCI_5542

| File | Enhancement |
|------|-------------|
| `streamlit_app.py` | Persistent sidebar, Agent Chat trace expander, Reset Session, SQL Explorer UX |
| `sf_connect.py` | Keep-alive ping + startup assertions |
| `agent.py` | `logging` module, 6 TOOL SELECTION RULES in SYSTEM_PROMPT |
| `tools.py` | COMBINE WITH keyword triggers for multi-tool scenarios |
| `eval_scenarios.py` | 5 → 10 scenarios, 30s cooldown, S2-A regression fix |
| `health_server.py` | FastAPI `/health` endpoint |
| `.github/workflows/ci.yml` | GitHub Actions CI with 70% pass-rate gate |

---

## Tests Delivered

### `tests/test_cpp_gate.py` (NEW — CS5542_SmartSC repo)

**4 unit tests — all mock-based, zero Snowflake connection required:**

| Test | Input | Expected |
|------|-------|----------|
| `test_overweight_veto` | 40t vehicle, 30t bridge limit | `HARD_VETO` ✅ |
| `test_height_veto` | 4.5m vehicle, 3.8m clearance | `HARD_VETO` ✅ |
| `test_compliant_pass` | All bridges within limits | `PASS` ✅ |
| `test_no_bridges_on_route_pass` | No bridge intersections | `PASS` ✅ |

**Result: 4 passed in 0.17s ✅**

### Documentation

| File | Location |
|------|----------|
| `Week6_ARCHITECTURE.md` | `COMP_SCI_5542/Week_9/` |
| `Week6_EVALUATION.md` | `COMP_SCI_5542/Week_9/` |
| `P2_ARCHITECTURE.md` | `COMP_SCI_5542/Week_9/` |
| `P2_EVALUATION.md` | `COMP_SCI_5542/Week_9/` |

---

## Commit Summary

| Repo | Commit | Description |
|------|--------|-------------|
| COMP_SCI_5542 | `832d953` | Update (HEAD) |
| COMP_SCI_5542 | `50f2cd4` | **feat(lab9): Week 6 UI enhancements, keep-alive, retry, eval expansion** |
| COMP_SCI_5542 | `f9546cc` | chore(lab9): Finalize evaluation reports and eval_scenarios |
| COMP_SCI_5542 | `d155677` | Update Week6 |
| COMP_SCI_5542 | `da4c5d8` | Merge PR #14 lab9-enhancements |
| CS5542_SmartSC | `62b9947` | Merge branch 'main' (HEAD) |
| CS5542_SmartSC | `2caf35d` | **Lab9: update project with enhancement and fixes** |
| CS5542_SmartSC | `673d815` | System Implementation (Phase 2 base) |
| CS5542_SmartSC | `d0e0ec7` | Environment / requirements setup |

---

**34%** — Data pipeline automation, CPP agent architecture, Snowflake connectivity, compliance gate, ReMindRAG integration, logging modernisation, SILVER schema verification, unit tests, and Lab 6 eval/UI fixes.

---