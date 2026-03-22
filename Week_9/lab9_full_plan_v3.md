# Lab 9 Full Plan — 62 tasks across 5 phases

**Parallel owners:** Tony → Phases 1+3 eval fixes + Phase 4 pipeline/CPP agents · Daniel → Phase 4 dashboard · Joel → Phase 4 tests + CI · All three → Phase 2 benchmark runs (~$0.70 total). Phase 5 starts only after all CI is green.

**Already done:** agent.py + tools.py (Phase 1) · all 8 doc files · both PDF reports (draft — needs real commit links).

---

## Phase 1 — Week 6 fixes `COMP_SCI_5542/Week_6` (18 tasks)

### streamlit_app.py

- [x] Add persistent sidebar: project description, data-freshness, quick-link nav to all 9 tabs `code`
- [x] Redesign Agent Chat: message history, loading spinner, collapsible tool-call trace expander (name, args, latency) `code`
- [x] Add input validation + friendly error messages to SQL Explorer in Executive tab `code`
- [x] Add Reset Session button (clears st.session_state and Gemini history) `code`

### sf_connect.py

- [x] Add Snowflake keep-alive ping on startup — cold-start ~8s → <2s `fix`
- [x] Add startup assertions validating all env variables before dashboard renders `fix`

### agent.py ✅ fixed file generated

- [x] Replace print() with Python logging module (INFO default, DEBUG via LOG_LEVEL) `code`
- [x] Update SYSTEM_PROMPT with 6 TOOL SELECTION RULES — fixes S2, S6, S8 failures `fix`

### tools.py ✅ fixed file generated

- [x] get_fleet_performance: add COMBINE WITH get_safety_metrics for incidents/violations queries `fix`
- [x] get_fuel_spend_analysis: add keyword triggers + COMBINE WITH get_maintenance_health `fix`
- [x] get_maintenance_health: add keyword triggers + COMBINE WITH get_fuel_spend_analysis `fix`

### eval_scenarios.py

- [x] Increase COOLDOWN_SECONDS 15 → 30 to prevent cascading 429s `fix`
- [x] Expand 5 → 10 scenarios: S6 (fuel vs maintenance), S7 (monitoring), S8 (driver multi-hop), S9 (adversarial), S10 (implicit multi-tool) `test`
- [x] Revise S2 phrasing → S2-A (unified question, not two sentences); add to regression suite `fix`

### health_server.py (new)

- [x] Create FastAPI /health endpoint reporting Snowflake + Gemini key validity; wire to UptimeRobot `code`

### .github/workflows/ci.yml

- [x] Add GitHub Actions CI: runs eval_scenarios.py on every push, fails if pass rate <70% `test`

### Week_9/ docs ✅ files generated

- [x] Commit ARCHITECTURE.md to Week_9/ `doc`
- [x] Commit EVALUATION.md to Week_9/ `doc`

---

## Phase 2 — Week 7 fixes `ReMindRAG_Week7` (7 tasks)

### tests/ (new files)

- [ ] Create test_benchmark_smoke.py: 5 tests — pipeline, chunk retrieval, keyword overlap, answer quality, eval schema (mock LLM, zero API cost) `test`
- [ ] Create test_repro_variance.py: 4 tests — same chunks, count, nodes, deterministic answer (all-MiniLM-L6-v2, no HF token) `test`

### eval/ (constrained benchmark run ~$0.70 total)

- [ ] Run eval_LooGLE.py — 5-title subset, --seed 42, --judge_model_name gpt-4o-mini. Record F1 in eval_results.json `test`
- [ ] Run eval_Hotpot.py — 50 questions, --seed 42, --judge_model_name gpt-4o-mini. Record F1 `test`

### .github/workflows/ci.yml (update)

- [ ] Add new test files to CI — suite expands 17 → 26 tests `test`

### Root docs ✅ files generated (Phase 2)

- [ ] Commit BENCHMARKING.md (paper-vs-repro comparison, constrained run results, variance table) `doc`
- [ ] Commit REPRO_VERIFICATION.md (26-test suite, gap analysis, CI YAML) `doc`

---

## Phase 3 — Week 8 fixes `COMP_SCI_5542/Week_8` (9 tasks)

### sf_connect.py / CPP hard gate (most critical)

- [ ] Spatial SQL hard gate before any LLM call: MIN(weight_limit_tons / vertical_clearance_mt) via ST_INTERSECTS → HARD VETO if exceeded. Fixes Q13→Q14 monotonicity failure. `fix`

### adaption_method/prompt_adaptation.py

- [ ] Add [RETRIEVED CONSTRAINTS] block to all prompt templates — bridge limits injected from SQL, model not to recall from memory. Fixes hallucination (~40% → 8%). `fix`

### data/generate_dataset.py

- [ ] Update output format: full 4-step CoT (Disruption→Route→Constraint→Decision). Expand 100 → 300+ pairs, 5 disruption types. `code`
- [ ] Re-run QLoRA fine-tuning on Colab T4 with expanded CoT dataset. Save new adapter. `code`

### app/evaluation.py

- [ ] Replace binary scorer with 5-dim rubric (Decision, Grounding, Constraint, CoT, Jargon — 0–10, pass ≥7) `fix`
- [ ] Expand eval set 15 → 50 queries, 5 disruption categories. Metamorphic tests 3 → 13 pairs. `test`

### .github/workflows/lab8_eval.yml (new)

- [ ] GitHub Actions CI: evaluation.py mock mode on every push, fails if pass rate <70% `test`

### Week_8/ docs ✅ files generated

- [ ] Commit ARCHITECTURE.md to Week_8/ (4-layer CPP, failure mode analysis) `doc`
- [ ] Commit EVALUATION.md to Week_8/ (corrected accuracy, rubric, 50-query design) `doc`

---

## Phase 4 — Phase 2 project integration `CS5542_SmartSC_Optimization_System` (19 tasks)

### src/utils/snowflake_conn.py (Tony)

- [x] Port keep-alive ping + retry decorator from Lab 6 fixes `fix`
- [x] Add startup assertions validating all env variables at import time `fix`

### src/agents/compliance_agent.py (Tony — new)

- [x] Implement CPP Step 3A Spatial SQL hard gate: MIN(weight_limit_tons / vertical_clearance_mt) via ST_INTERSECTS → HARD VETO before any LLM call `fix`
- [x] Wire compliance_agent.py into cpp_agent.py as first gate in pipeline `code`

### src/agents/context_agent.py (Tony)

- [x] Wire ReMindRAG retrieval — pull from SILVER.LOGISTICS_VECTORIZED with seed=42 `code`
- [x] Add [RETRIEVED CONSTRAINTS] evidence injection block to prompt template `fix`

### src/run_pipeline.py + verify_pipeline.py (Tony)

- [x] Replace print() with Python logging module throughout both files `code`
- [x] Add row-count + schema assertions to verify_pipeline.py for all 4 SILVER tables `test`

### src/app/dashboard.py (Daniel)

- [ ] Add persistent sidebar + collapsible Reasoning Path expander (ReMindRAG steps + CPP decisions) `code`
- [ ] Merge Lab 6 analytics tabs (fleet, routes, fuel, safety) into unified dashboard `code`
- [ ] Apply retry decorator to all Cortex/external API calls. Add Reset Session button. `fix`

### tests/ (Joel)

- [ ] test_cpp_gate.py: unit tests (overweight→VETO, compliant→PASS, height violation→VETO) `test`
- [ ] test_pipeline.py: end-to-end smoke (Snowflake connectivity, SILVER row counts, HTTP 200) `test`
- [ ] Expand evaluate_system.py: 50 queries, 5-dim rubric, mock mode for CI `test`

### .github/workflows/ci.yml (Joel — new)

- [ ] Unified CI: pipeline smoke + CPP unit + system eval mock + ReMindRAG 26-tests + ≥70% gate `test`

### Root docs ✅ files generated (Phase 4)

- [ ] Commit ARCHITECTURE.md to repo root (4-layer system, CPP detail, data medallion) `doc`
- [ ] Commit EVALUATION.md to repo root (50-query rubric, RAG benchmarks, pipeline smoke) `doc`
- [ ] Update CONTRIBUTIONS.md: add Lab 9 contributions for all three teammates `doc`
- [ ] Update README.md: add CI badge, Lab 9 changelog, links to ARCHITECTURE.md + EVALUATION.md `doc`

---

## Phase 5 — Lab 9 report (write last — after all CI green) (9 tasks)

### Pre-flight

- [ ] Verify CI badges green on all 3 repos: COMP_SCI_5542, ReMindRAG_Week7, CS5542_SmartSC `test`
- [ ] Confirm all 3 live apps up: cs5542logisticsai · cs5542lab8 · cs5542hyperlogistics .streamlit.app `test`
- [ ] Take screenshots: Lab 6 sidebar + tool trace, Lab 8 CPP evidence panel, Phase 2 reasoning path expander `doc`

### Group Development Report ✅ PDF generated

- [ ] Sections 1–3: overview, feedback addressed (Labs 6+7), all enhancements incl. Phase 2 integration `doc`
- [ ] Section 4: before/after table across Labs 6, 7, 8, and Phase 2 `doc`
- [ ] Section 5: all GitHub + deployment links (update with real commit links once pushed) `doc`

### Individual Contribution Report ✅ PDF generated

- [ ] Fill in real GitHub commit hashes from all 3 repos once code is pushed `doc`
- [ ] Verify Phase 2 contributions listed: compliance_agent.py, context_agent.py, snowflake_conn.py, ARCHITECTURE.md, EVALUATION.md `doc`
- [ ] Submit both PDFs to Canvas `doc`
