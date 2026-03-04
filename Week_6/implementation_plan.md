# Lab 6 Implementation Plan — Team EVN

> **Team:** Tony Nguyen (`mosomo82`), Daniel Evans (`devans2718`), Joel Vinas (`joelvinas`)  
> **Repository:** `mosomo82/COMP_SCI_5542` → `Week_6/`

> [!NOTE]
> **Last updated:** 2026-03-02 by Daniel Evans  
> **Status:** Phase 2 (Daniel) ✅ complete — PR open on `daniel-lab6` → `main`.  
> Phase 2 (Joel) ⏳ pending. Phase 3 blocked until both Phase 2 PRs merge.

---

## Workspace Audit Checklist (2026-03-03)

### ✅ Completed in this folder

- [x] Tony Phase 1 foundation exists: `tools.py` (core 5), `tool_schemas.py` (core 5), `agent.py`, `test_tools.py`, `test_agent.py`, `task1_antigravity_report.md`
- [x] Daniel Phase 2 tools are implemented in `tools.py`: `get_route_profitability`, `get_delivery_performance`
- [x] Daniel Phase 2 schemas are implemented in `tool_schemas.py` for both new tools
- [x] Joel Phase 2 tools are implemented in `tools.py`: `get_maintenance_health`, `get_fuel_spend_analysis`
- [x] Joel tool schemas are implemented in `tool_schemas.py` (verified and fixed)
- [x] `agent.py` updated to register all 9 tools and enhanced system prompt
- [x] `task4_evaluation_report.md` exists (covers first 7 tools)
- [x] Streamlit AI tab is implemented in `app/streamlit_app.py`
- [x] Individual contribution files exist in `reports/` (`CONTRIBUTIONS_JV.md`, `CONTRIBUTIONS_TN.md`)

### ⏳ Missing / Incomplete in this folder

- [ ] `task4_evaluation_report.md` needs update to include Joel's new tools (8-9)
- [ ] `test_tools.py` currently validates through tool 7 only (no tests for tools 8–9)
- [ ] `test_agent.py` binds only one tool (`get_monthly_revenue`), not full toolset
- [ ] `README.md` is still Week 5 focused and does not document final Lab 6 (9-tool) state

### 🔎 Not verifiable from local folder only

- [ ] PR count / merge status in GitHub (requires remote repository check)

---

## Division of Labor Summary

| Member | Tools | Streamlit Chat (shared) | Other Deliverables |
|---|---|---|---|
| **Tony** | 5 existing ✅ | Agent session + wiring | `task1_antigravity_report.md` ✅, tests ✅, register new tools ⬅️ **ACTION NEEDED** |
| **Daniel** | 2 new ✅ | Chat UI + history | `task4_evaluation_report.md`, demo video |
| **Joel** | 2 new ⏳ | Tool logs + formatting | `README.md`, `CONTRIBUTIONS.md`, screenshots |

> After all work: **9 total tools** in `tools.py`

---

## Phase 0: Git Setup (All Members)

```bash
git clone https://github.com/mosomo82/COMP_SCI_5542.git
cd COMP_SCI_5542 && git checkout main && git pull origin main

# Create your branch:
git checkout -b tony/streamlit-agent-ui        # Tony
git checkout -b daniel/route-delivery-tools    # Daniel
git checkout -b joel/maintenance-fuel-tools    # Joel
```

Local environment:
```bash
cd Week_6
cp .env.example .env   # Fill in SNOWFLAKE_*, GEMINI_API_KEY
pip install -r ../requirements.txt
pip install google-generativeai python-dotenv folium
python test_tools.py   # Verify tools work
python test_agent.py   # Verify agent binding
```

---

## Phase 1: Tony's Foundation (✅ Already Completed)

Tony built the entire agent infrastructure that Daniel and Joel build upon:

| Deliverable | File | Status |
|---|---|---|
| 5 agent tools | `tools.py` | ✅ Done |
| Tool schemas (OpenAI format) | `tool_schemas.py` | ✅ Done |
| Gemini agent with CLI loop | `agent.py` | ✅ Done |
| System prompt + auto function calling | `agent.py` | ✅ Done |
| Antigravity IDE report | `task1_antigravity_report.md` | ✅ Done |
| Tool smoke tests | `test_tools.py` | ✅ Done |
| Agent binding validation | `test_agent.py` | ✅ Done |

**Existing tools built by Tony:**
1. `query_snowflake` — arbitrary read-only SQL
2. `get_monthly_revenue` — revenue trends by date range
3. `get_fleet_performance` — truck metrics with filters
4. `get_pipeline_logs` — system health and latency
5. `get_safety_metrics` — safety incident analytics

---

## Phase 2: New Tools (Parallel — Daniel & Joel)

### 👤 Daniel Evans — 2 New Tools ✅ COMPLETE

> Implemented 2026-03-02 on branch `daniel-lab6`. PR open → `main`.  
> `python test_tools.py` exits 0. All 7 tools verified.

#### Tool 6: `get_route_profitability` ✅
Queries `V_ROUTE_SCORECARD` for route-level profit margins. Implemented in `tools.py` + schema in `tool_schemas.py`.

#### Tool 7: `get_delivery_performance` ✅
Queries `DELIVERY_EVENTS` for on-time rates and detention analysis. Implemented in `tools.py` + schema in `tool_schemas.py`.

**Steps:** ~~Add to `tools.py` + `tool_schemas.py`~~ ✅ → ~~test~~ ✅ → ~~commit~~ ✅ → PR open ⏳ awaiting review/merge

---

### 👤 Joel Vinas — 2 New Tools

#### Tool 8: `get_maintenance_health`
Queries `MAINTENANCE_RECORDS` + `TRUCKS` for fleet maintenance cost and downtime.

```python
def get_maintenance_health(
    maintenance_type: Optional[str] = None, start_date: str = "2022-01-01",
    end_date: str = "2025-12-31", top_n: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves truck maintenance health metrics including costs, downtime, and event counts.
    Args:
        maintenance_type: 'Scheduled', 'Unscheduled', or 'Inspection'. None = all.
        start_date/end_date: Date range in 'YYYY-MM-DD' format.
        top_n: Max trucks to return, sorted by total cost. Defaults to 20.
    Returns:
        List of dicts with truck_id, make, model_year, maintenance_events, total_cost, avg_downtime.
    """
    type_clause = ""
    if maintenance_type:
        safe = maintenance_type.replace("'", "''")
        type_clause = f"AND mr.maintenance_type = '{safe}'"
    sql = f"""
    SELECT mr.truck_id, tk.make, tk.model_year, tk.fuel_type,
           COUNT(*) AS maintenance_events,
           ROUND(SUM(mr.total_cost),2) AS total_cost,
           ROUND(AVG(mr.downtime_hours),1) AS avg_downtime_hours,
           ROUND(SUM(mr.labor_cost),2) AS total_labor_cost,
           ROUND(SUM(mr.parts_cost),2) AS total_parts_cost
    FROM CS5542_WEEK5.PUBLIC.MAINTENANCE_RECORDS mr
    JOIN CS5542_WEEK5.PUBLIC.TRUCKS tk ON mr.truck_id = tk.truck_id
    WHERE mr.maintenance_date BETWEEN '{start_date}' AND '{end_date}' {type_clause}
    GROUP BY mr.truck_id, tk.make, tk.model_year, tk.fuel_type
    ORDER BY total_cost DESC LIMIT {top_n};
    """
    return query_snowflake(sql)
```

#### Tool 9: `get_fuel_spend_analysis`
Queries `V_FUEL_SPEND` for geographic fuel cost breakdown.

```python
def get_fuel_spend_analysis(
    states: Optional[List[str]] = None, top_n: int = 15
) -> List[Dict[str, Any]]:
    """Retrieves fuel spend analysis aggregated by state and city.
    Args:
        states: State abbreviations to filter (e.g. ['TX','CA']). None = all.
        top_n: Max locations to return. Defaults to 15.
    Returns:
        List of dicts with state, city, total spend, gallons, avg price per gallon.
    """
    state_clause = ""
    if states:
        safe = lambda s: str(s).strip().replace("'", "''")
        state_filter = ", ".join(f"'{safe(s)}'" for s in states)
        state_clause = f"WHERE location_state IN ({state_filter})"
    sql = f"""
    SELECT location_state, location_city, purchases,
           total_gallons, avg_price_per_gallon, total_spend
    FROM CS5542_WEEK5.PUBLIC.V_FUEL_SPEND {state_clause}
    ORDER BY total_spend DESC LIMIT {top_n};
    """
    return query_snowflake(sql)
```

**Steps:** Add to `tools.py` + `tool_schemas.py` → test → commit → PR (`joel/maintenance-fuel-tools` → `main`)

---

## Phase 3: Streamlit Agent Chat Tab (All 3 — Teamwork)

> [!IMPORTANT]
> Merge Phase 2 PRs first, then collaborate on shared branch `team/streamlit-agent-chat`.

```bash
# Tony creates the branch:
git checkout main && git pull origin main
git checkout -b team/streamlit-agent-chat && git push origin team/streamlit-agent-chat

# Daniel & Joel check it out:
git fetch origin && git checkout team/streamlit-agent-chat
```

### Sub-Tasks

| Sub-Task | Owner | What to Build |
|---|---|---|
| Agent session setup | **Tony** | Initialize Gemini model with all 9 tools in `st.session_state`, manage chat session lifecycle, register Daniel's + Joel's new tools in `agent_tools` |
| Chat UI + conversation history | **Daniel** | `st.chat_input()`, `st.chat_message()` display loop, `st.session_state["messages"]` persistence, `st.spinner()` loading |
| Tool logs + response formatting | **Joel** | `st.expander("🔧 Tool Usage")` showing which tools were called, markdown rendering, error display |

### Commit Workflow (all push to same branch)
```bash
git pull origin team/streamlit-agent-chat   # Always pull first!
# Make changes
git add app/streamlit_app.py
git commit -m "feat(<name>): <description>"
git push origin team/streamlit-agent-chat
```

PR: `team/streamlit-agent-chat` → `main` — all 3 review and merge.

---

## Phase 4: Evaluation, Demo & Documentation

### 👤 Tony — Register New Tools + Final Integration

> [!IMPORTANT]
> **Daniel's PR is open (`daniel-lab6` → `main`).** Once you merge it, complete steps 1–2 below.
> Joel's tools will follow — do a second registration pass after his PR merges, or do both at once.

1. After merging Daniel's PR, update `agent.py` — add to the `agent_tools` list:
   ```python
   tools.get_route_profitability,
   tools.get_delivery_performance,
   # Add Joel's once his PR merges:
   # tools.get_maintenance_health,
   # tools.get_fuel_spend_analysis,
   ```
2. Run `python test_agent.py` to confirm Gemini binds all tools correctly
3. Write individual `CONTRIBUTION.md` for Canvas submission
4. Branch: `tony/register-tools` → PR → merge

### 👤 Daniel — Evaluation Report + Demo Video
1. Design 3 evaluation scenarios:
   - **Simple (1 tool):** "Show monthly revenue Jan–Jun 2023" → `get_monthly_revenue`
   - **Medium (2–3 tools):** "Compare Diesel vs Electric fleet + flag safety issues" → `get_fleet_performance` + `get_safety_metrics`
   - **Complex (3+ tools):** "Executive ops summary: pipeline health, top routes, maintenance needs" → `get_pipeline_logs` + `get_route_profitability` + `get_maintenance_health`
2. Write `task4_evaluation_report.md`: query, tools used, steps, latency, accuracy, failures
3. Record 3–5 min demo video → upload to YouTube (unlisted) → add link to README
4. Write individual `CONTRIBUTION.md` for Canvas
5. Branch: `daniel/evaluation-demo` → PR → merge

### 👤 Joel — README + CONTRIBUTIONS + Screenshots
1. Screenshot Antigravity analyzing `Week_6/` → save to `screenshots/`
2. Update `README.md`: system workflow, setup instructions, all 9 tools table, demo link, checklist
3. Update `CONTRIBUTIONS.md` with Lab 6 division of labor and all PRs
4. Write individual `CONTRIBUTION.md` for Canvas
5. Branch: `joel/readme-docs` → PR → merge

---

## Final Submission Checklist

| Deliverable | File | Owner |
|---|---|---|
| Antigravity Screenshot | `screenshots/` | Joel |
| Task 1 Report | `task1_antigravity_report.md` | Tony ✅ |
| Agent Tools (9 total) | `tools.py` | Tony (5) + Daniel (2) + Joel (2) |
| Tool Schemas | `tool_schemas.py` | Tony (5) + Daniel (2) + Joel (2) |
| Agent Implementation | `agent.py` | Tony ✅ + Tony registers new tools |
| Streamlit + Agent Chat | `app/streamlit_app.py` | All 3 (team) |
| Evaluation Report | `task4_evaluation_report.md` | Daniel |
| Demo Video Link | `README.md` | Daniel |
| Updated README | `README.md` | Joel |
| CONTRIBUTIONS.md | `CONTRIBUTIONS.md` | Joel |
| Individual CONTRIBUTION.md | each member's own | All 3 |

## Verification
```bash
python test_tools.py                    # All 9 tools return data
python test_agent.py                    # Gemini binds all 9 tools
streamlit run app/streamlit_app.py      # 8 original tabs + 🤖 Agent Chat works
# Confirm ≥5 PRs in GitHub (one per Phase 2 member + team + Phase 4)
```
