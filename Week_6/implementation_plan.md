# Lab 6 Implementation Plan — Team EVN

> **Team:** Tony Nguyen (`mosomo82`), Daniel Evans (`devans2718`), Joel Vinas (`joelvinas`)  
> **Repository:** `mosomo82/COMP_SCI_5542` → `Week_6/`

---

## Current State (Already Completed)

| File | Status | Owner |
|---|---|---|
| `tools.py` (5 tools) | ✅ Done | Tony |
| `tool_schemas.py` | ✅ Done | Tony |
| `agent.py` (Gemini CLI agent) | ✅ Done | Tony |
| `task1_antigravity_report.md` | ✅ Done | Tony |
| `test_tools.py` / `test_agent.py` | ✅ Done | Tony |
| `app/streamlit_app.py` | 🔄 Needs agent chat tab | — |

**Existing tools:** `query_snowflake`, `get_monthly_revenue`, `get_fleet_performance`, `get_pipeline_logs`, `get_safety_metrics`

---

## Updated Division of Labor

| Member | Tools (new) | Shared Work | Individual Deliverables |
|---|---|---|---|
| **Tony Nguyen** | — (already built 5) | Streamlit Agent Chat tab | Agent architecture lead |
| **Daniel Evans** | `get_route_profitability`, `get_delivery_performance` | Streamlit Agent Chat tab | `task4_evaluation_report.md`, demo video |
| **Joel Vinas** | `get_maintenance_health`, `get_fuel_spend_analysis` | Streamlit Agent Chat tab | `README.md`, `CONTRIBUTIONS.md`, screenshots |

> [!IMPORTANT]
> After this work the project will have **9 total tools** (5 existing + 2 from Daniel + 2 from Joel).

---

## Phase 0: Git Setup (All Members)

### 0.1 — Clone & Branch
```bash
git clone https://github.com/mosomo82/COMP_SCI_5542.git
cd COMP_SCI_5542
git checkout main && git pull origin main

# Each member creates their branch:
git checkout -b tony/streamlit-agent-ui      # Tony
git checkout -b daniel/route-delivery-tools  # Daniel
git checkout -b joel/maintenance-fuel-tools  # Joel
```

### 0.2 — Local Environment
```bash
cd Week_6
cp .env.example .env   # Fill in SNOWFLAKE_*, GEMINI_API_KEY
pip install -r ../requirements.txt
pip install google-generativeai python-dotenv folium
```

### 0.3 — Verify Existing Code
```bash
python test_tools.py   # All 4 tool tests pass
python test_agent.py   # "Agent setup and tool binding valid."
```

---

## Phase 1: New Tools (Parallel — Daniel & Joel)

---

### 👤 Daniel Evans — 2 New Tools

#### Tool 6: `get_route_profitability`
Queries `V_ROUTE_SCORECARD` for route-level profit margins.

```python
def get_route_profitability(
    min_loads: int = 3,
    min_margin_pct: float = 0.0,
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves route profitability metrics including revenue, fuel cost, and margin.

    Args:
        min_loads: Minimum completed loads for a route to be included. Defaults to 3.
        min_margin_pct: Minimum gross margin percentage to filter routes. Defaults to 0.0.
        top_n: Maximum number of routes to return, sorted by gross profit. Defaults to 20.

    Returns:
        List of dicts with route_label, total_loads, total_revenue, total_fuel_cost,
        gross_profit, margin_pct, avg_mpg.
    """
    sql = f"""
    SELECT route_label, total_loads, total_revenue, total_fuel_cost,
           gross_profit, margin_pct, avg_mpg
    FROM CS5542_WEEK5.PUBLIC.V_ROUTE_SCORECARD
    WHERE total_loads >= {min_loads}
      AND margin_pct >= {min_margin_pct}
    ORDER BY gross_profit DESC
    LIMIT {top_n};
    """
    return query_snowflake(sql)
```

#### Tool 7: `get_delivery_performance`
Queries `DELIVERY_EVENTS` for on-time delivery rates and detention analysis.

```python
def get_delivery_performance(
    event_type: str = "Delivery",
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves delivery event performance including on-time rates and detention times.

    Args:
        event_type: Filter by event type — 'Delivery' or 'Pickup'. Defaults to 'Delivery'.
        start_date: Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'.
        end_date: End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'.
        limit: Maximum number of rows to return. Defaults to 20.

    Returns:
        List of dicts with facility info, event counts, on-time rate, avg detention minutes.
    """
    safe_type = event_type.replace("'", "''")
    sql = f"""
    SELECT de.location_city, de.location_state,
           COUNT(*) AS total_events,
           ROUND(AVG(CASE WHEN de.on_time_flag THEN 1 ELSE 0 END) * 100, 1) AS on_time_pct,
           ROUND(AVG(de.detention_minutes), 1) AS avg_detention_min
    FROM CS5542_WEEK5.PUBLIC.DELIVERY_EVENTS de
    WHERE de.event_type = '{safe_type}'
      AND CAST(de.scheduled_datetime AS DATE) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY de.location_city, de.location_state
    ORDER BY total_events DESC
    LIMIT {limit};
    """
    return query_snowflake(sql)
```

#### Steps:
1. Add both functions to `tools.py`
2. Add corresponding schemas to `tool_schemas.py`
3. Register both in `agent.py` → `agent_tools` list
4. Test: `python -c "import tools; print(tools.get_route_profitability(top_n=2))"`
5. Commit & push:
```bash
git add tools.py tool_schemas.py agent.py
git commit -m "feat: Add get_route_profitability and get_delivery_performance tools"
git push origin daniel/route-delivery-tools
```
6. **Open PR** → base: `main` ← compare: `daniel/route-delivery-tools`

---

### 👤 Joel Vinas — 2 New Tools

#### Tool 8: `get_maintenance_health`
Queries `MAINTENANCE_RECORDS` for fleet maintenance cost and downtime analysis.

```python
def get_maintenance_health(
    maintenance_type: Optional[str] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    top_n: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves truck maintenance health metrics including costs, downtime, and event counts.

    Args:
        maintenance_type: Filter by type — 'Scheduled', 'Unscheduled', or 'Inspection'. Defaults to all if None.
        start_date: Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'.
        end_date: End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'.
        top_n: Maximum number of trucks to return, sorted by total cost. Defaults to 20.

    Returns:
        List of dicts with truck_id, maintenance events, total cost, avg downtime hours.
    """
    type_clause = ""
    if maintenance_type:
        safe = maintenance_type.replace("'", "''")
        type_clause = f"AND mr.maintenance_type = '{safe}'"
    sql = f"""
    SELECT mr.truck_id, tk.make, tk.model_year, tk.fuel_type,
           COUNT(*) AS maintenance_events,
           ROUND(SUM(mr.total_cost), 2) AS total_cost,
           ROUND(AVG(mr.downtime_hours), 1) AS avg_downtime_hours,
           ROUND(SUM(mr.labor_cost), 2) AS total_labor_cost,
           ROUND(SUM(mr.parts_cost), 2) AS total_parts_cost
    FROM CS5542_WEEK5.PUBLIC.MAINTENANCE_RECORDS mr
    JOIN CS5542_WEEK5.PUBLIC.TRUCKS tk ON mr.truck_id = tk.truck_id
    WHERE mr.maintenance_date BETWEEN '{start_date}' AND '{end_date}'
    {type_clause}
    GROUP BY mr.truck_id, tk.make, tk.model_year, tk.fuel_type
    ORDER BY total_cost DESC
    LIMIT {top_n};
    """
    return query_snowflake(sql)
```

#### Tool 9: `get_fuel_spend_analysis`
Queries `V_FUEL_SPEND` for geographic fuel cost breakdown.

```python
def get_fuel_spend_analysis(
    states: Optional[List[str]] = None,
    top_n: int = 15
) -> List[Dict[str, Any]]:
    """Retrieves fuel spend analysis aggregated by state and city.

    Args:
        states: List of state abbreviations to filter (e.g. ['TX', 'CA']). Defaults to all if None.
        top_n: Maximum number of locations to return, sorted by spend. Defaults to 15.

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
    FROM CS5542_WEEK5.PUBLIC.V_FUEL_SPEND
    {state_clause}
    ORDER BY total_spend DESC
    LIMIT {top_n};
    """
    return query_snowflake(sql)
```

#### Steps:
1. Add both functions to `tools.py`
2. Add corresponding schemas to `tool_schemas.py`
3. Register both in `agent.py` → `agent_tools` list
4. Test: `python -c "import tools; print(tools.get_maintenance_health(top_n=2))"`
5. Commit & push:
```bash
git add tools.py tool_schemas.py agent.py
git commit -m "feat: Add get_maintenance_health and get_fuel_spend_analysis tools"
git push origin joel/maintenance-fuel-tools
```
6. **Open PR** → base: `main` ← compare: `joel/maintenance-fuel-tools`

---

## Phase 2: Streamlit Agent Chat Tab (All 3 Members — Teamwork)

> [!IMPORTANT]
> **Merge Phase 1 PRs first**, then all 3 members collaborate on a single branch for the chat UI.

### 2.1 — Create Shared Branch
One member (Tony) creates the branch after merging Phase 1 PRs:
```bash
git checkout main && git pull origin main
git checkout -b team/streamlit-agent-chat
git push origin team/streamlit-agent-chat
```
Other members check it out:
```bash
git fetch origin
git checkout team/streamlit-agent-chat
```

### 2.2 — Sub-Tasks by Member

| Sub-Task | Owner | Description |
|---|---|---|
| Agent session setup | **Tony** | Wire Gemini model + all 9 tools into Streamlit via `st.session_state`, manage chat session lifecycle |
| Chat UI & conversation history | **Daniel** | Build `st.chat_input()`, `st.chat_message()` display, conversation history persistence, loading spinner |
| Tool logs & response formatting | **Joel** | Add expandable `st.expander("🔧 Tool Usage")` to show which tools the agent called, polish markdown rendering |

### 2.3 — Implementation Outline (new tab in `app/streamlit_app.py`)
```python
# Inside the tab definition (add as 9th tab: "🤖 Agent Chat")

# Tony's part — agent session init
import google.generativeai as genai
# Initialize model with all 9 tools and system prompt
# Store chat session in st.session_state["agent_chat"]

# Daniel's part — chat UI
# st.chat_input("Ask the logistics agent...")
# Loop through st.session_state["messages"] with st.chat_message()
# st.spinner("Agent is thinking...") around the send_message call

# Joel's part — tool logs display
# After response, show st.expander with tool call names and arguments
# Format agent response as markdown
```

### 2.4 — Coordinate Commits
Each member pushes to the **same branch** (`team/streamlit-agent-chat`):
```bash
git pull origin team/streamlit-agent-chat   # Always pull first!
# Make your changes
git add app/streamlit_app.py
git commit -m "feat(<your-name>): <what you added to Agent Chat tab>"
git push origin team/streamlit-agent-chat
```

### 2.5 — Open Team PR
- Title: `feat: Add 🤖 Agent Chat tab to Streamlit (team effort)`
- All 3 members review → merge to `main`

---

## Phase 3: Evaluation, Demo & Documentation

---

### 👤 Daniel Evans — Evaluation & Demo

#### 3.1 — Design 3 Evaluation Scenarios

| Scenario | Complexity | Expected Tools | Description |
|---|---|---|---|
| **Simple** | 1 tool | `get_monthly_revenue` | "Show monthly revenue from Jan–Jun 2023" |
| **Medium** | 2–3 tools | `get_fleet_performance` → `get_safety_metrics` | "Compare Diesel vs Electric fleet and flag safety issues for top drivers" |
| **Complex** | 3+ tools | `get_pipeline_logs` → `get_route_profitability` → `get_maintenance_health` | "Give me an executive ops summary: pipeline health, top routes, and trucks needing maintenance" |

#### 3.2 — Write `task4_evaluation_report.md`
For each scenario document: query, tools used, reasoning steps, latency, accuracy (1–5), failures.

#### 3.3 — Record Demo Video (3–5 min)
Show: project overview → agent chat interaction → tool usage → final outputs.
Upload to YouTube (unlisted) and add link to README.

#### 3.4 — Commit & PR
```bash
git checkout -b daniel/evaluation-demo
git add task4_evaluation_report.md
git commit -m "feat: Add evaluation report with 3 scenarios and demo link"
git push origin daniel/evaluation-demo
```
Open PR → merge after review.

---

### 👤 Joel Vinas — Documentation & Repository Polish

#### 3.5 — Take Antigravity Screenshot
- Open `Week_6/` in Antigravity → ask it to analyze the project → screenshot to `screenshots/`

#### 3.6 — Update `README.md`
Include: system workflow diagram, setup instructions, all 9 tools table, demo video link, deliverables checklist.

#### 3.7 — Update `CONTRIBUTIONS.md` (Lab 6 section)
Add Lab 6 division of labor table with all PRs and commits.

#### 3.8 — Commit & PR
```bash
git checkout -b joel/readme-docs
git add README.md CONTRIBUTIONS.md screenshots/
git commit -m "docs: Update README with agent workflow, all 9 tools, and Lab 6 contributions"
git push origin joel/readme-docs
```
Open PR → merge after review.

---

### All Members — Individual `CONTRIBUTION.md`
Each student writes their own file for **individual Canvas submission** covering:
- Personal responsibilities and implemented components
- Links to commits and PRs
- Reflection on technical contributions and learning outcomes

---

## Final Submission Checklist

| Deliverable | File | Owner |
|---|---|---|
| Antigravity Screenshot | `screenshots/` | Joel |
| Task 1 Report | `task1_antigravity_report.md` | Tony ✅ |
| Agent Tools (9 total) | `tools.py` | Tony + Daniel + Joel |
| Tool Schemas | `tool_schemas.py` | Tony + Daniel + Joel |
| Agent Implementation | `agent.py` | Tony + Daniel + Joel |
| Streamlit + Agent Chat | `app/streamlit_app.py` | All 3 (team) |
| Evaluation Report | `task4_evaluation_report.md` | Daniel |
| Demo Video Link | `README.md` | Daniel |
| Updated README | `README.md` | Joel |
| CONTRIBUTIONS.md | `CONTRIBUTIONS.md` | Joel |

---

## Verification Plan

```bash
# Tools (all 9)
python test_tools.py

# Agent binding
python test_agent.py

# Streamlit app
streamlit run app/streamlit_app.py
# → Verify all original 8 tabs + new 🤖 Agent Chat tab
# → Send a message, confirm response + spinner + history

# PR history
# → Confirm minimum 5 PRs: Daniel tools, Joel tools, team Streamlit, Daniel eval, Joel docs
```
