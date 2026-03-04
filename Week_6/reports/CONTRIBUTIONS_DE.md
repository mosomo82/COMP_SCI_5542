# CONTRIBUTIONS_DE.md — CS 5542 Lab 6 Individual Report

> **Name:** Daniel Evans (`devans2718`)
> **Team:** EVN — Tony Nguyen (`mosomo82`), Daniel Evans (`devans2718`), Joel Vinas (`joelvinas`)
> **Project:** AI Agent for Trucking Logistics Analytics — Snowflake + Gemini
> **Lab 6 Deadline:** March 3, 2026

---

## My Role & Overview

My primary Lab 6 responsibilities were:

| Phase       | Deliverable                                                      | Status      |
| ----------- | ---------------------------------------------------------------- | ----------- |
| **Phase 2** | `get_route_profitability` tool (`tools.py` + `tool_schemas.py`)  | ✅ Complete |
| **Phase 2** | `get_delivery_performance` tool (`tools.py` + `tool_schemas.py`) | ✅ Complete |
| **Phase 3** | Streamlit Chat UI — input loop, message history, spinner         | ✅ Complete |
| **Phase 4** | `task4_evaluation_report.md` — 5-scenario evaluation             | ✅ Complete |
| **Phase 4** | This individual contribution report                              | ✅ Complete |

---

## Phase 2 — New Agent Tools

### Tool 6: `get_route_profitability`

- **File:** `tools.py`
- **Schema:** `tool_schemas.py` (OpenAI function-calling format)
- **What it does:** Queries the `V_ROUTE_SCORECARD` Snowflake view to return route-level profit margins, gross profit, revenue, and cost per mile. Supports optional `min_margin` and `top_n` filtering.
- **Returns:** List of dicts with `origin_city`, `destination_city`, `total_trips`, `total_revenue`, `total_cost`, `gross_profit`, `profit_margin_pct`, and `avg_cost_per_mile`.
- **Why it matters:** Routes with high revenue but poor margins are invisible without this tool. It enables the agent to give actionable cost-reduction recommendations rather than just reporting totals.

### Tool 7: `get_delivery_performance`

- **File:** `tools.py`
- **Schema:** `tool_schemas.py`
- **What it does:** Queries the `DELIVERY_EVENTS` table for on-time delivery rates, average detention time, and early/late/on-time breakdowns. Supports `city` and `status` filtering.
- **Returns:** List of dicts with `destination_city`, `total_deliveries`, `on_time_count`, `late_count`, `early_count`, `on_time_rate_pct`, and `avg_detention_minutes`.
- **Why it matters:** Profitability data alone can be misleading — a highly profitable route may also have chronic late deliveries. Pairing this with `get_route_profitability` allows the agent to surface the "high profit, low reliability" problem (demonstrated in Scenario S3 of the evaluation).

### Testing

- Both tools are covered in `test_tools.py` (updated as part of this phase).
- Ran `python test_tools.py` — all 7 tools at the time exited 0.
- Confirmed schemas parse without error in `python test_agent.py` (Gemini tool binding).

### Evidence

- **Branch:** `origin/daniel-lab6`
- **Top commit:** [`923e986`](https://github.com/mosomo82/COMP_SCI_5542/commit/923e986) — Phase 2 tools implementation
- **Pull Request:** `daniel-lab6` → `main` (merged into `main` prior to Phase 3)

---

## Phase 3 — Streamlit Agent Chat Tab

I built the conversational UI layer in `app/streamlit_app.py` (under the **🤖 AI Assistant** tab), collaborating with Tony (agent session wiring) and Joel (tool-usage expander + formatting).

### My specific sub-tasks

| Component                    | Implementation                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Chat input**               | `st.chat_input("Ask the logistics agent...")` — captures user prompt each rerun                         |
| **Message history**          | `st.session_state["messages"]` — list of `{"role": ..., "content": ...}` dicts, persisted across reruns |
| **Chat display loop**        | `for msg in st.session_state["messages"]: st.chat_message(msg["role"]).markdown(msg["content"])`        |
| **Loading spinner**          | `with st.spinner("Agent is thinking..."):` wrapping the Gemini `send_message()` call                    |
| **User message append**      | Append user message to `session_state["messages"]` before the agent call                                |
| **Assistant message append** | Append agent response after the call completes                                                          |

### Design decisions

- Used `st.session_state` rather than re-instantiating the agent on each rerun, so the Gemini chat session retains its full conversation context.
- The spinner provides clear feedback for the 7–28 second latency range observed during evaluation.
- Messages use `.markdown()` rendering so the agent's structured responses (tables, bullet lists) display correctly.

---

## Phase 4 — Agent Evaluation Report

I wrote `task4_evaluation_report.md` (229 lines), which documents the systematic evaluation of the Gemini agent across five scenarios of increasing complexity.

### Evaluation Design

| Scenario | Complexity | Query Focus                                               | Tool(s) Expected                                      |
| -------- | ---------- | --------------------------------------------------------- | ----------------------------------------------------- |
| **S1**   | Simple     | Monthly revenue Jan–Jun 2024                              | `get_monthly_revenue`                                 |
| **S2**   | Medium     | Top diesel trucks + safety cross-reference                | `get_fleet_performance`, `get_safety_metrics`         |
| **S3**   | Complex    | Profit vs. on-time delivery correlation + recommendations | `get_route_profitability`, `get_delivery_performance` |
| **S4**   | Medium     | Highest maintenance cost truck + cost breakdown           | `get_maintenance_health`                              |
| **S5**   | Complex    | Regional fuel vs. maintenance correlation (CA vs TX)      | `get_fuel_spend_analysis`, `get_maintenance_health`   |

### Key Findings

- **Overall pass rate:** 4/5 scenarios (80% fully passed; S2 was a partial fail)
- **Tool selection accuracy:** 6/7 expected tool calls correctly made (86%)
- **S2 failure root cause:** The model prematurely concluded it could not cross-reference driver safety data even though `DRIVER_NAME` was present in the fleet results — a context-comprehension gap, not a data gap
- **Parallel tool execution:** The agent parallelized the two tool calls in S3 and S5 into a single round, demonstrating efficient multi-step reasoning
- **Latency range:** 7.11 s (simple) to 27.70 s (complex synthesis)

### Mitigations Recommended

1. Harden the system prompt with explicit multi-domain tool instructions
2. Add `driver_name` parameter to `get_safety_metrics` to guide cross-referencing
3. Use `query_snowflake` as a self-correction fallback for custom joins

---

## Technical Reflection

Working on Lab 6 gave me hands-on experience building an LLM layer over data. Tool design, schema authoring, UI integration, and formal evaluation.

- **Tool design instincts:** Writing `get_route_profitability` and `get_delivery_performance` required thinking about what a conversational agent needs vs. what a SQL query returns. I had to design the function signatures so Gemini could reliably select and parameterize them from natural language — the schema docstrings matter as much as the code.

- **Evaluation methodology:** The agent's failure in S2 was not a missing tool or a bad query — it was a reasoning failure about data it already had. Systematic evaluation is the only way to catch this class of bug.

- **Streamlit session state:** The chat UI required thought about Streamlit's rerun model. A naive implementation re-creates the Gemini session on every message, losing all context. Persisting the session object in `st.session_state` is necessary for the agent to maintain conversation context.

- **Collaboration with AI tooling (Antigravity):** I used Antigravity to accelerate writing the tool implementations and the evaluation report structure, then reviewed and validated every output against the actual Snowflake data and test results. The collaboration was most effective where the task was _well-specified_ (like tool signatures and schemas) and less effective where the task was analytical (interpreting the S2 failure).

---

## Summary of Contributions

| Artifact                      | My Contribution                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| `tools.py`                    | Implemented `get_route_profitability` (Tool 6) and `get_delivery_performance` (Tool 7) |
| `tool_schemas.py`             | Wrote OpenAI-format JSON schemas for both new tools                                    |
| `app/streamlit_app.py`        | Built the Chat UI: `st.chat_input`, message history, display loop, spinner             |
| `task4_evaluation_report.md`  | Designed 5 evaluation scenarios, ran evaluation, wrote full analysis                   |
| `reports/CONTRIBUTIONS_DE.md` | This document                                                                          |
