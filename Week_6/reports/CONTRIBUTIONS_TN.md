# CONTRIBUTIONS_TN.md — CS 5542 Lab 6 Individual Report

> **Name:** Tony Nguyen (`mosomo82` / `mtuan82`)
> **Team:** EVN — Tony Nguyen (`mosomo82`), Daniel Evans (`devans2718`), Joel Vinas (`joelvinas`)
> **Project:** AI Agent for Trucking Logistics Analytics — Snowflake + Gemini
> **Lab 6 Deadline:** March 3, 2026

---

## My Role & Overview

My primary Lab 6 responsibilities were:

| Phase       | Deliverable                                                          | Status      |
| ----------- | -------------------------------------------------------------------- | ----------- |
| **Phase 1** | Core agent infrastructure (`agent.py`, `tools.py` Tools 1–5)        | ✅ Complete |
| **Phase 1** | `task1_antigravity_report.md` — automated assistance documentation   | ✅ Complete |
| **Phase 2** | `get_safety_metrics` tool (Tool 5) — driver incident analytics       | ✅ Complete |
| **Phase 3** | Streamlit session integration — `st.session_state`, tool registration | ✅ Complete |
| **Phase 4** | This individual contribution report                                  | ✅ Complete |

---

## Phase 1 — Agent Infrastructure & Core Tools

### Agent Architecture (`agent.py`)

- Built the agent execution loop using Google Gemini 2.5 Flash with `enable_automatic_function_calling=True`.
- Designed the system prompt for logistics-domain reasoning with multi-step tool chaining.
- Configured the toolkit declaration to accept Python functions directly, leveraging the Gemini SDK's automatic schema generation from type hints and docstrings.

### Core Tools (`tools.py` — Tools 1–5)

| # | Tool                    | Description                                                          |
|---|-------------------------|----------------------------------------------------------------------|
| 1 | `query_snowflake`       | Arbitrary read-only SQL execution against the Snowflake warehouse    |
| 2 | `get_monthly_revenue`   | Aggregated monthly revenue trends from `V_MONTHLY_REVENUE`           |
| 3 | `get_fleet_performance` | Truck-level metrics (trips, miles, MPG, revenue) from `V_TRIP_PERFORMANCE` |
| 4 | `get_pipeline_logs`     | System health and ingestion latency from local CSV logs              |
| 5 | `get_safety_metrics`    | Driver incident analytics (claims, at-fault, injuries) from `SAFETY_INCIDENTS` |

### Design Decisions

- **`query_snowflake` as escape hatch:** Included a generic SQL tool so the agent can self-correct when specialized tools don't cover a question. This follows the ReAct pattern — the agent can reason about what data it needs and construct a query.
- **Datetime serialization:** All datetime columns are converted to ISO-8601 strings before returning to Gemini, preventing JSON serialization errors.
- **Error encapsulation:** Every tool wraps exceptions in `{"error": ...}` dicts instead of raising, so the agent can gracefully report failures to the user.

### Snowflake Connection (`scripts/sf_connect.py`)

- Built the centralized connection module with `.env` and Streamlit secrets dual-resolution (`os.getenv` → `st.secrets` fallback).
- Supports both password and `externalbrowser` / SSO authentication via the optional `SNOWFLAKE_AUTHENTICATOR` variable.

### Evidence

- **Branch:** `tony/lab6-agent-integration`
- **Commit:** [`4458fd7`](https://github.com/mosomo82/COMP_SCI_5542/commit/4458fd7) — feat(Tony): agent session setup with all 9 tools, chat lifecycle, tool logging.
- **Commit:** [`68fcf89`](https://github.com/mosomo82/COMP_SCI_5542/commit/68fcf89) — docs: add Lab 6 contribution addendum for Tony.

---

## Phase 1 — Antigravity Report (`task1_antigravity_report.md`)

Documented how Antigravity IDE was used during development:

- **Prompts given:** dependency fixes, Snowflake refactoring, pipeline documentation, S3 automation, contribution updates.
- **Improvements accepted:** centralized `snowflake_conn.py`, missing library installs.
- **Improvements modified:** S3 bucket paths, IAM policies, contribution reflections.
- **Reflection:** Antigravity excels at contextual refactoring and boilerplate but requires human oversight for domain-specific configuration.

---

## Phase 3 — Streamlit Session Integration

Wired the AI Agent into `app/streamlit_app.py` as the **🤖 Agent Chat** tab:

| Component                  | Implementation                                                                                      |
| -------------------------- | --------------------------------------------------------------------------------------------------- |
| **Session initialization** | `st.session_state["chat"]` — persists the Gemini chat session across Streamlit reruns               |
| **Dynamic tool registry**  | All 9 tools loaded into `agent_tools` list — allows teammates to merge new tools without conflicts  |
| **Tool-usage logging**     | Each tool call is logged to `st.session_state["tool_log"]` and displayed in sidebar expanders       |
| **Clear Chat button**      | Resets `st.session_state["chat"]`, `messages`, and `tool_log` for a fresh session                   |

### Design Decisions

- **Hot-loading architecture:** The dynamic tool registration system lets Daniel's and Joel's tools be added by simply appending to `agent_tools` in `agent.py` — no session rewiring needed per tool.
- **Session persistence:** Used `st.session_state` rather than re-instantiating the Gemini model on each rerun, preserving full conversation context.

### Testing

- Verified agent binds to all 9 active tools and maintains multi-turn conversation history.
- Validated tool-usage transparency via the sidebar and message expanders.
- Confirmed "Clear Chat" button correctly resets the Gemini session and GUI.

---

## Technical Reflection

Lab 6 was my first experience building an LLM-powered tool-calling agent end-to-end. Key takeaways:

- **Tool signature design is critical.** Gemini selects tools based on function names, docstrings, and parameter types. Ambiguous names or missing docstrings cause tool-selection failures. I learned to treat the function signature as a user-facing API contract.

- **Automatic vs. manual function calling trade-offs.** `enable_automatic_function_calling=True` is great for production (clean code, no loop management) but unsuitable for evaluation — you can't count steps or intercept tool calls. The manual loop in `eval_scenarios.py` was necessary for instrumentation.

- **Session state management in Streamlit is non-trivial.** Streamlit reruns the entire script on every interaction. Naively recreating the Gemini session loses conversation context. Persisting the session in `st.session_state` was the key insight.

- **Rate-limit handling is essential for CI.** Even with a Pro API key, running 5 evaluation scenarios sequentially can hit per-minute quotas. Adding cooldown + retry logic was necessary for reproducible evaluation runs.

---

## Summary of Contributions

| Artifact                               | My Contribution                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| `agent.py`                             | Full agent architecture: system prompt, toolkit declaration, execution loop         |
| `tools.py` (Tools 1–5)                | `query_snowflake`, `get_monthly_revenue`, `get_fleet_performance`, `get_pipeline_logs`, `get_safety_metrics` |
| `scripts/sf_connect.py`               | Centralized Snowflake connection with dual-resolution credentials                   |
| `app/streamlit_app.py` (Agent tab)     | Session integration, dynamic tool registry, tool-usage logging, Clear Chat          |
| `task1_antigravity_report.md`          | Antigravity IDE usage documentation                                                 |
| `reports/CONTRIBUTIONS_TN.md`          | This document                                                                       |
