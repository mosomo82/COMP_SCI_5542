# ADDENDUM: Lab 6 — Multi-Agent Analytics Progress

> **Team Name:** EVN 
> **Team Members:** Tony Nguyen (mosomo82), Daniel Evans (devans2718), Joel Vinas (joelvinas)
> **Project Title:** Logistics Operation Dashboard — Snowflake Integration

> **Deadline:** Feb. 24, 2026 — 11:59 PM

---

## Roles and Responsibilities

### Member 1: Tony Nguyen (mosomo82) — Backend & AI Lead
- **Responsibilities:**
  - **Phase 1 (Infrastructure):** Designed and implemented the first 5 specialized agent tools in `tools.py` (`query_snowflake`, `get_monthly_revenue`, `get_fleet_performance`, `get_pipeline_logs`, `get_safety_metrics`).
  - **Agent Architecture:** Created `agent.py` using Gemini 2.5 Flash with automatic function calling and a robust system prompt for logistics analysis.
  - **Phase 3 (Streamlit Integration):** Wired the AI Agent into `app/streamlit_app.py`. Implemented `st.session_state` management for chat history and tool logs to prevent session resets.
  - **Framework Development:** Built the dynamic tool registration system that allows Daniel and Joel's new tools to be "hot-loaded" into the agent upon merging.
- **Evidence (PR/commits):**
  - Branch: `tony/lab6-agent-integration`
  - Commit [`4458fd7`](https://github.com/mosomo82/COMP_SCI_5542/commit/4458fd7) — feat(Tony): agent session setup with all 9 tools, chat lifecycle, tool logging.
- **Tested:**
  - Verified agent can handle multi-step reasoning (e.g., "Find the truck with high revenue and check its safety logs").
  - Confirmed "Clear Chat" button correctly resets the Gemini session and GUI.
  - Validated tool-usage logging in the "🔧 Tool Usage" expanders.

## Lab 6 Division of Labor Summary

| Name | Primary Lab 6 Contribution | Tool Ownership |
| :--- | :--- | :--- |
| **Tony Nguyen** | Agent Infrastructure & Session Wiring | Tools 1-5 (Core) |
| **Daniel Evans** | Chat UI & History Polish | Tools 6-7 (Profit & Delivery) |
| **Joel Vinas** | Tool Logging & Formatting | Tools 8-9 (Maint & Fuel) |
