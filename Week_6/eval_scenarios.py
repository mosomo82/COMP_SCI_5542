"""
Task 4 - Agent Evaluation Scenarios
====================================
Runs three evaluation scenarios (Simple, Medium, Complex) against the
Gemini-powered logistics agent and records:
  - tools selected by the model
  - number of reasoning steps (tool calls)
  - wall-clock latency
  - accuracy / correctness vs. ground-truth checks
  - any failures

Usage:
    py eval_scenarios.py          # runs all three scenarios
    py eval_scenarios.py --json   # also dumps raw results to eval_results.json
"""

import os, sys, json, time, pathlib, textwrap, io
from datetime import datetime
from typing import List, Dict, Any

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# -- PATH SETUP ----------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tools  # noqa: E402

# -- ENV SETUP -----------------------------------------------------------------
_env = find_dotenv(filename=".env", raise_error_if_not_found=False) or str(ROOT / ".env")
load_dotenv(_env)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("FATAL: GEMINI_API_KEY not found in .env"); sys.exit(1)
genai.configure(api_key=api_key)


# -- TOOL DISPATCHER -----------------------------------------------------------
TOOL_MAP = {
    "query_snowflake":         tools.query_snowflake,
    "get_monthly_revenue":     tools.get_monthly_revenue,
    "get_fleet_performance":   tools.get_fleet_performance,
    "get_pipeline_logs":       tools.get_pipeline_logs,
    "get_safety_metrics":      tools.get_safety_metrics,
    "get_route_profitability": tools.get_route_profitability,
    "get_delivery_performance":tools.get_delivery_performance,
}

AGENT_TOOLS = list(TOOL_MAP.values())

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a highly capable AI Data Analytics Agent for a trucking logistics company.
    You have access to specialized tools to query the company's Snowflake database.
    Answer questions about revenue, fleet performance, pipeline logs, safety metrics,
    route profitability, and delivery performance.

    Rules:
    1. Call the most appropriate tool(s) to get the data you need.
    2. If data is insufficient, call additional tools (multi-step reasoning).
    3. Synthesize tool results into a clear, professional response.
    4. If a tool returns an error, inform the user gracefully.
""")


# ==============================================================================
#  EVALUATION SCENARIOS
# ==============================================================================
SCENARIOS: List[Dict[str, Any]] = [
    # --- SIMPLE: single tool --------------------------------------------------
    {
        "id": "S1",
        "level": "Simple",
        "description": "Single-tool retrieval -- monthly revenue lookup",
        "query": (
            "Show me the monthly revenue from January 2024 to June 2024."
        ),
        "expected_tools": ["get_monthly_revenue"],
        "ground_truth_check": (
            "Response should contain monthly revenue figures for the "
            "Jan-Jun 2024 period, with recognisable month labels and dollar amounts."
        ),
        "expected_min_steps": 1,
        "expected_max_steps": 1,
    },
    # --- MEDIUM: multiple tools -----------------------------------------------
    {
        "id": "S2",
        "level": "Medium",
        "description": "Multi-tool retrieval -- fleet performance + safety cross-reference",
        "query": (
            "Which are our top 5 performing diesel trucks by revenue, "
            "and do any of those drivers have safety incidents on record? "
            "Provide both the fleet data and the safety data."
        ),
        "expected_tools": ["get_fleet_performance", "get_safety_metrics"],
        "ground_truth_check": (
            "Response should include (a) a top-5 diesel truck list with revenue "
            "numbers, and (b) safety incident data for the named drivers."
        ),
        "expected_min_steps": 2,
        "expected_max_steps": 3,
    },
    # --- COMPLEX: reasoning + synthesis ---------------------------------------
    {
        "id": "S3",
        "level": "Complex",
        "description": "Reasoning & synthesis -- profitability vs. delivery reliability analysis",
        "query": (
            "I need a strategic analysis: compare our top 10 most profitable routes "
            "against delivery performance in the same cities. Are our most profitable "
            "routes also the most reliable in terms of on-time delivery? "
            "Identify any routes where high profit coincides with poor on-time rates, "
            "and recommend corrective actions."
        ),
        "expected_tools": [
            "get_route_profitability",
            "get_delivery_performance",
        ],
        "ground_truth_check": (
            "Response should (a) list top profitable routes with margins, "
            "(b) include on-time delivery percentages for matching cities, "
            "(c) identify mismatches between profit and reliability, and "
            "(d) provide actionable recommendations."
        ),
        "expected_min_steps": 2,
        "expected_max_steps": 4,
    },
]


# ==============================================================================
#  MANUAL AGENT LOOP  (so we can instrument each step)
# ==============================================================================
def run_scenario(scenario: dict) -> dict:
    """Execute one evaluation scenario and return structured metrics."""
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=AGENT_TOOLS,
        system_instruction=SYSTEM_PROMPT,
    )
    chat = model.start_chat()

    result = {
        "id":           scenario["id"],
        "level":        scenario["level"],
        "description":  scenario["description"],
        "query":        scenario["query"],
        "tools_called": [],
        "reasoning_steps": 0,
        "latency_s":    0.0,
        "final_answer":  "",
        "accuracy":     "",
        "pass":         False,
        "error":        None,
    }

    t0 = time.perf_counter()

    def _send_with_retry(send_fn, max_retries=3):
        """Wrapper that retries on 429 rate-limit errors with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                return send_fn()
            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries:
                    wait = (2 ** attempt) * 5  # 5s, 10s, 20s
                    print(f"    [rate-limit] 429 received, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise

    try:
        # Initial send (with retry)
        response = _send_with_retry(lambda: chat.send_message(scenario["query"]))

        # Agentic loop: keep resolving tool calls until Gemini gives text
        max_iterations = 10
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # Check candidate parts for function calls
            fn_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    fn_calls.append(part.function_call)

            if not fn_calls:
                # No more tool calls -- we have the final text answer
                break

            # Execute each function call
            fn_responses = []
            for fc in fn_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                result["tools_called"].append(fn_name)
                result["reasoning_steps"] += 1

                print(f"    -> Tool call #{result['reasoning_steps']}: {fn_name}({list(fn_args.keys())})")

                if fn_name in TOOL_MAP:
                    try:
                        fn_result = TOOL_MAP[fn_name](**fn_args)
                    except Exception as tool_err:
                        fn_result = {"error": str(tool_err)}
                else:
                    fn_result = {"error": f"Unknown tool: {fn_name}"}

                fn_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn_name,
                            response={"result": json.loads(json.dumps(fn_result, default=str))},
                        )
                    )
                )

            # Send tool results back to model (with retry)
            response = _send_with_retry(
                lambda: chat.send_message(
                    genai.protos.Content(parts=fn_responses)
                )
            )

        # Extract final text
        text_parts = [
            p.text for p in response.candidates[0].content.parts
            if hasattr(p, "text") and p.text
        ]
        result["final_answer"] = "\n".join(text_parts)

    except Exception as exc:
        result["error"] = str(exc)

    result["latency_s"] = round(time.perf_counter() - t0, 2)

    # -- Accuracy heuristic ----------------------------------------------------
    expected = set(scenario["expected_tools"])
    called   = set(result["tools_called"])
    tools_ok = expected.issubset(called)

    steps_ok = (
        scenario["expected_min_steps"]
        <= result["reasoning_steps"]
        <= scenario["expected_max_steps"] + 2   # allow slight overshoot
    )

    answer_ok = len(result["final_answer"]) > 50 and result["error"] is None

    if tools_ok and steps_ok and answer_ok:
        result["accuracy"] = "PASS -- correct tools, reasonable steps, substantive answer"
        result["pass"] = True
    else:
        reasons = []
        if not tools_ok:
            reasons.append(f"missing expected tools {expected - called}")
        if not steps_ok:
            reasons.append(f"steps={result['reasoning_steps']} outside [{scenario['expected_min_steps']},{scenario['expected_max_steps']+2}]")
        if not answer_ok:
            reasons.append("answer too short or error occurred")
        result["accuracy"] = "FAIL -- " + "; ".join(reasons)

    return result


# ==============================================================================
#  PRETTY PRINTER (ASCII-safe for Windows cp1252)
# ==============================================================================
def print_result(r: dict):
    border = "=" * 72
    print(f"\n{border}")
    print(f"  [{r['id']}] {r['level'].upper()} -- {r['description']}")
    print(border)
    print(f"  Query:            {r['query'][:90]}...")
    print(f"  Tools called:     {r['tools_called']}")
    print(f"  Reasoning steps:  {r['reasoning_steps']}")
    print(f"  Latency:          {r['latency_s']} s")
    print(f"  Accuracy:         {r['accuracy']}")
    if r["error"]:
        print(f"  [!] Error:        {r['error'][:120]}")
    print(f"  Answer preview:   {r['final_answer'][:200]}...")
    print(border)


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    print("=" * 66)
    print("  Task 4 -- Agent Evaluation Scenarios")
    print("=" * 66)
    print()

    COOLDOWN_SECONDS = 15  # pause between scenarios to stay within RPM limits

    all_results = []
    for i, scenario in enumerate(SCENARIOS):
        if i > 0:
            print(f"  [cooldown] Waiting {COOLDOWN_SECONDS}s before next scenario...")
            time.sleep(COOLDOWN_SECONDS)
        print(f"> Running scenario {scenario['id']} ({scenario['level']})...")
        r = run_scenario(scenario)
        print_result(r)
        all_results.append(r)

    # Summary
    passed = sum(1 for r in all_results if r["pass"])
    print(f"\n{'-'*72}")
    print(f"  SUMMARY:  {passed}/{len(all_results)} scenarios passed")
    print(f"  Total latency: {sum(r['latency_s'] for r in all_results):.1f} s")
    print(f"{'-'*72}\n")

    # Optionally dump JSON
    if "--json" in sys.argv:
        out = ROOT / "eval_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
        print(f"  Raw results saved to {out}")

    return all_results


if __name__ == "__main__":
    main()
