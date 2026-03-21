import os
import sys
import json
import time
import pathlib
import logging
import google.generativeai as genai
from functools import wraps
from dotenv import load_dotenv, find_dotenv

# Setup paths and environment
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tools

_env_path = find_dotenv(filename=".env", raise_error_if_not_found=False) or str(ROOT / ".env")
load_dotenv(_env_path)

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.critical("FATAL: GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
    sys.exit(1)

genai.configure(api_key=api_key)

# ── EXPONENTIAL BACKOFF RETRY ──────────────────────────────────────────────────
def retry_gemini(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise
    return wrapper

# ── 1. Define the System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a highly capable AI Data Analytics Agent for a trucking logistics company.
You have access to a suite of specialized tools that allow you to query the company's Snowflake database.
Your job is to answer user questions about revenue, fleet performance, pipeline logs, safety metrics,
route profitability, delivery performance, maintenance health, and fuel spend analysis.

TOOL SELECTION RULES — follow these before calling any tool:
1. ALWAYS prefer a specialized tool over query_snowflake when the question falls within a tool's domain.
   Only use query_snowflake when NO specialized tool covers the data needed.
2. When a query covers TWO domains, call BOTH relevant specialized tools — never combine into one
   query_snowflake call. Examples:
   - Trucks + safety records    → get_fleet_performance AND get_safety_metrics
   - Fuel spend + maintenance   → get_fuel_spend_analysis AND get_maintenance_health
   - Drivers + incidents        → get_fleet_performance AND get_safety_metrics
3. Questions about top trucks by revenue AND incidents/violations/safety: always call BOTH
   get_fleet_performance AND get_safety_metrics, even if the query uses the word "separately".
4. Questions about fuel cost by state/region: use get_fuel_spend_analysis, not query_snowflake.
5. Questions about maintenance costs, repair costs, or truck downtime: use get_maintenance_health,
   not query_snowflake.
6. Questions about driver revenue, driver performance, or driver-level metrics: use
   get_fleet_performance, not query_snowflake.

When the user asks a question:
1. Apply the TOOL SELECTION RULES above to decide which tools to call.
2. If the data returned is not sufficient, or prompts further questions, call another tool (multi-step reasoning).
3. Once you have all the data you need, synthesize it into a clear, concise, and professional final response. Do not expose raw JSON unless explicitly asked.
4. If a tool returns an error, gracefully inform the user and try a different approach.
"""

# ── 2. Toolkit Declaration ────────────────────────────────────────────────────
agent_tools = [
    tools.query_snowflake,
    tools.get_monthly_revenue,
    tools.get_fleet_performance,
    tools.get_pipeline_logs,
    tools.get_safety_metrics,
    tools.get_route_profitability,
    tools.get_delivery_performance,
    tools.get_maintenance_health,
    tools.get_fuel_spend_analysis
]

# ── 3. Agent Execution Loop ───────────────────────────────────────────────────
def run_agent():
    logger.info("🚛 Logistics Agent Initializing... (Type 'quit' or 'exit' to stop)")
    
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        tools=agent_tools,
        system_instruction=SYSTEM_PROMPT
    )
    
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    @retry_gemini
    def safe_send_message(msg):
        return chat.send_message(msg)

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                logger.info("Agent shutting down. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            logger.info("🤖 Agent is thinking...")
            
            response = safe_send_message(user_input)
            
            try:
                answer = response.text
            except ValueError:
                text_parts = [
                    p.text for p in response.candidates[0].content.parts
                    if hasattr(p, "text") and p.text
                ]
                answer = "\n".join(text_parts) if text_parts else (
                    "I processed your request but couldn't generate a text summary. "
                    "Please try rephrasing your question."
                )
            
            print(f"\n🚛 Agent: {answer}")
            
        except Exception as e:
            logger.error(f"❌ Agent error: {str(e)}")
            logger.error("The agent encountered an error but is still running. Try another question.")

if __name__ == "__main__":
    run_agent()
