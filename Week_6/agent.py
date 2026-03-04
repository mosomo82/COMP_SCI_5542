import os
import sys
import json
import pathlib
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

# Setup paths and environment
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tools

_env_path = find_dotenv(filename=".env", raise_error_if_not_found=False) or str(ROOT / ".env")
load_dotenv(_env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("FATAL: GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
    sys.exit(1)

genai.configure(api_key=api_key)

# ── 1. Define the System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a highly capable AI Data Analytics Agent for a trucking logistics company.
You have access to a suite of specialized tools that allow you to query the company's Snowflake database.
Your job is to answer user questions about revenue, fleet performance, pipeline logs, safety metrics, 
route profitability, delivery performance, maintenance health, and fuel spend analysis.

When the user asks a question:
1. Determine if you need to use a tool to fetch the data. If so, call the appropriate tool.
2. If the data returned is not sufficient, or prompts further questions, call another tool (Multi-step reasoning).
3. Once you have all the data you need, synthesize it into a clear, concise, and professional final response for the user. Do not expose raw JSON to the user unless explicitly asked.
4. If a tool returns an error, gracefully inform the user about the limitation or try a different approach.
"""

# ── 2. Toolkit Declaration ────────────────────────────────────────────────────
# Gemini Python SDK allows passing actual Python functions as tools,
# and it automatically handles the schema generation based on type hints and docstrings!
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
    print("🚛 Logistics Agent Initializing... (Type 'quit' or 'exit' to stop)")
    
    # Initialize the model with the tools and system prompt
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        tools=agent_tools,
        system_instruction=SYSTEM_PROMPT
    )
    
    # Start a chat session (this manages the conversation history automatically)
    chat = model.start_chat(enable_automatic_function_calling=True)
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Agent shutting down. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("🤖 Agent is thinking...")
            
            # Send message. 
            # `enable_automatic_function_calling=True` means the SDK handles the loop!
            # It will:
            # 1. Send the prompt to Gemini.
            # 2. If Gemini requests a tool, the SDK calls our Python function locally.
            # 3. The SDK sends the function result back to Gemini.
            # 4. Steps 2-3 repeat until Gemini returns a final text response.
            response = chat.send_message(user_input)
            
            print(f"\n🚛 Agent: {response.text}")
            
        except Exception as e:
            print(f"\n❌ Loop Error: {str(e)}")
            print("The agent encountered an unexpected error and recovered.")

if __name__ == "__main__":
    run_agent()
