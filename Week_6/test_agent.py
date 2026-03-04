import sys
import os
import pathlib
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tools

# Load env
_env_path = find_dotenv(filename=".env", raise_error_if_not_found=False) or str(ROOT / ".env")
load_dotenv(_env_path)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("FATAL: GEMINI_API_KEY is missing from .env")
    sys.exit(1)

genai.configure(api_key=api_key)

try:
    # Test generation and tool bindings
    model = genai.GenerativeModel('gemini-2.5-flash', tools=[tools.get_monthly_revenue,
                                                             tools.get_fleet_performance, 
                                                             tools.get_pipeline_logs, 
                                                             tools.get_safety_metrics, 
                                                             tools.get_route_profitability, 
                                                             tools.get_delivery_performance, 
                                                             tools.get_maintenance_health])
    chat = model.start_chat()
    print("Agent setup and tool binding valid.")
    sys.exit(0)
except Exception as e:
    print(f"Error binding tools: {e}")
    sys.exit(1)
