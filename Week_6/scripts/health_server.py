"""
FastAPI Server for Uptime Monitoring.
Runs a health check returning Snowflake connectivity and Gemini API key status.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import os
from .sf_connect import get_conn, _get

app = FastAPI(title="Logistics AI Dashboard - Health Server")

@app.get("/health")
def health_check():
    health_status = {
        "status": "ok",
        "snowflake_connected": False,
        "gemini_configured": False
    }

    # 1. Check Gemini Key
    if _get("GEMINI_API_KEY"):
        health_status["gemini_configured"] = True
    else:
        health_status["status"] = "degraded"

    # 2. Check Snowflake
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        health_status["snowflake_connected"] = True
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        return JSONResponse(status_code=503, content=health_status)

    if health_status["status"] == "ok":
        return health_status
    else:
        return JSONResponse(status_code=200, content=health_status)
