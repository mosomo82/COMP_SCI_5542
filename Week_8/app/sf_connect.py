import os
import pathlib
import snowflake.connector
from dotenv import load_dotenv, find_dotenv
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Ensure logging is set up
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_env_path = find_dotenv(
    filename=".env",
    raise_error_if_not_found=False,
    usecwd=False,
) or str(pathlib.Path(__file__).resolve().parents[1] / ".env")
load_dotenv(_env_path, override=False)

def _get(key: str) -> str | None:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None

def get_conn():
    required = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
    ]
    if not _get("SNOWFLAKE_AUTHENTICATOR"):
        required.append("SNOWFLAKE_PASSWORD")

    missing = [k for k in required if not _get(k)]
    if missing:
        raise RuntimeError(
            f"Missing Snowflake credentials: {missing}. "
            "Set them in .env (local) or Streamlit Cloud → Manage app → Secrets."
        )

    conn_kwargs = dict(
        account=_get("SNOWFLAKE_ACCOUNT"),
        user=_get("SNOWFLAKE_USER"),
        password=_get("SNOWFLAKE_PASSWORD"),
        role=_get("SNOWFLAKE_ROLE"),
        warehouse=_get("SNOWFLAKE_WAREHOUSE"),
        database=_get("SNOWFLAKE_DATABASE"),
        schema=_get("SNOWFLAKE_SCHEMA"),
    )

    authenticator = _get("SNOWFLAKE_AUTHENTICATOR")
    if authenticator:
        conn_kwargs["authenticator"] = authenticator
        conn_kwargs.pop("password", None)

    return snowflake.connector.connect(**{k: v for k, v in conn_kwargs.items() if v})

def _verify_and_ping():
    if globals().get("_snowflake_pinged"):
        return
        
    logger.info("Initializing Snowflake connection & verifying environment...")
    required = [
        "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
    ]
    if not _get("SNOWFLAKE_AUTHENTICATOR"):
        required.append("SNOWFLAKE_PASSWORD")

    missing = [k for k in required if not _get(k)]
    assert not missing, f"Startup Assertion Failed: Missing Snowflake credentials: {missing}. Set them in .env or Streamlit Secrets."

    # Gemini key check
    assert _get("GEMINI_API_KEY"), "Startup Assertion Failed: Missing GEMINI_API_KEY in environment."

    # Keep-alive ping
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        logger.info("Snowflake keep-alive ping successful.")
    except Exception as e:
        raise RuntimeError(f"Startup Ping Failed: Could not connect to Snowflake. {e}")
        
    globals()["_snowflake_pinged"] = True

_verify_and_ping()

@dataclass
class ComplianceResult:
    verdict: str  # "PASS" or "HARD_VETO"
    failing_bridges: List[Dict] = field(default_factory=list)
    min_weight_margin: Optional[float] = None
    min_clearance_margin: Optional[float] = None
    intersecting_count: int = 0
    veto_reason: str = ""

def check_route_compliance(session, route_linestring: str, vehicle_weight_tons: float, vehicle_height_mt: float) -> ComplianceResult:
    """
    CPP Step 3A: Spatial SQL Hard Gate.
    Calculates MIN(weight_limit_tons / vehicle_weight) and MIN(clearance_mt - vehicle_height)
    for all bridges intersecting the assigned route via ST_INTERSECTS.
    Returns HARD VETO if limit exceeded (ratio < 1.0 or clearance margin < 0.0).
    """
    # 1. Query offending bridges directly
    veto_sql = f"""
    SELECT 
        BRIDGE_ID,
        CLEARANCE_MT,
        WEIGHT_LIMIT_TONS,
        WEIGHT_LIMIT_TONS / {vehicle_weight_tons} AS WEIGHT_MARGIN,
        CLEARANCE_MT - {vehicle_height_mt} AS CLEARANCE_MARGIN
    FROM SILVER.DOT_BRIDGES
    WHERE ST_INTERSECTS(GEOMETRY, TO_GEOGRAPHY('{route_linestring}'))
      AND (WEIGHT_LIMIT_TONS < {vehicle_weight_tons} OR CLEARANCE_MT < {vehicle_height_mt})
    """
    
    # 2. Query aggregate safely
    agg_sql = f"""
    SELECT 
        COUNT(*) AS INTERSECTING_COUNT,
        MIN(WEIGHT_LIMIT_TONS / {vehicle_weight_tons}) AS MIN_WEIGHT_MARGIN,
        MIN(CLEARANCE_MT - {vehicle_height_mt}) AS MIN_CLEARANCE_MARGIN
    FROM SILVER.DOT_BRIDGES
    WHERE ST_INTERSECTS(GEOMETRY, TO_GEOGRAPHY('{route_linestring}'))
    """
    
    # Execute offending query
    violating_bridges_rows = session.sql(veto_sql).collect()
    
    # Execute aggregate query
    agg_row = session.sql(agg_sql).collect()[0]
    
    count = agg_row["INTERSECTING_COUNT"]
    min_weight = agg_row["MIN_WEIGHT_MARGIN"]
    min_clearance = agg_row["MIN_CLEARANCE_MARGIN"]
    
    if count == 0:
        return ComplianceResult(
            verdict="PASS",
            failing_bridges=[],
            min_weight_margin=None,
            min_clearance_margin=None,
            intersecting_count=0,
            veto_reason=""
        )
        
    failing_list = []
    veto_reasons = []
    for row in violating_bridges_rows:
        bridge_id = row["BRIDGE_ID"]
        weight_margin = row["WEIGHT_MARGIN"]
        clearance_margin = row["CLEARANCE_MARGIN"]
        
        failing_list.append({
            "bridge_id": bridge_id,
            "weight_limit_tons": row["WEIGHT_LIMIT_TONS"],
            "clearance_mt": row["CLEARANCE_MT"]
        })
        
        if weight_margin < 1.0:
            veto_reasons.append(f"Weight limit violated at {bridge_id}")
        if clearance_margin < 0.0:
            veto_reasons.append(f"Clearance limit violated at {bridge_id}")
            
    if failing_list:
        return ComplianceResult(
            verdict="HARD_VETO",
            failing_bridges=failing_list,
            min_weight_margin=min_weight,
            min_clearance_margin=min_clearance,
            intersecting_count=count,
            veto_reason="; ".join(set(veto_reasons))
        )
        
    return ComplianceResult(
        verdict="PASS",
        failing_bridges=[],
        min_weight_margin=min_weight,
        min_clearance_margin=min_clearance,
        intersecting_count=count,
        veto_reason=""
    )
