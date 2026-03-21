# new sf_connect.py
import os
import pathlib
import snowflake.connector
from dotenv import load_dotenv, find_dotenv
import logging

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

# ── Startup Assertions & Keep-Alive Ping ──────────────────────────────────────
# Execute immediately upon module import
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
