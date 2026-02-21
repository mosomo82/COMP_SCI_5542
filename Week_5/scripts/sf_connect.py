import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

# ── Secret resolution ─────────────────────────────────────────────────────────
# Priority: environment variable → Streamlit secrets → None
# This makes the module work both locally (.env) and on Streamlit Cloud (secrets).

def _get(key: str) -> str | None:
    """Return the value for *key* from env vars or st.secrets (in that order)."""
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
    # Password only required when NOT using externalbrowser / SSO
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

    # Optional SSO / external browser auth
    authenticator = _get("SNOWFLAKE_AUTHENTICATOR")
    if authenticator:
        conn_kwargs["authenticator"] = authenticator
        conn_kwargs.pop("password", None)

    return snowflake.connector.connect(**{k: v for k, v in conn_kwargs.items() if v})
