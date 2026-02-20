import os
import time
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from scripts.sf_connect import get_conn

LOG_PATH = "logs/pipeline_logs.csv"

def log_event(team: str, user: str, query_name: str, latency_ms: int, rows: int, error: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "team": team,
        "user": user,
        "query_name": query_name,
        "latency_ms": latency_ms,
        "rows_returned": rows,
        "error": error,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)

def run_query(sql: str):
    t0 = time.time()
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    latency_ms = int((time.time() - t0) * 1000)
    return df, latency_ms

st.title("CS 5542 — Week 5 Snowflake Dashboard Starter")

team = st.text_input("Team name", value="TeamX")
user = st.text_input("Your name", value="StudentName")

st.subheader("Filters")
category = st.text_input("Category filter (optional)", value="")
limit = st.slider("Limit rows", 10, 200, 50)

base_where = ""
if category.strip():
    safe = category.strip().replace("'", "''")
    base_where = f"WHERE CATEGORY ILIKE '%{safe}%'"

q1 = f"""
SELECT TEAM, CATEGORY, COUNT(*) AS N, AVG(VALUE) AS AVG_VALUE
FROM CS5542_WEEK5.PUBLIC.EVENTS
{base_where}
GROUP BY TEAM, CATEGORY
ORDER BY N DESC
LIMIT {limit};
"""

q2 = f"""
SELECT CATEGORY, COUNT(*) AS N_24H
FROM CS5542_WEEK5.PUBLIC.EVENTS
WHERE EVENT_TIME >= DATEADD('hour', -24, CURRENT_TIMESTAMP())
GROUP BY CATEGORY
ORDER BY N_24H DESC
LIMIT 20;
"""

q3 = f"""
SELECT U.TEAM, U.ROLE, E.CATEGORY, COUNT(*) AS N
FROM CS5542_WEEK5.PUBLIC.USERS U
JOIN CS5542_WEEK5.PUBLIC.EVENTS E
  ON U.TEAM = E.TEAM
GROUP BY U.TEAM, U.ROLE, E.CATEGORY
ORDER BY N DESC
LIMIT {limit};
"""

choice = st.selectbox("Choose query", ["Q1: Team x Category stats", "Q2: Category last 24h", "Q3: Join users x events"])
sql = {"Q1: Team x Category stats": q1, "Q2: Category last 24h": q2, "Q3: Join users x events": q3}[choice]

if st.button("Run"):
    try:
        df, latency_ms = run_query(sql)
        st.caption(f"Latency: {latency_ms} ms | Rows: {len(df)}")
        st.dataframe(df, use_container_width=True)

        if "N" in df.columns and "CATEGORY" in df.columns:
            chart = alt.Chart(df).mark_bar().encode(x="CATEGORY:N", y="N:Q")
            st.altair_chart(chart, use_container_width=True)

        log_event(team, user, choice, latency_ms, len(df), "")
    except Exception as e:
        st.error(str(e))
        log_event(team, user, choice, 0, 0, str(e))

st.subheader("Logs preview")
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH).tail(50), use_container_width=True)
else:
    st.info("No logs yet. Run a query to generate logs.")
