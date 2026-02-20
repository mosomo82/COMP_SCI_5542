"""
CS 5542 — Week 5  ·  Trucking-Logistics Snowflake Dashboard
Tabs: Overview | Fleet & Drivers | Routes | Fuel Spend | Monitoring
"""

import os, time
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# ── connection helper (reuse existing module) ────────────────────────────────
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.sf_connect import get_conn

# ── config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CS 5542 Trucking Dashboard",
    page_icon="🚛",
    layout="wide",
)

LOG_PATH = "logs/pipeline_logs.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def _perf_note(latency_ms: int, rows: int, error: str = "") -> str:
    """Auto-generate a short performance observation."""
    if error:
        return f"ERROR: {error[:120]}"
    parts: list[str] = []
    if latency_ms > 5000:
        parts.append("High latency (>5 s) — consider warehouse scaling or query optimization")
    elif latency_ms > 2000:
        parts.append("Moderate latency (2-5 s) — acceptable for ad-hoc analysis")
    else:
        parts.append("Fast response (<2 s)")
    if rows == 0:
        parts.append("no rows returned — check filters or data availability")
    elif rows > 500:
        parts.append(f"large result set ({rows} rows) — consider tighter filters")
    return "; ".join(parts)


def log_event(team: str, user: str, query_name: str, latency_ms: int, rows: int, error: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    note = _perf_note(latency_ms, rows, error)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "team": team,
        "user": user,
        "query_name": query_name,
        "latency_ms": latency_ms,
        "rows_returned": rows,
        "error": error,
        "perf_note": note,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)


@st.cache_data(ttl=120, show_spinner="Querying Snowflake …")
def run_query(sql: str) -> tuple[pd.DataFrame, int]:
    """Execute *sql* and return (DataFrame, latency_ms)."""
    t0 = time.time()
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    return df, int((time.time() - t0) * 1000)


def safe(text: str) -> str:
    """Minimal SQL-string escape."""
    return text.strip().replace("'", "''")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️  Session")
    team = st.text_input("Team name", value="TeamX")
    user = st.text_input("Your name", value="Student")
    st.divider()
    st.caption("Queries hit the 5 views created in `04_views.sql`.")

# ── title ────────────────────────────────────────────────────────────────────
st.title("🚛 CS 5542 — Trucking Logistics Dashboard")
st.caption("Live connection to **Snowflake** · parameterized inputs · Altair charts")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_fleet, tab_routes, tab_fuel, tab_monitor = st.tabs(
    ["📊 Overview", "🚛 Fleet & Drivers", "🗺️ Routes", "⛽ Fuel Spend", "📈 Monitoring"]
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview  (monthly revenue trend + KPIs)
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Monthly Revenue Trend")

    # -- parameterized inputs --------------------------------------------------
    ov_c1, ov_c2 = st.columns(2)
    with ov_c1:
        start_month = st.date_input("Start month", value=pd.Timestamp("2023-01-01"), key="ov_start")
    with ov_c2:
        end_month   = st.date_input("End month",   value=pd.Timestamp("2025-12-31"), key="ov_end")

    sql_overview = f"""
    SELECT *
    FROM CS5542_WEEK5.PUBLIC.V_MONTHLY_REVENUE
    WHERE MONTH >= '{start_month}'
      AND MONTH <= '{end_month}'
    ORDER BY MONTH;
    """

    if st.button("Run Overview Query", key="btn_ov"):
        df, ms = run_query(sql_overview)
        log_event(team, user, "overview_monthly_revenue", ms, len(df))
        st.caption(f"⏱ {ms} ms  ·  {len(df)} rows")

        if df.empty:
            st.warning("No data for the selected date range.")
        else:
            # KPI cards
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue",        f"${df['REVENUE'].sum():,.0f}")
            k2.metric("Total Loads",           f"{df['TOTAL_LOADS'].sum():,}")
            k3.metric("Avg Rev / Load",        f"${df['AVG_REVENUE_PER_LOAD'].mean():,.0f}")
            k4.metric("Unique Customers (avg)", f"{df['UNIQUE_CUSTOMERS'].mean():,.0f}")

            # line chart — monthly revenue
            line = (
                alt.Chart(df, title="Monthly Revenue ($)")
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("MONTH:T", title="Month"),
                    y=alt.Y("REVENUE:Q", title="Revenue ($)", scale=alt.Scale(zero=False)),
                    tooltip=["MONTH:T", "REVENUE:Q", "TOTAL_LOADS:Q"],
                )
                .properties(height=370)
            )
            st.altair_chart(line, use_container_width=True)

            # raw table
            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fleet & Drivers  (trip performance view)
# ═════════════════════════════════════════════════════════════════════════════
with tab_fleet:
    st.subheader("Fleet & Driver Performance")

    # -- parameterized inputs --------------------------------------------------
    fl_c1, fl_c2, fl_c3 = st.columns(3)
    with fl_c1:
        min_trips = st.slider("Minimum trips", 1, 50, 5, key="fl_min_trips")
    with fl_c2:
        fuel_types = st.multiselect(
            "Fuel type", ["Diesel", "CNG", "Electric"], default=["Diesel", "CNG", "Electric"], key="fl_fuel"
        )
    with fl_c3:
        fleet_limit = st.slider("Top N trucks", 5, 100, 30, key="fl_limit")

    fuel_filter = ", ".join(f"'{safe(f)}'" for f in fuel_types) if fuel_types else "'Diesel'"

    sql_fleet = f"""
    SELECT
        TRUCK_ID,
        TRUCK_MAKE,
        TRUCK_YEAR,
        FUEL_TYPE,
        DRIVER_NAME,
        DRIVER_TERMINAL,
        COUNT(*)                                 AS trips,
        ROUND(SUM(ACTUAL_DISTANCE_MILES), 0)     AS total_miles,
        ROUND(AVG(AVERAGE_MPG), 2)               AS avg_mpg,
        ROUND(SUM(REVENUE), 2)                   AS total_revenue
    FROM CS5542_WEEK5.PUBLIC.V_TRIP_PERFORMANCE
    WHERE TRIP_STATUS = 'Completed'
      AND FUEL_TYPE IN ({fuel_filter})
    GROUP BY TRUCK_ID, TRUCK_MAKE, TRUCK_YEAR, FUEL_TYPE, DRIVER_NAME, DRIVER_TERMINAL
    HAVING COUNT(*) >= {min_trips}
    ORDER BY total_revenue DESC
    LIMIT {fleet_limit};
    """

    if st.button("Run Fleet Query", key="btn_fl"):
        df, ms = run_query(sql_fleet)
        log_event(team, user, "fleet_performance", ms, len(df))
        st.caption(f"⏱ {ms} ms  ·  {len(df)} rows")

        if df.empty:
            st.warning("No trucks match the current filters.")
        else:
            # bar chart — top trucks by revenue
            bar = (
                alt.Chart(df.head(15), title="Top Trucks by Revenue")
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("TOTAL_REVENUE:Q", title="Revenue ($)"),
                    y=alt.Y("TRUCK_ID:N", sort="-x", title="Truck"),
                    color=alt.Color("FUEL_TYPE:N", legend=alt.Legend(title="Fuel")),
                    tooltip=["TRUCK_ID:N", "TRUCK_MAKE:N", "TOTAL_REVENUE:Q", "TRIPS:Q", "AVG_MPG:Q"],
                )
                .properties(height=400)
            )
            st.altair_chart(bar, use_container_width=True)

            st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Routes  (route scorecard view)
# ═════════════════════════════════════════════════════════════════════════════
with tab_routes:
    st.subheader("Route Scorecard")

    # -- parameterized inputs --------------------------------------------------
    rt_c1, rt_c2, rt_c3 = st.columns(3)
    with rt_c1:
        min_loads = st.slider("Minimum loads per route", 1, 50, 5, key="rt_min")
    with rt_c2:
        margin_threshold = st.slider("Min margin %", -100, 100, 0, key="rt_margin")
    with rt_c3:
        routes_limit = st.slider("Top N routes", 5, 100, 20, key="rt_limit")

    sql_routes = f"""
    SELECT *
    FROM CS5542_WEEK5.PUBLIC.V_ROUTE_SCORECARD
    WHERE TOTAL_LOADS >= {min_loads}
      AND MARGIN_PCT >= {margin_threshold}
    ORDER BY GROSS_PROFIT DESC
    LIMIT {routes_limit};
    """

    if st.button("Run Routes Query", key="btn_rt"):
        df, ms = run_query(sql_routes)
        log_event(team, user, "route_scorecard", ms, len(df))
        st.caption(f"⏱ {ms} ms  ·  {len(df)} rows")

        if df.empty:
            st.warning("No routes match the current filters.")
        else:
            # KPI cards
            rk1, rk2, rk3 = st.columns(3)
            rk1.metric("Avg Margin %",   f"{df['MARGIN_PCT'].mean():.1f}%")
            rk2.metric("Total Revenue",  f"${df['TOTAL_REVENUE'].sum():,.0f}")
            rk3.metric("Avg MPG",        f"{df['AVG_MPG'].mean():.1f}")

            # horizontal bar — profitability
            bar = (
                alt.Chart(df.head(15), title="Route Gross Profit ($)")
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("GROSS_PROFIT:Q", title="Gross Profit ($)"),
                    y=alt.Y("ROUTE_LABEL:N", sort="-x", title="Route"),
                    color=alt.Color(
                        "MARGIN_PCT:Q",
                        scale=alt.Scale(scheme="redyellowgreen"),
                        legend=alt.Legend(title="Margin %"),
                    ),
                    tooltip=["ROUTE_LABEL:N", "TOTAL_LOADS:Q", "TOTAL_REVENUE:Q",
                             "TOTAL_FUEL_COST:Q", "GROSS_PROFIT:Q", "MARGIN_PCT:Q"],
                )
                .properties(height=420)
            )
            st.altair_chart(bar, use_container_width=True)

            st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Fuel Spend  (fuel spend by state/city)
# ═════════════════════════════════════════════════════════════════════════════
with tab_fuel:
    st.subheader("Fuel Spend by Location")

    # -- parameterized inputs --------------------------------------------------
    fu_c1, fu_c2 = st.columns(2)
    with fu_c1:
        state_filter = st.text_input("State abbreviation (comma-separated, or blank for all)", key="fu_state")
    with fu_c2:
        fuel_limit = st.slider("Top N locations", 5, 100, 25, key="fu_limit")

    state_where = ""
    if state_filter.strip():
        states = ", ".join(f"'{safe(s)}'" for s in state_filter.split(","))
        state_where = f"WHERE LOCATION_STATE IN ({states})"

    sql_fuel = f"""
    SELECT *
    FROM CS5542_WEEK5.PUBLIC.V_FUEL_SPEND
    {state_where}
    ORDER BY TOTAL_SPEND DESC
    LIMIT {fuel_limit};
    """

    if st.button("Run Fuel Query", key="btn_fu"):
        df, ms = run_query(sql_fuel)
        log_event(team, user, "fuel_spend", ms, len(df))
        st.caption(f"⏱ {ms} ms  ·  {len(df)} rows")

        if df.empty:
            st.warning("No fuel data for the selected states.")
        else:
            # KPI cards
            fk1, fk2, fk3 = st.columns(3)
            fk1.metric("Total Spend",     f"${df['TOTAL_SPEND'].sum():,.0f}")
            fk2.metric("Total Gallons",   f"{df['TOTAL_GALLONS'].sum():,.0f}")
            fk3.metric("Avg $/gal",       f"${df['AVG_PRICE_PER_GALLON'].mean():.3f}")

            # bar chart — spend by state
            state_agg = df.groupby("LOCATION_STATE", as_index=False).agg({"TOTAL_SPEND": "sum"}).sort_values("TOTAL_SPEND", ascending=False)
            bar = (
                alt.Chart(state_agg, title="Fuel Spend by State")
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("LOCATION_STATE:N", sort="-y", title="State"),
                    y=alt.Y("TOTAL_SPEND:Q",   title="Total Spend ($)"),
                    color=alt.Color("LOCATION_STATE:N", legend=None),
                    tooltip=["LOCATION_STATE:N", "TOTAL_SPEND:Q"],
                )
                .properties(height=370)
            )
            st.altair_chart(bar, use_container_width=True)

            st.dataframe(df, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Monitoring  (pipeline logs + performance analysis)
# ═════════════════════════════════════════════════════════════════════════════
with tab_monitor:
    st.subheader("Pipeline Monitoring")

    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0:
        logs = pd.read_csv(LOG_PATH)
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")

        # ── KPI cards ────────────────────────────────────────────────────
        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Total Queries",   f"{len(logs):,}")
        mk2.metric("Avg Latency",     f"{logs['latency_ms'].mean():,.0f} ms")
        mk3.metric("Max Latency",     f"{logs['latency_ms'].max():,.0f} ms")
        mk4.metric("Error Rate",      f"{(logs['error'].astype(bool).sum() / len(logs) * 100):.1f}%")

        # ── Latency over time chart ──────────────────────────────────────
        st.markdown("#### Latency Over Time")
        lat_chart = (
            alt.Chart(logs, title="Query Latency (ms)")
            .mark_circle(size=60)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("latency_ms:Q", title="Latency (ms)"),
                color=alt.Color("query_name:N", legend=alt.Legend(title="Query")),
                tooltip=["timestamp:T", "query_name:N", "latency_ms:Q", "rows_returned:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(lat_chart, use_container_width=True)

        # ── Per-query stats table ────────────────────────────────────────
        st.markdown("#### Per-Query Statistics")
        stats = (
            logs.groupby("query_name")
            .agg(
                runs=("latency_ms", "count"),
                avg_latency_ms=("latency_ms", "mean"),
                max_latency_ms=("latency_ms", "max"),
                avg_rows=("rows_returned", "mean"),
                errors=("error", lambda x: x.astype(bool).sum()),
            )
            .round(0)
            .reset_index()
            .sort_values("avg_latency_ms", ascending=False)
        )
        st.dataframe(stats, use_container_width=True, hide_index=True)

        # ── Performance summary note ─────────────────────────────────────
        st.markdown("#### Performance Notes")
        slowest = logs.loc[logs["latency_ms"].idxmax()]
        summary_lines = [
            f"- **Total queries logged:** {len(logs)}",
            f"- **Average latency:** {logs['latency_ms'].mean():,.0f} ms",
            f"- **Slowest query:** `{slowest['query_name']}` at {slowest['latency_ms']:,} ms",
        ]
        high_lat = logs[logs["latency_ms"] > 5000]
        if len(high_lat):
            summary_lines.append(f"- ⚠️ **{len(high_lat)} queries exceeded 5 s** — consider Snowflake warehouse scaling or adding filters to reduce scan volume.")
        else:
            summary_lines.append("- ✅ All queries completed under 5 s — no bottlenecks detected.")
        if logs["error"].astype(bool).any():
            summary_lines.append(f"- 🔴 **{logs['error'].astype(bool).sum()} errors** detected — check the log table below for details.")
        st.markdown("\n".join(summary_lines))

        # ── Full log table ───────────────────────────────────────────────
        st.markdown("#### Full Query Log")
        st.dataframe(logs.sort_values("timestamp", ascending=False).head(100), use_container_width=True, hide_index=True)
    else:
        st.info("No logs yet. Run a query from any tab to start recording.")
