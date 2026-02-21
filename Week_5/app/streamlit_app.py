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
    st.caption("Queries hit views (`04_views.sql`) and derived tables (`05_derived_analytics.sql`).")

# ── title ────────────────────────────────────────────────────────────────────
st.title("🚛 CS 5542 — Trucking Logistics Dashboard")
st.caption("Live connection to **Snowflake** · parameterized inputs · Altair charts")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_fleet, tab_routes, tab_fuel, tab_monitor, tab_analytics = st.tabs(
    ["📊 Overview", "🚛 Fleet & Drivers", "🗺️ Routes", "⛽ Fuel Spend", "📈 Monitoring", "🔬 Analytics"]
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

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — Analytics  (derived tables from 05_derived_analytics.sql)
# ═════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.subheader("Advanced Derived Analytics")
    st.caption("Queries run against the 4 materialized tables built by `sql/05_derived_analytics.sql`. "
               "Run that script in Snowflake once after loading data.")

    an1, an2, an3, an4 = st.tabs([
        "🧑‍✈️ Driver Rankings",
        "🚛 Truck Health",
        "🗺️ Route Quality",
        "📅 Monthly Ops",
    ])

    # ── Sub-tab 1: Driver Performance Ranked ─────────────────────────────────
    with an1:
        st.markdown("#### Driver Performance — Ranked (`DT_DRIVER_PERFORMANCE_RANKED`)")
        an1c1, an1c2 = st.columns(2)
        with an1c1:
            revenue_quartile = st.multiselect(
                "Revenue quartile (1 = top 25%)",
                [1, 2, 3, 4], default=[1, 2], key="an1_q"
            )
        with an1c2:
            an1_limit = st.slider("Top N drivers", 5, 150, 30, key="an1_lim")

        q_filter = ", ".join(str(q) for q in revenue_quartile) if revenue_quartile else "1,2,3,4"
        sql_driver = f"""
        SELECT
            driver_name, home_terminal, cdl_class, years_experience,
            total_trips, total_miles, total_revenue,
            avg_mpg, avg_on_time_pct, safety_score,
            revenue_rank, mpg_rank, revenue_quartile, mpg_decile
        FROM CS5542_WEEK5.PUBLIC.DT_DRIVER_PERFORMANCE_RANKED
        WHERE revenue_quartile IN ({q_filter})
        ORDER BY revenue_rank ASC
        LIMIT {an1_limit};
        """
        if st.button("Run Driver Rankings", key="btn_an1"):
            df, ms = run_query(sql_driver)
            log_event(team, user, "analytics_driver_ranked", ms, len(df))
            st.caption(f"⏱ {ms} ms · {len(df)} rows")
            if df.empty:
                st.warning("No data — have you run `05_derived_analytics.sql` in Snowflake?")
            else:
                ak1, ak2, ak3 = st.columns(3)
                ak1.metric("Avg Safety Score", f"{df['SAFETY_SCORE'].mean():.1f} / 100")
                ak2.metric("Avg On-Time %",    f"{df['AVG_ON_TIME_PCT'].mean():.1f}%")
                ak3.metric("Avg MPG",           f"{df['AVG_MPG'].mean():.2f}")

                bar = (
                    alt.Chart(df.head(20), title="Top Drivers by Revenue ($)")
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("TOTAL_REVENUE:Q", title="Revenue ($)"),
                        y=alt.Y("DRIVER_NAME:N", sort="-x", title="Driver"),
                        color=alt.Color("SAFETY_SCORE:Q",
                                        scale=alt.Scale(scheme="redyellowgreen"),
                                        legend=alt.Legend(title="Safety Score")),
                        tooltip=["DRIVER_NAME:N", "TOTAL_REVENUE:Q",
                                 "AVG_MPG:Q", "SAFETY_SCORE:Q", "AVG_ON_TIME_PCT:Q"],
                    ).properties(height=420)
                )
                st.altair_chart(bar, use_container_width=True)
                with st.expander("Full table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Sub-tab 2: Truck Health Scorecard ────────────────────────────────────
    with an2:
        st.markdown("#### Truck Health Scorecard (`DT_TRUCK_HEALTH_SCORECARD`)")
        an2c1, an2c2, an2c3 = st.columns(3)
        with an2c1:
            fuel_type_f = st.multiselect(
                "Fuel type", ["Diesel", "CNG", "Electric"],
                default=["Diesel", "CNG", "Electric"], key="an2_ft"
            )
        with an2c2:
            min_health = st.slider("Min health score", 0, 100, 40, key="an2_hs")
        with an2c3:
            an2_limit = st.slider("Top N trucks", 5, 120, 30, key="an2_lim")

        ft_filter = ", ".join(f"'{safe(f)}'" for f in fuel_type_f) if fuel_type_f else "'Diesel'"
        sql_truck = f"""
        SELECT
            truck_id, make, model_year, fuel_type, status, home_terminal,
            total_trips, total_miles, avg_mpg, total_revenue,
            maintenance_events, total_maintenance_cost, maintenance_cost_per_mile,
            incident_count, avg_utilization_pct, health_score,
            revenue_rank, mpg_quartile
        FROM CS5542_WEEK5.PUBLIC.DT_TRUCK_HEALTH_SCORECARD
        WHERE fuel_type IN ({ft_filter})
          AND health_score >= {min_health}
        ORDER BY health_score DESC
        LIMIT {an2_limit};
        """
        if st.button("Run Truck Health", key="btn_an2"):
            df, ms = run_query(sql_truck)
            log_event(team, user, "analytics_truck_health", ms, len(df))
            st.caption(f"⏱ {ms} ms · {len(df)} rows")
            if df.empty:
                st.warning("No data — have you run `05_derived_analytics.sql` in Snowflake?")
            else:
                bk1, bk2, bk3 = st.columns(3)
                bk1.metric("Avg Health Score",    f"{df['HEALTH_SCORE'].mean():.1f} / 100")
                bk2.metric("Avg Utilization",     f"{df['AVG_UTILIZATION_PCT'].mean():.1f}%")
                bk3.metric("Avg Maint Cost/Mile", f"${df['MAINTENANCE_COST_PER_MILE'].mean():.4f}")

                scatter = (
                    alt.Chart(df, title="Health Score vs MPG (bubble = revenue)")
                    .mark_circle()
                    .encode(
                        x=alt.X("AVG_MPG:Q", title="Avg MPG"),
                        y=alt.Y("HEALTH_SCORE:Q", title="Health Score"),
                        size=alt.Size("TOTAL_REVENUE:Q", legend=alt.Legend(title="Revenue ($)")),
                        color=alt.Color("FUEL_TYPE:N", legend=alt.Legend(title="Fuel")),
                        tooltip=["TRUCK_ID:N", "MAKE:N", "MODEL_YEAR:Q",
                                 "HEALTH_SCORE:Q", "AVG_MPG:Q", "TOTAL_REVENUE:Q"],
                    ).properties(height=380)
                )
                st.altair_chart(scatter, use_container_width=True)
                with st.expander("Full table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Sub-tab 3: Route Delivery Quality ────────────────────────────────────
    with an3:
        st.markdown("#### Route Delivery Quality (`DT_ROUTE_DELIVERY_QUALITY`)")
        an3c1, an3c2 = st.columns(2)
        with an3c1:
            min_ontime = st.slider("Min on-time event %", 0, 100, 0, key="an3_ot")
        with an3c2:
            an3_limit  = st.slider("Top N routes", 5, 100, 20, key="an3_lim")

        sql_route_q = f"""
        SELECT
            route_label, typical_distance_miles,
            total_loads, total_revenue, avg_revenue_per_load,
            avg_detention_min, on_time_event_pct,
            revenue_per_mile, low_detention_rank,
            ROUND(detention_pct_rank * 100, 1) AS detention_pct_rank
        FROM CS5542_WEEK5.PUBLIC.DT_ROUTE_DELIVERY_QUALITY
        WHERE on_time_event_pct >= {min_ontime}
        ORDER BY revenue_per_mile DESC
        LIMIT {an3_limit};
        """
        if st.button("Run Route Quality", key="btn_an3"):
            df, ms = run_query(sql_route_q)
            log_event(team, user, "analytics_route_quality", ms, len(df))
            st.caption(f"⏱ {ms} ms · {len(df)} rows")
            if df.empty:
                st.warning("No data — have you run `05_derived_analytics.sql` in Snowflake?")
            else:
                ck1, ck2, ck3 = st.columns(3)
                ck1.metric("Avg On-Time %",     f"{df['ON_TIME_EVENT_PCT'].mean():.1f}%")
                ck2.metric("Avg Detention",      f"{df['AVG_DETENTION_MIN'].mean():.0f} min")
                ck3.metric("Avg Rev / Mile",     f"${df['REVENUE_PER_MILE'].mean():.2f}")

                bar = (
                    alt.Chart(df.head(15), title="Revenue per Mile by Route")
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("REVENUE_PER_MILE:Q", title="Revenue / Mile ($)"),
                        y=alt.Y("ROUTE_LABEL:N", sort="-x", title="Route"),
                        color=alt.Color(
                            "ON_TIME_EVENT_PCT:Q",
                            scale=alt.Scale(scheme="redyellowgreen"),
                            legend=alt.Legend(title="On-Time %")
                        ),
                        tooltip=["ROUTE_LABEL:N", "REVENUE_PER_MILE:Q",
                                 "ON_TIME_EVENT_PCT:Q", "AVG_DETENTION_MIN:Q",
                                 "TOTAL_LOADS:Q"],
                    ).properties(height=420)
                )
                st.altair_chart(bar, use_container_width=True)
                with st.expander("Full table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Sub-tab 4: Monthly Operations Summary ────────────────────────────────
    with an4:
        st.markdown("#### Monthly Operations Summary (`DT_MONTHLY_OPERATIONS_SUMMARY`)")
        an4c1, an4c2 = st.columns(2)
        with an4c1:
            mo_start = st.date_input("From month", value=pd.Timestamp("2022-01-01"), key="an4_s")
        with an4c2:
            mo_end   = st.date_input("To month",   value=pd.Timestamp("2025-12-31"), key="an4_e")

        sql_monthly = f"""
        SELECT
            month, loads, load_revenue, fuel_cost,
            maintenance_cost, incidents, claim_amount,
            est_net_margin, revenue_mom_pct, loads_mom_pct, revenue_3mo_avg
        FROM CS5542_WEEK5.PUBLIC.DT_MONTHLY_OPERATIONS_SUMMARY
        WHERE month >= '{mo_start}'
          AND month <= '{mo_end}'
        ORDER BY month;
        """
        if st.button("Run Monthly Ops", key="btn_an4"):
            df, ms = run_query(sql_monthly)
            log_event(team, user, "analytics_monthly_ops", ms, len(df))
            st.caption(f"⏱ {ms} ms · {len(df)} rows")
            if df.empty:
                st.warning("No data — have you run `05_derived_analytics.sql` in Snowflake?")
            else:
                dk1, dk2, dk3, dk4 = st.columns(4)
                dk1.metric("Total Revenue",    f"${df['LOAD_REVENUE'].sum():,.0f}")
                dk2.metric("Total Fuel Cost",  f"${df['FUEL_COST'].sum():,.0f}")
                dk3.metric("Total Incidents",  f"{df['INCIDENTS'].sum():,}")
                dk4.metric("Avg Net Margin/mo", f"${df['EST_NET_MARGIN'].mean():,.0f}")

                # Multi-line chart: revenue, 3-mo avg, net margin
                df_melted = df.melt(
                    id_vars="MONTH",
                    value_vars=["LOAD_REVENUE", "REVENUE_3MO_AVG", "EST_NET_MARGIN"],
                    var_name="Metric", value_name="Value"
                )
                line = (
                    alt.Chart(df_melted, title="Monthly Revenue, 3-Mo Avg & Net Margin")
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("MONTH:T", title="Month"),
                        y=alt.Y("Value:Q",  title="USD ($)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Metric:N", legend=alt.Legend(title="Series")),
                        tooltip=["MONTH:T", "Metric:N", "Value:Q"],
                    ).properties(height=370)
                )
                st.altair_chart(line, use_container_width=True)

                # MoM growth bar
                mom = df.dropna(subset=["REVENUE_MOM_PCT"])
                if not mom.empty:
                    st.markdown("##### Month-over-Month Revenue Growth %")
                    mom_bar = (
                        alt.Chart(mom)
                        .mark_bar()
                        .encode(
                            x=alt.X("MONTH:T", title="Month"),
                            y=alt.Y("REVENUE_MOM_PCT:Q", title="MoM Growth (%)"),
                            color=alt.condition(
                                alt.datum.REVENUE_MOM_PCT > 0,
                                alt.value("#2ecc71"), alt.value("#e74c3c")
                            ),
                            tooltip=["MONTH:T", "REVENUE_MOM_PCT:Q", "LOADS_MOM_PCT:Q"],
                        ).properties(height=220)
                    )
                    st.altair_chart(mom_bar, use_container_width=True)

                with st.expander("Full table"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
