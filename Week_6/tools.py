import os
import csv
import pandas as pd
import pathlib
import sys
from typing import List, Dict, Any, Optional

# Setup path to import sf_connect
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from sf_connect import get_conn

LOG_PATH = ROOT / "logs" / "pipeline_logs.csv"

def query_snowflake(sql_query: str) -> List[Dict[str, Any]]:
    """Executes a read-only SQL query against the Snowflake database.
    
    Args:
        sql_query (str): The SQL query string to execute.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the query results.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description] if cur.description else []
                
        # Convert to a list of dicts for JSON serialization
        df = pd.DataFrame(rows, columns=cols)
        # Convert datetime objects to string using ISO format
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Failed to execute query: {str(e)}"}]


def get_monthly_revenue(start_month: str, end_month: str) -> List[Dict[str, Any]]:
    """Retrieves aggregated monthly revenue trends within a specified date range.
    
    Args:
        start_month (str): Start month in 'YYYY-MM-DD' format (e.g. '2023-01-01').
        end_month (str): End month in 'YYYY-MM-DD' format (e.g. '2025-12-31').
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing monthly revenue data.
    """
    sql_overview = f"""
    SELECT *
    FROM CS5542_WEEK5.PUBLIC.V_MONTHLY_REVENUE
    WHERE MONTH >= '{start_month}'
      AND MONTH <= '{end_month}'
    ORDER BY MONTH;
    """
    return query_snowflake(sql_overview)


def get_fleet_performance(min_trips: int = 5, top_n: int = 30, fuel_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Retrieves truck performance metrics based on specified filters.
    
    Args:
        min_trips (int): Minimum number of trips completed by a truck to be included. Defaults to 5.
        top_n (int): Maximum number of top-performing trucks to return. Defaults to 30.
        fuel_types (Optional[List[str]]): List of fuel types to include (e.g. ['Diesel', 'CNG', 'Electric']). Defaults to all if None.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing truck performance metrics.
    """
    if not fuel_types:
        fuel_types = ["Diesel", "CNG", "Electric"]
        
    # SQL minimal escape
    safe = lambda t: str(t).strip().replace("'", "''")
    fuel_filter = ", ".join(f"'{safe(f)}'" for f in fuel_types)
    
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
    LIMIT {top_n};
    """
    return query_snowflake(sql_fleet)


def get_pipeline_logs(limit: int = 100) -> List[Dict[str, Any]]:
    """Reads the automated ingestion pipeline logs to return system health and latency data.
    
    Args:
        limit (int): Maximum number of recent log entries to return. Defaults to 100.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing log entries.
    """
    try:
        if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
            return [{"error": "Pipeline logs do not exist or are empty."}]
            
        logs_df = pd.read_csv(LOG_PATH)
        # Sort so most recent logs are first
        if "timestamp" in logs_df.columns:
            logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
            logs_df = logs_df.sort_values(by="timestamp", ascending=False)
            logs_df["timestamp"] = logs_df["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
        logs_df = logs_df.head(limit)
        
        # Replace NaN with None
        logs_df = logs_df.where(pd.notnull(logs_df), None)
        return logs_df.to_dict(orient="records")
    except Exception as e:
        return [{"error": f"Failed to read logs: {str(e)}"}]


def get_safety_metrics(
    incident_types: Optional[List[str]] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    limit: int = 15
) -> List[Dict[str, Any]]:
    """Retrieves safety incident metrics for top drivers based on filters.
    
    Args:
        incident_types (Optional[List[str]]): List of incident types to include (e.g. ['Collision', 'Near Miss']). Defaults to all if None.
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'.
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'.
        limit (int): Maximum number of top drivers by incident count to return. Defaults to 15.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing driver safety metrics.
    """
    if not incident_types:
        incident_types = [
            "Moving Violation", "Collision", "Equipment Failure",
            "Near Miss", "Cargo Damage", "DOT Inspection"
        ]
        
    # SQL minimal escape
    safe = lambda t: str(t).strip().replace("'", "''")
    type_filter = ", ".join(f"'{safe(t)}'" for t in incident_types)
    
    sql = f"""
    SELECT
        d.first_name || ' ' || d.last_name   AS driver_name,
        d.home_terminal,
        COUNT(si.incident_id)                AS total_incidents,
        ROUND(SUM(si.claim_amount), 0)       AS total_claims,
        SUM(IFF(si.at_fault_flag, 1, 0))    AS at_fault_count,
        SUM(IFF(si.injury_flag, 1, 0))      AS injury_count
    FROM CS5542_WEEK5.PUBLIC.SAFETY_INCIDENTS si
    JOIN CS5542_WEEK5.PUBLIC.DRIVERS d ON si.driver_id = d.driver_id
    WHERE si.incident_type IN ({type_filter})
      AND CAST(si.incident_date AS DATE) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY driver_name, d.home_terminal
    ORDER BY total_incidents DESC
    LIMIT {limit};
    """
    return query_snowflake(sql)


def get_route_profitability(
    min_loads: int = 3, min_margin_pct: float = 0.0, top_n: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves route profitability metrics including revenue, fuel cost, and margin.

    Args:
        min_loads (int): Minimum completed loads for a route to be included. Defaults to 3.
        min_margin_pct (float): Minimum gross margin percentage. Defaults to 0.0.
        top_n (int): Max routes to return, sorted by gross profit. Defaults to 20.

    Returns:
        List[Dict[str, Any]]: List of dicts with route_label, total_loads, total_revenue,
            gross_profit, margin_pct, avg_mpg.
    """
    sql = f"""
    SELECT route_label, total_loads, total_revenue, total_fuel_cost,
           gross_profit, margin_pct, avg_mpg
    FROM CS5542_WEEK5.PUBLIC.V_ROUTE_SCORECARD
    WHERE total_loads >= {min_loads} AND margin_pct >= {min_margin_pct}
    ORDER BY gross_profit DESC LIMIT {top_n};
    """
    return query_snowflake(sql)


def get_delivery_performance(
    event_type: str = "Delivery",
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves delivery event performance including on-time rates and detention times.

    Args:
        event_type (str): 'Delivery' or 'Pickup'. Defaults to 'Delivery'.
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'.
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to '2025-12-31'.
        limit (int): Max rows to return. Defaults to 20.

    Returns:
        List[Dict[str, Any]]: List of dicts with city, state, event counts,
            on-time rate, avg detention minutes.
    """
    safe_type = event_type.replace("'", "''")
    sql = f"""
    SELECT de.location_city, de.location_state, COUNT(*) AS total_events,
           ROUND(AVG(CASE WHEN de.on_time_flag THEN 1 ELSE 0 END)*100,1) AS on_time_pct,
           ROUND(AVG(de.detention_minutes),1) AS avg_detention_min
    FROM CS5542_WEEK5.PUBLIC.DELIVERY_EVENTS de
    WHERE de.event_type = '{safe_type}'
      AND CAST(de.scheduled_datetime AS DATE) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY de.location_city, de.location_state
    ORDER BY total_events DESC LIMIT {limit};
    """
    return query_snowflake(sql)

def get_maintenance_health(
    maintenance_type: Optional[str] = None, start_date: str = "2022-01-01",
    end_date: str = "2025-12-31", top_n: int = 20
) -> List[Dict[str, Any]]:
    """Retrieves truck maintenance health metrics including costs, downtime, and event counts.
    Args:
        maintenance_type: 'Scheduled', 'Unscheduled', or 'Inspection'. None = all.
        start_date/end_date: Date range in 'YYYY-MM-DD' format.
        top_n: Max trucks to return, sorted by total cost. Defaults to 20.
    Returns:
        List of dicts with truck_id, make, model_year, maintenance_events, total_cost, avg_downtime.
    """
    type_clause = ""
    if maintenance_type:
        safe = maintenance_type.replace("'", "''")
        type_clause = f"AND mr.maintenance_type = '{safe}'"
    sql = f"""
    SELECT mr.truck_id, tk.make, tk.model_year, tk.fuel_type,
           COUNT(*) AS maintenance_events,
           ROUND(SUM(mr.total_cost),2) AS total_cost,
           ROUND(AVG(mr.downtime_hours),1) AS avg_downtime_hours,
           ROUND(SUM(mr.labor_cost),2) AS total_labor_cost,
           ROUND(SUM(mr.parts_cost),2) AS total_parts_cost
    FROM CS5542_WEEK5.PUBLIC.MAINTENANCE_RECORDS mr
    JOIN CS5542_WEEK5.PUBLIC.TRUCKS tk ON mr.truck_id = tk.truck_id
    WHERE mr.maintenance_date BETWEEN '{start_date}' AND '{end_date}' {type_clause}
    GROUP BY mr.truck_id, tk.make, tk.model_year, tk.fuel_type
    ORDER BY total_cost DESC LIMIT {top_n};
    """
    return query_snowflake(sql)

def get_fuel_spend_analysis(
    states: Optional[List[str]] = None, top_n: int = 15
) -> List[Dict[str, Any]]:
    """Retrieves fuel spend analysis aggregated by state and city.
    Args:
        states: State abbreviations to filter (e.g. ['TX','CA']). None = all.
        top_n: Max locations to return. Defaults to 15.
    Returns:
        List of dicts with state, city, total spend, gallons, avg price per gallon.
    """
    state_clause = ""
    if states:
        safe = lambda s: str(s).strip().replace("'", "''")
        state_filter = ", ".join(f"'{safe(s)}'" for s in states)
        state_clause = f"WHERE location_state IN ({state_filter})"
    sql = f"""
    SELECT location_state, location_city, purchases,
           total_gallons, avg_price_per_gallon, total_spend
    FROM CS5542_WEEK5.PUBLIC.V_FUEL_SPEND {state_clause}
    ORDER BY total_spend DESC LIMIT {top_n};
    """
    return query_snowflake(sql)