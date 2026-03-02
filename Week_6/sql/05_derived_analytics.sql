-- 05_derived_analytics.sql
-- Advanced transformation & derived analytics tables.
-- Uses window functions, rankings, percentiles, CTEs, and cross-domain joins.
-- Run AFTER 01_create_schema.sql, 02_stage_and_load.sql, and 04_views.sql.

USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;

--------------------------------------------------------------------------------
-- DT1: DRIVER_PERFORMANCE_RANKED
--   Per-driver enriched metrics with window-function rankings and percentiles.
--   Combines trips, loads, fuel, incidents, and monthly metrics.
--   Adds: revenue_rank, mpg_rank, safety_score, percentile bands.
--------------------------------------------------------------------------------
CREATE OR REPLACE TABLE DT_DRIVER_PERFORMANCE_RANKED AS

WITH driver_trips AS (
    SELECT
        t.driver_id,
        COUNT(t.trip_id)                              AS total_trips,
        ROUND(SUM(t.actual_distance_miles), 0)        AS total_miles,
        ROUND(SUM(l.revenue), 2)                      AS total_revenue,
        ROUND(AVG(t.average_mpg), 3)                   AS avg_mpg,
        ROUND(SUM(t.fuel_gallons_used), 1)            AS total_fuel_gallons,
        ROUND(AVG(t.idle_time_hours), 2)              AS avg_idle_hours
    FROM TRIPS t
    JOIN LOADS l ON t.load_id = l.load_id
    WHERE t.trip_status = 'Completed'
      AND l.load_status  = 'Completed'
    GROUP BY t.driver_id
),
driver_incidents AS (
    SELECT
        driver_id,
        COUNT(*)                                       AS total_incidents,
        SUM(CASE WHEN preventable_flag THEN 1 ELSE 0 END) AS preventable_incidents,
        ROUND(SUM(claim_amount), 2)                    AS total_claim_amount
    FROM SAFETY_INCIDENTS
    GROUP BY driver_id
),
driver_ontime AS (
    SELECT
        driver_id,
        ROUND(AVG(on_time_delivery_rate) * 100, 1)    AS avg_on_time_pct
    FROM DRIVER_MONTHLY_METRICS
    GROUP BY driver_id
)

SELECT
    d.driver_id,
    d.first_name || ' ' || d.last_name                AS driver_name,
    d.home_terminal,
    d.cdl_class,
    d.years_experience,
    dt.total_trips,
    dt.total_miles,
    dt.total_revenue,
    dt.avg_mpg,
    dt.total_fuel_gallons,
    dt.avg_idle_hours,
    COALESCE(di.total_incidents, 0)                   AS total_incidents,
    COALESCE(di.preventable_incidents, 0)             AS preventable_incidents,
    COALESCE(di.total_claim_amount, 0)                AS total_claim_amount,
    COALESCE(dot.avg_on_time_pct, 0)                  AS avg_on_time_pct,

    -- Safety score: 100 minus 5 per incident (floor 0)
    GREATEST(0, 100 - (COALESCE(di.total_incidents, 0) * 5))
                                                       AS safety_score,

    -- Window rankings (lower ordinal = better)
    RANK() OVER (ORDER BY dt.total_revenue DESC)       AS revenue_rank,
    RANK() OVER (ORDER BY dt.avg_mpg     DESC)         AS mpg_rank,
    RANK() OVER (ORDER BY dt.total_miles DESC)         AS miles_rank,

    -- Percentile bands
    NTILE(4) OVER (ORDER BY dt.total_revenue DESC)     AS revenue_quartile,   -- 1=top
    NTILE(10) OVER (ORDER BY dt.avg_mpg     DESC)      AS mpg_decile,         -- 1=top

    -- Rolling 3-month revenue (requires ordered window; uses miles as proxy dimension)
    SUM(dt.total_revenue) OVER (
        ORDER BY dt.total_revenue DESC
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    )                                                  AS rolling3_revenue_sum

FROM DRIVERS d
JOIN driver_trips dt       ON d.driver_id = dt.driver_id
LEFT JOIN driver_incidents di ON d.driver_id = di.driver_id
LEFT JOIN driver_ontime    dot ON d.driver_id = dot.driver_id;


--------------------------------------------------------------------------------
-- DT2: TRUCK_HEALTH_SCORECARD
--   Per-truck derived table blending utilization, maintenance cost burden,
--   safety exposure, and MPG efficiency into a composite health score.
--------------------------------------------------------------------------------
CREATE OR REPLACE TABLE DT_TRUCK_HEALTH_SCORECARD AS

WITH truck_ops AS (
    SELECT
        t.truck_id,
        COUNT(t.trip_id)                              AS total_trips,
        ROUND(SUM(t.actual_distance_miles), 0)        AS total_miles,
        ROUND(AVG(t.average_mpg), 3)                   AS avg_mpg,
        ROUND(SUM(t.fuel_gallons_used), 1)            AS total_fuel_gallons,
        ROUND(SUM(l.revenue), 2)                      AS total_revenue
    FROM TRIPS t
    JOIN LOADS l ON t.load_id = l.load_id
    WHERE t.trip_status = 'Completed'
    GROUP BY t.truck_id
),
truck_maint AS (
    SELECT
        truck_id,
        COUNT(*)                                       AS maintenance_events,
        ROUND(SUM(total_cost), 2)                      AS total_maintenance_cost,
        ROUND(SUM(downtime_hours), 1)                  AS total_downtime_hours,
        MAX(maintenance_date)                          AS last_maintenance_date
    FROM MAINTENANCE_RECORDS
    GROUP BY truck_id
),
truck_safety AS (
    SELECT
        truck_id,
        COUNT(*)                                       AS incident_count,
        ROUND(SUM(vehicle_damage_cost), 2)             AS total_damage_cost
    FROM SAFETY_INCIDENTS
    GROUP BY truck_id
),
truck_util AS (
    SELECT
        truck_id,
        ROUND(AVG(utilization_rate) * 100, 1)          AS avg_utilization_pct,
        ROUND(AVG(downtime_hours), 1)                  AS avg_monthly_downtime
    FROM TRUCK_UTILIZATION_METRICS
    GROUP BY truck_id
)

SELECT
    tk.truck_id,
    tk.make,
    tk.model_year,
    tk.fuel_type,
    tk.status,
    tk.home_terminal,

    -- Operational metrics
    COALESCE(o.total_trips, 0)                        AS total_trips,
    COALESCE(o.total_miles, 0)                        AS total_miles,
    COALESCE(o.avg_mpg, 0)                            AS avg_mpg,
    COALESCE(o.total_revenue, 0)                      AS total_revenue,

    -- Maintenance metrics
    COALESCE(m.maintenance_events, 0)                 AS maintenance_events,
    COALESCE(m.total_maintenance_cost, 0)             AS total_maintenance_cost,
    COALESCE(m.total_downtime_hours, 0)               AS total_downtime_hours,
    m.last_maintenance_date,

    -- Safety metrics
    COALESCE(s.incident_count, 0)                     AS incident_count,
    COALESCE(s.total_damage_cost, 0)                  AS total_damage_cost,

    -- Utilization
    COALESCE(u.avg_utilization_pct, 0)                AS avg_utilization_pct,
    COALESCE(u.avg_monthly_downtime, 0)               AS avg_monthly_downtime,

    -- Cost per mile (maintenance / operational miles)
    ROUND(
        COALESCE(m.total_maintenance_cost, 0)
        / NULLIF(COALESCE(o.total_miles, 0), 0),
        4
    )                                                  AS maintenance_cost_per_mile,

    -- Composite health score (0–100):
    --   40% MPG efficiency (vs fleet avg ~6.5 mpg)
    --   30% low downtime  (vs max 500 h)
    --   30% low incident rate (0 incidents = 30 pts)
    ROUND(
        LEAST(40, (COALESCE(o.avg_mpg, 0) / 6.5) * 40)
        + LEAST(30, (1 - COALESCE(u.avg_monthly_downtime, 0) / 500.0) * 30)
        + GREATEST(0, 30 - COALESCE(s.incident_count, 0) * 10),
        1
    )                                                  AS health_score,

    -- Fleet-wide window rankings
    RANK() OVER (ORDER BY COALESCE(o.total_revenue, 0)          DESC) AS revenue_rank,
    RANK() OVER (ORDER BY COALESCE(o.avg_mpg, 0)                DESC) AS mpg_rank,
    RANK() OVER (ORDER BY COALESCE(m.total_maintenance_cost, 0) ASC)  AS low_maint_cost_rank,
    NTILE(4) OVER (ORDER BY COALESCE(o.avg_mpg, 0) DESC)              AS mpg_quartile

FROM TRUCKS tk
LEFT JOIN truck_ops    o ON tk.truck_id = o.truck_id
LEFT JOIN truck_maint  m ON tk.truck_id = m.truck_id
LEFT JOIN truck_safety s ON tk.truck_id = s.truck_id
LEFT JOIN truck_util   u ON tk.truck_id = u.truck_id;


--------------------------------------------------------------------------------
-- DT3: ROUTE_DELIVERY_QUALITY
--   Per-route on-time performance derived from DELIVERY_EVENTS.
--   Joins to route and load data; computes detention burden and on-time rate.
--   Window: percentile rank of detention burden across all routes.
--------------------------------------------------------------------------------
CREATE OR REPLACE TABLE DT_ROUTE_DELIVERY_QUALITY AS

WITH event_agg AS (
    SELECT
        de.load_id,
        SUM(de.detention_minutes)                      AS total_detention_min,
        COUNT(*)                                       AS total_events,
        SUM(CASE WHEN de.on_time_flag THEN 1 ELSE 0 END) AS on_time_events,
        SUM(CASE WHEN de.event_type = 'Pickup'   THEN 1 ELSE 0 END) AS pickups,
        SUM(CASE WHEN de.event_type = 'Delivery' THEN 1 ELSE 0 END) AS deliveries
    FROM DELIVERY_EVENTS de
    GROUP BY de.load_id
)

SELECT
    r.route_id,
    r.origin_city    || ', ' || r.origin_state
        || ' → '
        || r.destination_city || ', ' || r.destination_state   AS route_label,
    r.typical_distance_miles,

    COUNT(DISTINCT l.load_id)                                  AS total_loads,
    ROUND(SUM(l.revenue), 2)                                   AS total_revenue,
    ROUND(AVG(l.revenue), 2)                                   AS avg_revenue_per_load,

    -- Delivery quality
    ROUND(AVG(ea.total_detention_min), 1)                      AS avg_detention_min,
    ROUND(SUM(ea.on_time_events) * 100.0
          / NULLIF(SUM(ea.total_events), 0), 1)                AS on_time_event_pct,

    -- Window: how this route's detention compares to all routes
    RANK() OVER (ORDER BY AVG(ea.total_detention_min) ASC)    AS low_detention_rank,
    PERCENT_RANK() OVER (ORDER BY AVG(ea.total_detention_min) ASC) AS detention_pct_rank,

    -- Revenue efficiency per mile
    ROUND(SUM(l.revenue) / NULLIF(r.typical_distance_miles, 0), 2) AS revenue_per_mile

FROM LOADS l
JOIN ROUTES r ON l.route_id = r.route_id
LEFT JOIN event_agg ea ON l.load_id = ea.load_id
WHERE l.load_status = 'Completed'
GROUP BY r.route_id, r.origin_city, r.origin_state,
         r.destination_city, r.destination_state, r.typical_distance_miles
HAVING COUNT(DISTINCT l.load_id) >= 3;


--------------------------------------------------------------------------------
-- DT4: MONTHLY_OPERATIONS_SUMMARY
--   Cross-domain monthly rollup: revenue, loads, trips, fuel cost, incidents,
--   maintenance cost, and MoM growth rates using LAG().
--------------------------------------------------------------------------------
CREATE OR REPLACE TABLE DT_MONTHLY_OPERATIONS_SUMMARY AS

WITH monthly_revenue AS (
    SELECT
        DATE_TRUNC('MONTH', load_date)     AS month,
        COUNT(load_id)                     AS loads,
        ROUND(SUM(revenue), 2)             AS load_revenue,
        ROUND(SUM(fuel_surcharge), 2)      AS fuel_surcharge_revenue
    FROM LOADS
    WHERE load_status = 'Completed'
    GROUP BY DATE_TRUNC('MONTH', load_date)
),
monthly_fuel AS (
    SELECT
        DATE_TRUNC('MONTH', purchase_date) AS month,
        ROUND(SUM(total_cost), 2)          AS fuel_cost,
        ROUND(SUM(gallons), 1)             AS fuel_gallons
    FROM FUEL_PURCHASES
    GROUP BY DATE_TRUNC('MONTH', purchase_date)
),
monthly_maint AS (
    SELECT
        DATE_TRUNC('MONTH', maintenance_date) AS month,
        ROUND(SUM(total_cost), 2)             AS maintenance_cost,
        COUNT(*)                              AS maintenance_events
    FROM MAINTENANCE_RECORDS
    GROUP BY DATE_TRUNC('MONTH', maintenance_date)
),
monthly_incidents AS (
    SELECT
        DATE_TRUNC('MONTH', incident_date::DATE) AS month,
        COUNT(*)                                  AS incidents,
        ROUND(SUM(claim_amount), 2)               AS claim_amount
    FROM SAFETY_INCIDENTS
    GROUP BY DATE_TRUNC('MONTH', incident_date::DATE)
)

SELECT
    mr.month,
    mr.loads,
    mr.load_revenue,
    mr.fuel_surcharge_revenue,
    COALESCE(mf.fuel_cost, 0)             AS fuel_cost,
    COALESCE(mf.fuel_gallons, 0)          AS fuel_gallons,
    COALESCE(mm.maintenance_cost, 0)      AS maintenance_cost,
    COALESCE(mm.maintenance_events, 0)    AS maintenance_events,
    COALESCE(mi.incidents, 0)             AS incidents,
    COALESCE(mi.claim_amount, 0)          AS claim_amount,

    -- Derived: net operating margin estimate
    ROUND(
        mr.load_revenue
        - COALESCE(mf.fuel_cost, 0)
        - COALESCE(mm.maintenance_cost, 0)
        - COALESCE(mi.claim_amount, 0),
        2
    )                                     AS est_net_margin,

    -- MoM growth rates using LAG
    ROUND(
        (mr.load_revenue - LAG(mr.load_revenue) OVER (ORDER BY mr.month))
        / NULLIF(LAG(mr.load_revenue) OVER (ORDER BY mr.month), 0) * 100,
        2
    )                                     AS revenue_mom_pct,

    ROUND(
        (mr.loads - LAG(mr.loads) OVER (ORDER BY mr.month))
        / NULLIF(LAG(mr.loads) OVER (ORDER BY mr.month), 0) * 100,
        2
    )                                     AS loads_mom_pct,

    -- 3-month rolling average revenue
    ROUND(AVG(mr.load_revenue) OVER (
        ORDER BY mr.month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                 AS revenue_3mo_avg

FROM monthly_revenue mr
LEFT JOIN monthly_fuel      mf ON mr.month = mf.month
LEFT JOIN monthly_maint     mm ON mr.month = mm.month
LEFT JOIN monthly_incidents mi ON mr.month = mi.month
ORDER BY mr.month;