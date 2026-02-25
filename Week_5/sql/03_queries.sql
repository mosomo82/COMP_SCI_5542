-- 03_queries.sql
-- Five analytical queries across the 7-table trucking subset.
-- Run after data is loaded.

USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;

--------------------------------------------------------------------------------
-- Q1: Revenue by customer (aggregation + group-by + join)
--     Top 20 customers by total completed-load revenue.
--------------------------------------------------------------------------------
SELECT
    c.customer_id,
    c.customer_name,
    c.customer_type,
    COUNT(l.load_id)            AS total_loads,
    ROUND(SUM(l.revenue), 2)   AS total_revenue,
    ROUND(AVG(l.revenue), 2)   AS avg_revenue_per_load
FROM LOADS l
JOIN CUSTOMERS c ON l.customer_id = c.customer_id
WHERE l.load_status = 'Completed'
GROUP BY c.customer_id, c.customer_name, c.customer_type
ORDER BY total_revenue DESC
LIMIT 20;

--------------------------------------------------------------------------------
-- Q2: Driver fuel efficiency (aggregation + join + HAVING filter)
--     Average MPG and total miles per driver (min 10 trips), ranked.
--------------------------------------------------------------------------------
SELECT
    d.driver_id,
    d.first_name || ' ' || d.last_name  AS driver_name,
    d.home_terminal,
    COUNT(t.trip_id)                          AS total_trips,
    ROUND(SUM(t.actual_distance_miles), 0)    AS total_miles,
    ROUND(AVG(t.average_mpg), 2)              AS avg_mpg,
    ROUND(SUM(t.fuel_gallons_used), 1)        AS total_fuel_gallons
FROM TRIPS t
JOIN DRIVERS d ON t.driver_id = d.driver_id
WHERE t.trip_status = 'Completed'
GROUP BY d.driver_id, d.first_name, d.last_name, d.home_terminal
HAVING COUNT(t.trip_id) >= 10
ORDER BY avg_mpg DESC
LIMIT 25;

--------------------------------------------------------------------------------
-- Q3: Route profitability (4-table join)
--     Revenue minus fuel cost per route, with margin %.
--     Joins: LOADS → ROUTES, LOADS → TRIPS → FUEL_PURCHASES
--------------------------------------------------------------------------------
SELECT
    r.route_id,
    r.origin_city || ', ' || r.origin_state
        || ' → '
        || r.destination_city || ', ' || r.destination_state   AS route,
    r.typical_distance_miles,
    COUNT(DISTINCT l.load_id)                                  AS total_loads,
    ROUND(SUM(l.revenue + l.fuel_surcharge), 2)                AS total_revenue,
    ROUND(SUM(fp.total_cost), 2)                               AS total_fuel_cost,
    ROUND(SUM(l.revenue + l.fuel_surcharge) - SUM(fp.total_cost), 2) AS gross_profit,
    ROUND(
        (SUM(l.revenue + l.fuel_surcharge) - SUM(fp.total_cost))
        / NULLIF(SUM(l.revenue + l.fuel_surcharge), 0) * 100,
        1
    )                                                          AS margin_pct
FROM LOADS l
JOIN ROUTES r         ON l.route_id  = r.route_id
JOIN TRIPS  t         ON l.load_id   = t.load_id
LEFT JOIN FUEL_PURCHASES fp ON t.trip_id = fp.trip_id
WHERE l.load_status = 'Completed'
GROUP BY r.route_id, r.origin_city, r.origin_state,
         r.destination_city, r.destination_state, r.typical_distance_miles
HAVING total_loads >= 5
ORDER BY gross_profit DESC
LIMIT 20;

--------------------------------------------------------------------------------
-- Q4: Monthly revenue trend (time-based analysis)
--     Revenue, load count, and avg revenue by month over the full date range.
--------------------------------------------------------------------------------
SELECT
    DATE_TRUNC('MONTH', l.load_date)          AS month,
    COUNT(l.load_id)                          AS loads,
    ROUND(SUM(l.revenue), 2)                  AS monthly_revenue,
    ROUND(AVG(l.revenue), 2)                  AS avg_revenue_per_load,
    ROUND(SUM(l.fuel_surcharge), 2)           AS total_fuel_surcharge,
    ROUND(SUM(l.accessorial_charges), 2)      AS total_accessorials
FROM LOADS l
WHERE l.load_status = 'Completed'
GROUP BY DATE_TRUNC('MONTH', l.load_date)
ORDER BY month;

--------------------------------------------------------------------------------
-- Q5: Truck fleet utilization (filtered + multi-join + aggregation)
--     Miles, trips, fuel cost, and revenue per truck. Filtered to Active trucks
--     with completed trips only.
--------------------------------------------------------------------------------
SELECT
    tk.truck_id,
    tk.make,
    tk.model_year,
    tk.home_terminal,
    COUNT(t.trip_id)                              AS trips_completed,
    ROUND(SUM(t.actual_distance_miles), 0)        AS total_miles,
    ROUND(SUM(t.fuel_gallons_used), 1)            AS total_fuel_gallons,
    ROUND(AVG(t.average_mpg), 2)                  AS avg_mpg,
    ROUND(SUM(fp.total_cost), 2)                  AS total_fuel_cost,
    ROUND(SUM(l.revenue), 2)                      AS total_revenue,
    ROUND(SUM(l.revenue) - SUM(fp.total_cost), 2) AS net_after_fuel
FROM TRUCKS tk
JOIN TRIPS t             ON tk.truck_id = t.truck_id
JOIN LOADS l             ON t.load_id   = l.load_id
LEFT JOIN FUEL_PURCHASES fp ON t.trip_id = fp.trip_id
WHERE tk.status = 'Active'
  AND t.trip_status = 'Completed'
GROUP BY tk.truck_id, tk.make, tk.model_year, tk.home_terminal
ORDER BY total_revenue DESC
LIMIT 30;
