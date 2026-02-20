-- 04_views.sql
-- Derived views for the Streamlit dashboard.
-- Run after data is loaded (after 02_stage_and_load.sql).

USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;

--------------------------------------------------------------------------------
-- V_LOAD_DETAILS: Denormalized load view (loads + customer + route info)
--   Flattens the most common dashboard join so the app hits one view.
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW V_LOAD_DETAILS AS
SELECT
    l.load_id,
    l.load_date,
    l.load_type,
    l.weight_lbs,
    l.pieces,
    l.revenue,
    l.fuel_surcharge,
    l.accessorial_charges,
    l.revenue + l.fuel_surcharge + l.accessorial_charges  AS total_charge,
    l.load_status,
    l.booking_type,
    -- customer
    c.customer_id,
    c.customer_name,
    c.customer_type,
    c.primary_freight_type,
    -- route
    r.route_id,
    r.origin_city,
    r.origin_state,
    r.destination_city,
    r.destination_state,
    r.typical_distance_miles,
    r.origin_city || ', ' || r.origin_state
        || ' → '
        || r.destination_city || ', ' || r.destination_state  AS route_label
FROM LOADS l
LEFT JOIN CUSTOMERS c ON l.customer_id = c.customer_id
LEFT JOIN ROUTES    r ON l.route_id    = r.route_id;

--------------------------------------------------------------------------------
-- V_TRIP_PERFORMANCE: Trip + driver + truck view for operational dashboards
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW V_TRIP_PERFORMANCE AS
SELECT
    t.trip_id,
    t.dispatch_date,
    t.actual_distance_miles,
    t.actual_duration_hours,
    t.fuel_gallons_used,
    t.average_mpg,
    t.idle_time_hours,
    t.trip_status,
    -- driver
    d.driver_id,
    d.first_name || ' ' || d.last_name  AS driver_name,
    d.home_terminal                      AS driver_terminal,
    d.cdl_class,
    d.years_experience,
    -- truck
    tk.truck_id,
    tk.make                              AS truck_make,
    tk.model_year                        AS truck_year,
    tk.fuel_type,
    -- load
    l.load_id,
    l.revenue,
    l.load_type
FROM TRIPS t
LEFT JOIN DRIVERS   d  ON t.driver_id = d.driver_id
LEFT JOIN TRUCKS    tk ON t.truck_id  = tk.truck_id
LEFT JOIN LOADS     l  ON t.load_id   = l.load_id;

--------------------------------------------------------------------------------
-- V_MONTHLY_REVENUE: Time-series aggregation for trend charts
--   One row per month with key financial KPIs.
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW V_MONTHLY_REVENUE AS
SELECT
    DATE_TRUNC('MONTH', l.load_date)                         AS month,
    COUNT(l.load_id)                                         AS total_loads,
    COUNT(DISTINCT l.customer_id)                            AS unique_customers,
    ROUND(SUM(l.revenue), 2)                                 AS revenue,
    ROUND(SUM(l.fuel_surcharge), 2)                          AS fuel_surcharges,
    ROUND(SUM(l.accessorial_charges), 2)                     AS accessorials,
    ROUND(SUM(l.revenue + l.fuel_surcharge + l.accessorial_charges), 2) AS total_charges,
    ROUND(AVG(l.revenue), 2)                                 AS avg_revenue_per_load,
    ROUND(AVG(l.weight_lbs), 0)                              AS avg_weight_lbs
FROM LOADS l
WHERE l.load_status = 'Completed'
GROUP BY DATE_TRUNC('MONTH', l.load_date)
ORDER BY month;

--------------------------------------------------------------------------------
-- V_ROUTE_SCORECARD: Route-level KPIs (profitability + volume)
--   Joins loads → routes → trips → fuel to compute per-route metrics.
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW V_ROUTE_SCORECARD AS
SELECT
    r.route_id,
    r.origin_city || ', ' || r.origin_state
        || ' → '
        || r.destination_city || ', ' || r.destination_state  AS route_label,
    r.typical_distance_miles,
    r.base_rate_per_mile,
    COUNT(DISTINCT l.load_id)                                 AS total_loads,
    ROUND(SUM(l.revenue + l.fuel_surcharge), 2)               AS total_revenue,
    ROUND(AVG(l.revenue), 2)                                  AS avg_load_revenue,
    ROUND(SUM(fp.total_cost), 2)                              AS total_fuel_cost,
    ROUND(
        SUM(l.revenue + l.fuel_surcharge) - SUM(fp.total_cost), 2
    )                                                         AS gross_profit,
    ROUND(
        (SUM(l.revenue + l.fuel_surcharge) - SUM(fp.total_cost))
        / NULLIF(SUM(l.revenue + l.fuel_surcharge), 0) * 100,
        1
    )                                                         AS margin_pct,
    ROUND(AVG(t.average_mpg), 2)                              AS avg_mpg
FROM ROUTES r
JOIN LOADS l              ON r.route_id = l.route_id
JOIN TRIPS t              ON l.load_id  = t.load_id
LEFT JOIN FUEL_PURCHASES fp ON t.trip_id = fp.trip_id
WHERE l.load_status = 'Completed'
GROUP BY r.route_id, r.origin_city, r.origin_state,
         r.destination_city, r.destination_state,
         r.typical_distance_miles, r.base_rate_per_mile;

--------------------------------------------------------------------------------
-- V_FUEL_SPEND: Fuel purchase aggregation by state for geo analysis
--------------------------------------------------------------------------------
CREATE OR REPLACE VIEW V_FUEL_SPEND AS
SELECT
    fp.location_state,
    fp.location_city,
    COUNT(*)                              AS purchases,
    ROUND(SUM(fp.gallons), 1)             AS total_gallons,
    ROUND(AVG(fp.price_per_gallon), 3)    AS avg_price_per_gallon,
    ROUND(SUM(fp.total_cost), 2)          AS total_spend,
    MIN(fp.purchase_date)                 AS first_purchase,
    MAX(fp.purchase_date)                 AS last_purchase
FROM FUEL_PURCHASES fp
GROUP BY fp.location_state, fp.location_city
ORDER BY total_spend DESC;
