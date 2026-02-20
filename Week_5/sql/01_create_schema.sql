-- 01_create_schema.sql
-- Trucking/logistics data warehouse — 7-table subset (Week 5 scope)
-- Run this in a Snowflake Worksheet first.

CREATE DATABASE IF NOT EXISTS CS5542_WEEK5;
USE DATABASE CS5542_WEEK5;
CREATE SCHEMA IF NOT EXISTS CS5542_WEEK5.PUBLIC;
USE SCHEMA PUBLIC;

--------------------------------------------------------------------------------
-- DIMENSION TABLES
--------------------------------------------------------------------------------

CREATE OR REPLACE TABLE CUSTOMERS (
    customer_id            STRING      NOT NULL,
    customer_name          STRING,
    customer_type          STRING,        -- Dedicated | Brokerage | Contract
    credit_terms_days      INT,
    primary_freight_type   STRING,        -- General | Refrigerated | Hazmat | ...
    account_status         STRING,        -- Active | Inactive
    contract_start_date    DATE,
    annual_revenue_potential FLOAT,
    CONSTRAINT pk_customers PRIMARY KEY (customer_id)
);

CREATE OR REPLACE TABLE DRIVERS (
    driver_id              STRING      NOT NULL,
    first_name             STRING,
    last_name              STRING,
    hire_date              DATE,
    termination_date       DATE,          -- NULL if still employed
    license_number         STRING,
    license_state          STRING,
    date_of_birth          DATE,
    home_terminal          STRING,
    employment_status      STRING,        -- Active | Inactive | Leave
    cdl_class              STRING,        -- A | B
    years_experience       INT,
    CONSTRAINT pk_drivers PRIMARY KEY (driver_id)
);

CREATE OR REPLACE TABLE TRUCKS (
    truck_id               STRING      NOT NULL,
    unit_number            INT,
    make                   STRING,        -- Peterbilt | Kenworth | Freightliner | Volvo
    model_year             INT,
    vin                    STRING,
    acquisition_date       DATE,
    acquisition_mileage    INT,
    fuel_type              STRING,        -- Diesel | CNG | Electric
    tank_capacity_gallons  INT,
    status                 STRING,        -- Active | In Maintenance | Retired
    home_terminal          STRING,
    CONSTRAINT pk_trucks PRIMARY KEY (truck_id)
);

CREATE OR REPLACE TABLE ROUTES (
    route_id               STRING      NOT NULL,
    origin_city            STRING,
    origin_state           STRING,
    destination_city       STRING,
    destination_state      STRING,
    typical_distance_miles FLOAT,
    base_rate_per_mile     FLOAT,
    fuel_surcharge_rate    FLOAT,
    typical_transit_days   INT,
    CONSTRAINT pk_routes PRIMARY KEY (route_id)
);

--------------------------------------------------------------------------------
-- FACT TABLES
--------------------------------------------------------------------------------

CREATE OR REPLACE TABLE LOADS (
    load_id                STRING      NOT NULL,
    customer_id            STRING,        -- FK → CUSTOMERS
    route_id               STRING,        -- FK → ROUTES
    load_date              DATE,
    load_type              STRING,        -- Dry Van | Refrigerated | Flatbed | Tanker
    weight_lbs             FLOAT,
    pieces                 INT,
    revenue                FLOAT,
    fuel_surcharge         FLOAT,
    accessorial_charges    FLOAT,
    load_status            STRING,        -- Completed | In Transit | Cancelled
    booking_type           STRING,        -- Contract | Spot
    CONSTRAINT pk_loads PRIMARY KEY (load_id)
);

CREATE OR REPLACE TABLE TRIPS (
    trip_id                STRING      NOT NULL,
    load_id                STRING,        -- FK → LOADS
    driver_id              STRING,        -- FK → DRIVERS
    truck_id               STRING,        -- FK → TRUCKS
    trailer_id             STRING,        -- FK → TRAILERS (deferred)
    dispatch_date          DATE,
    actual_distance_miles  FLOAT,
    actual_duration_hours  FLOAT,
    fuel_gallons_used      FLOAT,
    average_mpg            FLOAT,
    idle_time_hours        FLOAT,
    trip_status            STRING,        -- Completed | In Progress | Cancelled
    CONSTRAINT pk_trips PRIMARY KEY (trip_id)
);

CREATE OR REPLACE TABLE FUEL_PURCHASES (
    fuel_purchase_id       STRING      NOT NULL,
    trip_id                STRING,        -- FK → TRIPS
    truck_id               STRING,        -- FK → TRUCKS
    driver_id              STRING,        -- FK → DRIVERS
    purchase_date          TIMESTAMP_NTZ,
    location_city          STRING,
    location_state         STRING,
    gallons                FLOAT,
    price_per_gallon       FLOAT,
    total_cost             FLOAT,
    fuel_card_number       STRING,
    CONSTRAINT pk_fuel_purchases PRIMARY KEY (fuel_purchase_id)
);
