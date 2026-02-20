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

--------------------------------------------------------------------------------
-- EXTENSION TABLES (additional dataset ingestion)
--------------------------------------------------------------------------------

CREATE OR REPLACE TABLE TRAILERS (
    trailer_id             STRING      NOT NULL,
    trailer_number         INT,
    trailer_type           STRING,        -- Dry Van | Refrigerated | Flatbed | Tanker
    length_feet            INT,
    model_year             INT,
    vin                    STRING,
    acquisition_date       DATE,
    status                 STRING,        -- Active | In Maintenance | Retired
    current_location       STRING,
    CONSTRAINT pk_trailers PRIMARY KEY (trailer_id)
);

CREATE OR REPLACE TABLE FACILITIES (
    facility_id            STRING      NOT NULL,
    facility_name          STRING,
    facility_type          STRING,        -- Cross-Dock | Warehouse | Terminal | ...
    city                   STRING,
    state                  STRING,
    latitude               FLOAT,
    longitude              FLOAT,
    dock_doors             INT,
    operating_hours        STRING,
    CONSTRAINT pk_facilities PRIMARY KEY (facility_id)
);

CREATE OR REPLACE TABLE DELIVERY_EVENTS (
    event_id               STRING      NOT NULL,
    load_id                STRING,        -- FK → LOADS
    trip_id                STRING,        -- FK → TRIPS
    event_type             STRING,        -- Pickup | Delivery
    facility_id            STRING,        -- FK → FACILITIES
    scheduled_datetime     TIMESTAMP_NTZ,
    actual_datetime        TIMESTAMP_NTZ,
    detention_minutes      INT,
    on_time_flag           BOOLEAN,
    location_city          STRING,
    location_state         STRING,
    CONSTRAINT pk_delivery_events PRIMARY KEY (event_id)
);

CREATE OR REPLACE TABLE MAINTENANCE_RECORDS (
    maintenance_id         STRING      NOT NULL,
    truck_id               STRING,        -- FK → TRUCKS
    maintenance_date       DATE,
    maintenance_type       STRING,        -- Scheduled | Unscheduled | Inspection
    odometer_reading       INT,
    labor_hours            FLOAT,
    labor_cost             FLOAT,
    parts_cost             FLOAT,
    total_cost             FLOAT,
    facility_location      STRING,
    downtime_hours         FLOAT,
    service_description    STRING,
    CONSTRAINT pk_maintenance_records PRIMARY KEY (maintenance_id)
);

CREATE OR REPLACE TABLE SAFETY_INCIDENTS (
    incident_id            STRING      NOT NULL,
    trip_id                STRING,        -- FK → TRIPS
    truck_id               STRING,        -- FK → TRUCKS
    driver_id              STRING,        -- FK → DRIVERS
    incident_date          TIMESTAMP_NTZ,
    incident_type          STRING,        -- Moving Violation | Collision | Equipment Failure | ...
    location_city          STRING,
    location_state         STRING,
    at_fault_flag          BOOLEAN,
    injury_flag            BOOLEAN,
    vehicle_damage_cost    FLOAT,
    cargo_damage_cost      FLOAT,
    claim_amount           FLOAT,
    preventable_flag       BOOLEAN,
    description            STRING,
    CONSTRAINT pk_safety_incidents PRIMARY KEY (incident_id)
);

CREATE OR REPLACE TABLE DRIVER_MONTHLY_METRICS (
    driver_id              STRING,        -- FK → DRIVERS
    month                  DATE,
    trips_completed        INT,
    total_miles            FLOAT,
    total_revenue          FLOAT,
    average_mpg            FLOAT,
    total_fuel_gallons     FLOAT,
    on_time_delivery_rate  FLOAT,
    average_idle_hours     FLOAT
);

CREATE OR REPLACE TABLE TRUCK_UTILIZATION_METRICS (
    truck_id               STRING,        -- FK → TRUCKS
    month                  DATE,
    trips_completed        INT,
    total_miles            FLOAT,
    total_revenue          FLOAT,
    average_mpg            FLOAT,
    maintenance_events     INT,
    maintenance_cost       FLOAT,
    downtime_hours         FLOAT,
    utilization_rate       FLOAT
);
