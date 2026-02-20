-- 02_stage_and_load.sql
-- Warehouse, file format, stage, and COPY INTO for all 7 trucking tables.
-- Run this in a Snowflake Worksheet AFTER 01_create_schema.sql.
-- CSVs must already be uploaded to @CS5542_STAGE (via PUT or the Python loader).

-- Warehouse (idempotent)
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

USE WAREHOUSE COMPUTE_WH;
USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;

-- CSV file format
CREATE OR REPLACE FILE FORMAT CS5542_CSV_FMT
  TYPE = CSV
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF = ('', 'NULL', 'null');

-- Internal stage
CREATE OR REPLACE STAGE CS5542_STAGE
  FILE_FORMAT = CS5542_CSV_FMT;

--------------------------------------------------------------------------------
-- COPY INTO — dimensions first, then facts
--------------------------------------------------------------------------------

-- 1) Customers (200 rows)
COPY INTO CUSTOMERS
FROM @CS5542_STAGE/customers.csv.gz
ON_ERROR = 'CONTINUE';

-- 2) Drivers (150 rows)
COPY INTO DRIVERS
FROM @CS5542_STAGE/drivers.csv.gz
ON_ERROR = 'CONTINUE';

-- 3) Trucks (120 rows)
COPY INTO TRUCKS
FROM @CS5542_STAGE/trucks.csv.gz
ON_ERROR = 'CONTINUE';

-- 4) Routes (58 rows)
COPY INTO ROUTES
FROM @CS5542_STAGE/routes.csv.gz
ON_ERROR = 'CONTINUE';

-- 5) Loads (85K rows)
COPY INTO LOADS
FROM @CS5542_STAGE/loads.csv.gz
ON_ERROR = 'CONTINUE';

-- 6) Trips (85K rows)
COPY INTO TRIPS
FROM @CS5542_STAGE/trips.csv.gz
ON_ERROR = 'CONTINUE';

-- 7) Fuel purchases (196K rows)
COPY INTO FUEL_PURCHASES
FROM @CS5542_STAGE/fuel_purchases.csv.gz
ON_ERROR = 'CONTINUE';
