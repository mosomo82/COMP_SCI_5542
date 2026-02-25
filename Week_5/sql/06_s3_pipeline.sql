-- 06_s3_pipeline.sql
-- Automated S3 → Snowflake ingestion pipeline setup.
-- Run once (as ACCOUNTADMIN) after loading your CSVs to S3.
--
-- S3 bucket : s3://my-snowflake-pipeline-data-lab5--use2-az1--x-s3/data/
-- IAM Role  : arn:aws:iam::507041536990:role/snowflake_access_role

USE ROLE ACCOUNTADMIN;
USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;
USE WAREHOUSE COMPUTE_WH;

--------------------------------------------------------------------------------
-- STEP 1: Storage Integration (run once; requires ACCOUNTADMIN)
-- After creating, run DESCRIBE INTEGRATION s3_integration to get the
-- STORAGE_AWS_IAM_USER_ARN and STORAGE_AWS_EXTERNAL_ID, then update your
-- IAM trust policy in AWS Console.
--------------------------------------------------------------------------------
CREATE OR REPLACE STORAGE INTEGRATION s3_integration
  TYPE                      = EXTERNAL_STAGE
  STORAGE_PROVIDER          = 'S3'
  ENABLED                   = TRUE
  STORAGE_AWS_ROLE_ARN      = 'arn:aws:iam::507041536990:role/snowflake_access_role'
  STORAGE_ALLOWED_LOCATIONS = ('s3://my-snowflake-pipeline-data-lab5--use2-az1--x-s3/data/');

-- Show the Snowflake IAM principal you must trust in your AWS role:
DESCRIBE INTEGRATION s3_integration;

--------------------------------------------------------------------------------
-- STEP 2: Grant USAGE on the integration to SYSADMIN (optional best practice)
--------------------------------------------------------------------------------
GRANT USAGE ON INTEGRATION s3_integration TO ROLE SYSADMIN;

USE ROLE SYSADMIN;
USE DATABASE CS5542_WEEK5;
USE SCHEMA PUBLIC;

--------------------------------------------------------------------------------
-- STEP 3: S3 external file format
--------------------------------------------------------------------------------
CREATE OR REPLACE FILE FORMAT CS5542_S3_CSV_FMT
  TYPE                        = CSV
  SKIP_HEADER                 = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF                     = ('', 'NULL', 'null')
  EMPTY_FIELD_AS_NULL         = TRUE
  DATE_FORMAT                 = 'AUTO'
  TIMESTAMP_FORMAT            = 'AUTO';

--------------------------------------------------------------------------------
-- STEP 4: External stage pointing at S3 bucket /data/ prefix
--------------------------------------------------------------------------------
CREATE OR REPLACE STAGE CS5542_S3_STAGE
  URL             = 's3://my-snowflake-pipeline-data-lab5--use2-az1--x-s3/data/'
  STORAGE_INTEGRATION = s3_integration
  FILE_FORMAT     = CS5542_S3_CSV_FMT
  COMMENT         = 'CS5542 Week 5 — automated S3 ingestion stage';

-- Verify files are visible:
LIST @CS5542_S3_STAGE;

--------------------------------------------------------------------------------
-- STEP 5: COPY INTO — all 14 tables from S3 (idempotent with PURGE=FALSE)
-- Run in dependency order: dimensions first, then facts, then extension tables.
--------------------------------------------------------------------------------

-- Dimension tables ─────────────────────────────────────────────────────────
COPY INTO CUSTOMERS
  FROM @CS5542_S3_STAGE/customers.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO DRIVERS
  FROM @CS5542_S3_STAGE/drivers.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO TRUCKS
  FROM @CS5542_S3_STAGE/trucks.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO ROUTES
  FROM @CS5542_S3_STAGE/routes.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

-- Fact tables ──────────────────────────────────────────────────────────────
COPY INTO LOADS
  FROM @CS5542_S3_STAGE/loads.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO TRIPS
  FROM @CS5542_S3_STAGE/trips.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO FUEL_PURCHASES
  FROM @CS5542_S3_STAGE/fuel_purchases.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

-- Extension tables ─────────────────────────────────────────────────────────
COPY INTO TRAILERS
  FROM @CS5542_S3_STAGE/trailers.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO FACILITIES
  FROM @CS5542_S3_STAGE/facilities.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO DELIVERY_EVENTS
  FROM @CS5542_S3_STAGE/delivery_events.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO MAINTENANCE_RECORDS
  FROM @CS5542_S3_STAGE/maintenance_records.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO SAFETY_INCIDENTS
  FROM @CS5542_S3_STAGE/safety_incidents.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO DRIVER_MONTHLY_METRICS
  FROM @CS5542_S3_STAGE/driver_monthly_metrics.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

COPY INTO TRUCK_UTILIZATION_METRICS
  FROM @CS5542_S3_STAGE/truck_utilization_metrics.csv
  FILE_FORMAT = (FORMAT_NAME = CS5542_S3_CSV_FMT)
  ON_ERROR = 'CONTINUE';

--------------------------------------------------------------------------------
-- STEP 6: Verify row counts after ingestion
--------------------------------------------------------------------------------
SELECT 'CUSTOMERS'              AS tbl, COUNT(*) AS rows FROM CUSTOMERS
UNION ALL SELECT 'DRIVERS',              COUNT(*) FROM DRIVERS
UNION ALL SELECT 'TRUCKS',               COUNT(*) FROM TRUCKS
UNION ALL SELECT 'ROUTES',               COUNT(*) FROM ROUTES
UNION ALL SELECT 'LOADS',                COUNT(*) FROM LOADS
UNION ALL SELECT 'TRIPS',                COUNT(*) FROM TRIPS
UNION ALL SELECT 'FUEL_PURCHASES',       COUNT(*) FROM FUEL_PURCHASES
UNION ALL SELECT 'TRAILERS',             COUNT(*) FROM TRAILERS
UNION ALL SELECT 'FACILITIES',           COUNT(*) FROM FACILITIES
UNION ALL SELECT 'DELIVERY_EVENTS',      COUNT(*) FROM DELIVERY_EVENTS
UNION ALL SELECT 'MAINTENANCE_RECORDS',  COUNT(*) FROM MAINTENANCE_RECORDS
UNION ALL SELECT 'SAFETY_INCIDENTS',     COUNT(*) FROM SAFETY_INCIDENTS
UNION ALL SELECT 'DRIVER_MONTHLY_METRICS',   COUNT(*) FROM DRIVER_MONTHLY_METRICS
UNION ALL SELECT 'TRUCK_UTILIZATION_METRICS', COUNT(*) FROM TRUCK_UTILIZATION_METRICS
ORDER BY tbl;