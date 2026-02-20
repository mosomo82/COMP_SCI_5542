"""
load_local_csv_to_stage.py

Upload local CSV files to a Snowflake internal stage and COPY INTO tables.

Usage:
  Single file:   python scripts/load_local_csv_to_stage.py data/customers.csv CUSTOMERS
  Batch (all 7): python scripts/load_local_csv_to_stage.py --batch
"""
import os
import sys
import time
from sf_connect import get_conn

# ---------- configuration ----------

STAGE_NAME = "CS5542_STAGE"
FILE_FORMAT = "CS5542_CSV_FMT"

# Ordered list: dimensions first, then facts (respects FK dependencies)
BATCH_MANIFEST = [
    ("data/customers.csv",       "CUSTOMERS"),
    ("data/drivers.csv",         "DRIVERS"),
    ("data/trucks.csv",          "TRUCKS"),
    ("data/routes.csv",          "ROUTES"),
    ("data/loads.csv",           "LOADS"),
    ("data/trips.csv",           "TRIPS"),
    ("data/fuel_purchases.csv",  "FUEL_PURCHASES"),
]

# ---------- helpers ----------

def run_sql(sql: str):
    """Execute a single SQL statement and return results."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            try:
                return cur.fetchall()
            except Exception:
                return None


def ensure_stage():
    """Create file format and stage if they don't exist (idempotent)."""
    run_sql(f"""
    CREATE OR REPLACE FILE FORMAT {FILE_FORMAT}
      TYPE = CSV
      SKIP_HEADER = 1
      FIELD_OPTIONALLY_ENCLOSED_BY = '"'
      NULL_IF = ('', 'NULL', 'null');
    """)
    run_sql(f"CREATE OR REPLACE STAGE {STAGE_NAME} FILE_FORMAT = {FILE_FORMAT};")


def upload_and_load(local_path: str, target_table: str):
    """PUT a local CSV to the stage, then COPY INTO the target table."""
    abs_path = os.path.abspath(local_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"CSV not found: {abs_path}")

    filename = os.path.basename(local_path)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # PUT (upload + auto-compress)
            put_sql = (
                f"PUT file://{abs_path} @{STAGE_NAME} "
                f"AUTO_COMPRESS=TRUE OVERWRITE=TRUE;"
            )
            print(f"  PUT  {filename} → @{STAGE_NAME} ... ", end="", flush=True)
            cur.execute(put_sql)
            print("done")

            # COPY INTO
            copy_sql = f"""
            COPY INTO {target_table}
            FROM @{STAGE_NAME}/{filename}.gz
            ON_ERROR = 'CONTINUE';
            """
            t0 = time.time()
            cur.execute(copy_sql)
            res = cur.fetchall()
            dt_ms = int((time.time() - t0) * 1000)

            # Parse result — first row typically has (file, status, rows_parsed,
            # rows_loaded, errors_seen, ...)
            if res and len(res[0]) >= 4:
                rows_loaded = res[0][3]
            else:
                rows_loaded = "?"
            print(f"  COPY {target_table}: {rows_loaded} rows loaded ({dt_ms} ms)")

# ---------- main ----------

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--batch":
        print("=== Batch load: 7 trucking CSVs ===\n")
        ensure_stage()
        total_t0 = time.time()
        for csv_path, table in BATCH_MANIFEST:
            print(f"[{table}]")
            upload_and_load(csv_path, table)
            print()
        total_s = round(time.time() - total_t0, 1)
        print(f"=== Batch complete in {total_s}s ===")

    elif len(sys.argv) == 3:
        local_path = sys.argv[1]
        target_table = sys.argv[2].upper()
        ensure_stage()
        upload_and_load(local_path, target_table)

    else:
        print("Usage:")
        print("  Single:  python scripts/load_local_csv_to_stage.py <csv_path> <TABLE>")
        print("  Batch:   python scripts/load_local_csv_to_stage.py --batch")
        sys.exit(1)


if __name__ == "__main__":
    main()
