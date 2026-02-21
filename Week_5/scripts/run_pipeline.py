"""
run_pipeline.py — CS 5542 Week 5 Automated Data Ingestion Pipeline
===================================================================
Orchestrates the full end-to-end pipeline without any manual
Snowflake Worksheet steps:

  1.  Create / replace all 14 table DDLs  (01_create_schema.sql)
  2.  Create warehouse, internal file format & stage
  3.  Setup S3 external stage (06_s3_pipeline.sql steps 2–4)
  4.  COPY INTO all 14 tables from  → S3 external stage (preferred)
                                    OR internal stage (PUT + COPY fallback)
  5.  Create / replace 5 analytical views  (04_views.sql)
  6.  Create / replace 4 derived analytics tables  (05_derived_analytics.sql)
  7.  Print a row-count summary
  8.  Append a pipeline run record to  logs/pipeline_logs.csv

Usage (from project root):
  py scripts/run_pipeline.py                   # S3 mode (default)
  py scripts/run_pipeline.py --local           # internal stage / local CSV mode
  py scripts/run_pipeline.py --skip-s3-setup   # skip storage integration creation
"""

import os
import sys
import time
import pathlib
import argparse
import csv
from datetime import datetime, timezone

# ── path setup ────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parents[1]
SQL    = ROOT / "sql"
DATA   = ROOT / "data"
LOGS   = ROOT / "logs"
sys.path.insert(0, str(ROOT / "scripts"))
from sf_connect import get_conn                           # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
S3_BUCKET      = "s3://my-snowflake-pipeline-data-lab5--use2-az1--x-s3/data/"
S3_INTEGRATION = "s3_integration"
S3_STAGE       = "CS5542_S3_STAGE"
S3_FMT         = "CS5542_S3_CSV_FMT"
INT_STAGE      = "CS5542_STAGE"
INT_FMT        = "CS5542_CSV_FMT"
LOG_FILE       = LOGS / "pipeline_logs.csv"

# Table load order: dimensions → facts → extensions
TABLES = [
    ("customers.csv",                "CUSTOMERS"),
    ("drivers.csv",                  "DRIVERS"),
    ("trucks.csv",                   "TRUCKS"),
    ("routes.csv",                   "ROUTES"),
    ("loads.csv",                    "LOADS"),
    ("trips.csv",                    "TRIPS"),
    ("fuel_purchases.csv",           "FUEL_PURCHASES"),
    ("trailers.csv",                 "TRAILERS"),
    ("facilities.csv",               "FACILITIES"),
    ("delivery_events.csv",          "DELIVERY_EVENTS"),
    ("maintenance_records.csv",      "MAINTENANCE_RECORDS"),
    ("safety_incidents.csv",         "SAFETY_INCIDENTS"),
    ("driver_monthly_metrics.csv",   "DRIVER_MONTHLY_METRICS"),
    ("truck_utilization_metrics.csv","TRUCK_UTILIZATION_METRICS"),
]

SQL_FILES_POST_LOAD = [
    SQL / "04_views.sql",
    SQL / "05_derived_analytics.sql",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _banner(msg: str):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def _ok(msg: str):
    print(f"  ✓  {msg}")

def _warn(msg: str):
    print(f"  ⚠  {msg}")

def _split_sql(text: str) -> list[str]:
    """Split a .sql file into individual executable statements."""
    stmts = []
    for raw in text.split(";"):
        stmt = raw.strip()
        lines = [l for l in stmt.splitlines()
                 if l.strip() and not l.strip().startswith("--")]
        if lines:
            stmts.append(stmt)
    return stmts

def run_sql_file(conn, path: pathlib.Path, label: str) -> int:
    """Execute every statement in *path*. Returns number of statements run."""
    if not path.exists():
        _warn(f"{label}: file not found ({path}), skipping.")
        return 0
    stmts = _split_sql(path.read_text(encoding="utf-8"))
    ran = 0
    with conn.cursor() as cur:
        for stmt in stmts:
            try:
                cur.execute(stmt)
                ran += 1
            except Exception:
                pass   # USE DATABASE / idempotent DDL failures are non-fatal
    _ok(f"{label} — {ran} statements executed")
    return ran

def run_one(conn, sql: str) -> list:
    """Execute one SQL statement and return results."""
    with conn.cursor() as cur:
        cur.execute(sql)
        try:
            return cur.fetchall()
        except Exception:
            return []

# ── pipeline steps ────────────────────────────────────────────────────────────

def step_schema(conn):
    _banner("STEP 1 — Ensure schema & all 14 tables")
    run_sql_file(conn, SQL / "01_create_schema.sql", "Schema DDL")

def step_internal_stage(conn):
    _banner("STEP 2 — Internal stage & file format")
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE OR REPLACE FILE FORMAT {INT_FMT}
              TYPE = CSV SKIP_HEADER = 1
              FIELD_OPTIONALLY_ENCLOSED_BY = '"'
              NULL_IF = ('', 'NULL', 'null');
        """)
        cur.execute(f"CREATE OR REPLACE STAGE {INT_STAGE} FILE_FORMAT = {INT_FMT};")
    _ok(f"Stage @{INT_STAGE} ready")

def step_s3_stage(conn, skip_integration: bool):
    _banner("STEP 2 — S3 external stage & file format")
    with conn.cursor() as cur:
        if not skip_integration:
            try:
                cur.execute(f"""
                    CREATE OR REPLACE STORAGE INTEGRATION {S3_INTEGRATION}
                      TYPE = EXTERNAL_STAGE
                      STORAGE_PROVIDER = 'S3'
                      ENABLED = TRUE
                      STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::507041536990:role/snowflake_access_role'
                      STORAGE_ALLOWED_LOCATIONS = ('{S3_BUCKET}');
                """)
                _ok(f"Storage integration {S3_INTEGRATION} created / replaced")
            except Exception as exc:
                _warn(f"Could not create storage integration (need ACCOUNTADMIN?): {exc}")
                _warn("Continuing — assuming integration already exists.")

        cur.execute(f"""
            CREATE OR REPLACE FILE FORMAT {S3_FMT}
              TYPE = CSV SKIP_HEADER = 1
              FIELD_OPTIONALLY_ENCLOSED_BY = '"'
              NULL_IF = ('', 'NULL', 'null')
              EMPTY_FIELD_AS_NULL = TRUE
              DATE_FORMAT = 'AUTO' TIMESTAMP_FORMAT = 'AUTO';
        """)
        cur.execute(f"""
            CREATE OR REPLACE STAGE {S3_STAGE}
              URL = '{S3_BUCKET}'
              STORAGE_INTEGRATION = {S3_INTEGRATION}
              FILE_FORMAT = {S3_FMT};
        """)
    _ok(f"S3 stage @{S3_STAGE} pointing at {S3_BUCKET}")

def step_load_from_s3(conn) -> dict[str, int]:
    """COPY INTO each table from the S3 external stage."""
    _banner("STEP 3 — COPY INTO from S3")
    counts: dict[str, int] = {}
    for csv_file, table in TABLES:
        t0 = time.time()
        sql = f"""
            COPY INTO {table}
            FROM @{S3_STAGE}/{csv_file}
            FILE_FORMAT = (FORMAT_NAME = {S3_FMT})
            ON_ERROR = 'CONTINUE';
        """
        try:
            res = run_one(conn, sql)
            rows = res[0][3] if res and len(res[0]) >= 4 else "?"
            ms   = int((time.time() - t0) * 1000)
            counts[table] = int(rows) if str(rows).isdigit() else 0
            _ok(f"{table:<32} {rows} rows  ({ms} ms)")
        except Exception as exc:
            _warn(f"{table}: {exc}")
            counts[table] = -1
    return counts

def step_load_from_local(conn) -> dict[str, int]:
    """PUT local CSV → internal stage, then COPY INTO each table."""
    _banner("STEP 3 — PUT + COPY INTO (internal stage / local mode)")
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for csv_file, table in TABLES:
            local_path = DATA / csv_file
            abs_path   = local_path.resolve()
            if not abs_path.exists():
                _warn(f"{csv_file} not found — skipping {table}")
                counts[table] = -1
                continue

            # PUT
            print(f"  PUT  {csv_file} ... ", end="", flush=True)
            cur.execute(
                f"PUT file://{abs_path} @{INT_STAGE} AUTO_COMPRESS=TRUE OVERWRITE=TRUE;"
            )
            print("done")

            # COPY INTO
            t0  = time.time()
            cur.execute(f"""
                COPY INTO {table}
                FROM @{INT_STAGE}/{csv_file}.gz
                ON_ERROR = 'CONTINUE';
            """)
            res  = cur.fetchall()
            ms   = int((time.time() - t0) * 1000)
            rows = res[0][3] if res and len(res[0]) >= 4 else "?"
            counts[table] = int(rows) if str(rows).isdigit() else 0
            _ok(f"{table:<32} {rows} rows  ({ms} ms)")
    return counts

def step_post_load_sql(conn):
    _banner("STEP 4 — Views & derived analytics tables")
    for sql_path in SQL_FILES_POST_LOAD:
        run_sql_file(conn, sql_path, sql_path.name)

def step_row_counts(conn) -> dict[str, int]:
    _banner("STEP 5 — Row-count verification")
    counts: dict[str, int] = {}
    union = "\nUNION ALL ".join(
        f"SELECT '{t}' AS tbl, COUNT(*) AS rows FROM {t}"
        for _, t in TABLES
    )
    try:
        rows = run_one(conn, union + " ORDER BY tbl;")
        for table, count in rows:
            counts[table] = count
            _ok(f"{table:<32} {count:>10,} rows")
    except Exception as exc:
        _warn(f"Row count query failed: {exc}")
    return counts

def write_log(mode: str, duration_s: float, load_counts: dict, row_counts: dict):
    LOGS.mkdir(exist_ok=True)
    total_rows   = sum(v for v in row_counts.values() if v >= 0)
    tables_ok    = sum(1 for v in load_counts.values() if v >= 0)
    perf_note    = (
        "Fast (<60 s)" if duration_s < 60 else
        "Moderate (1–3 min)" if duration_s < 180 else
        "Slow (>3 min) — consider larger warehouse"
    )
    row = {
        "timestamp":      _ts(),
        "query_name":     f"run_pipeline ({mode})",
        "latency_ms":     int(duration_s * 1000),
        "rows_returned":  total_rows,
        "error":          "" if all(v >= 0 for v in load_counts.values()) else
                          f"{sum(1 for v in load_counts.values() if v < 0)} table(s) failed",
        "perf_note":      perf_note,
    }
    header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header:
            writer.writeheader()
        writer.writerow(row)
    _ok(f"Pipeline run logged → {LOG_FILE.relative_to(ROOT)}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="CS5542 automated ingestion pipeline")
    ap.add_argument("--local",         action="store_true",
                    help="Use internal stage (PUT local CSVs) instead of S3")
    ap.add_argument("--skip-s3-setup", action="store_true",
                    help="Skip storage integration creation (assume it already exists)")
    args = ap.parse_args()

    mode = "local" if args.local else "s3"
    print(f"\n{'═'*60}")
    print(f"  CS 5542 — Automated Ingestion Pipeline")
    print(f"  Mode : {mode.upper()}   |   {_ts()}")
    print(f"{'═'*60}")

    pipeline_start = time.time()

    with get_conn() as conn:
        step_schema(conn)

        if args.local:
            step_internal_stage(conn)
            load_counts = step_load_from_local(conn)
        else:
            step_s3_stage(conn, skip_integration=args.skip_s3_setup)
            load_counts = step_load_from_s3(conn)

        step_post_load_sql(conn)
        row_counts = step_row_counts(conn)

    duration = round(time.time() - pipeline_start, 1)
    write_log(mode, duration, load_counts, row_counts)

    _banner(f"Pipeline complete in {duration}s")
    failed = [t for t, v in load_counts.items() if v < 0]
    if failed:
        print(f"  ⚠  Failed tables: {failed}")
        sys.exit(1)
    else:
        print("  ✓  All 14 tables loaded successfully.")


if __name__ == "__main__":
    main()
