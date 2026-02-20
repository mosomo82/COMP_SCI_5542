import os
import sys
import time
from sf_connect import get_conn

def run(sql: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            try:
                return cur.fetchall()
            except Exception:
                return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/load_local_csv_to_stage.py <local_csv_path> <target_table>")
        print("Example: python scripts/load_local_csv_to_stage.py data/events.csv EVENTS")
        sys.exit(1)

    local_path = sys.argv[1]
    target_table = sys.argv[2].upper()

    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)

    stage_name = "CS5542_STAGE"
    file_format = "CS5542_CSV_FMT"

    # Create stage + file format (idempotent)
    run(f"""
    CREATE OR REPLACE FILE FORMAT {file_format}
      TYPE = CSV
      SKIP_HEADER = 1
      FIELD_OPTIONALLY_ENCLOSED_BY = '"'
      NULL_IF = ('', 'NULL', 'null');
    """)

    run(f"CREATE OR REPLACE STAGE {stage_name} FILE_FORMAT = {file_format};")

    # Upload
    with get_conn() as conn:
        with conn.cursor() as cur:
            put_sql = f"PUT file://{os.path.abspath(local_path)} @{stage_name} AUTO_COMPRESS=TRUE OVERWRITE=TRUE;"
            print(put_sql)
            cur.execute(put_sql)
            print(cur.fetchall())

            # COPY INTO
            filename = os.path.basename(local_path)
            copy_sql = f"""
            COPY INTO {target_table}
            FROM @{stage_name}/{filename}.gz
            ON_ERROR='CONTINUE';
            """
            t0 = time.time()
            cur.execute(copy_sql)
            res = cur.fetchall()
            dt_ms = int((time.time() - t0) * 1000)

            print("COPY result:", res)
            print(f"Load latency: {dt_ms} ms")

if __name__ == "__main__":
    main()
