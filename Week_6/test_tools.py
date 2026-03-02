import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tools

def run_tests():
    print("Testing get_monthly_revenue...")
    try:
        res = tools.get_monthly_revenue('2023-01-01', '2023-02-01')
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_fleet_performance...")
    try:
        res = tools.get_fleet_performance(top_n=2)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_pipeline_logs...")
    try:
        res = tools.get_pipeline_logs(limit=2)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_safety_metrics...")
    try:
        res = tools.get_safety_metrics(limit=2)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_tests()
