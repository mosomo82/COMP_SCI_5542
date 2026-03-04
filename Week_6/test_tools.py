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

    print("\n-----------------------\nTesting get_route_profitability...")
    try:
        res = tools.get_route_profitability(min_loads=3, top_n=2)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_delivery_performance...")
    try:
        res = tools.get_delivery_performance(event_type="Delivery", limit=2)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_maintenance_health...")
    try:
        res = tools.get_maintenance_health(maintenance_type="None", start_date="2023-01-01", end_date="2023-02-01", top_n=20)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-----------------------\nTesting get_fuel_spend_analysis...")
    try:
        res = tools.get_fuel_spend_analysis(states=["CA", "TX"], top_n=20)
        print(f"Result count: {len(res)}")
        if len(res) > 0:
            print(f"First item: {res[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_tests()
