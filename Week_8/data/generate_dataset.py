import json
import random

# Domain knowledge for generating synthetic examples
CITIES = ["Chicago", "Kansas City", "Omaha", "Denver", "Salt Lake City", "Dallas", "Houston", "Atlanta", "Charlotte"]
ROUTES = ["I-70", "I-80", "I-55", "I-10", "I-25", "I-35"]

# 5 Disruption Types
WEATHER_ALERTS = ["Heavy Snowfall (>10cm)", "Severe Icing", "Flash Flooding", "Low Visibility (<1mi)", "High Winds"]
ACCIDENT_SEVERITY = ["Major accident (Severity 4)", "Pileup (Severity 3)"]
TRAFFIC_ALERTS = ["Multi-mile construction backup", "Major event traffic lock", "Bridge closure detours"]
LOGISTICS_ALERTS = ["Driver HOS limit reached", "Mechanical failure (Engine)", "Tire blowout"]
FACILITY_ALERTS = ["Port/Dock strike", "Warehouse power outage", "Receiving facility equipment failure"]

LOAD_TYPES = ["Heavy Haul (80,000 lbs)", "Standard LTL", "Hazmat"]
VEHICLE_HEIGHTS = ["13ft 6in", "14ft", "Permitted Oversize (14ft 6in)"]
BRIDGE_VIOLATIONS = ["Bridge #4432 (Limit: 13ft 0in)", "Bridge #9981 (Limit: 40 Tons)", "Bridge #2210 (Structural Deficit)"]
SAFE_BRIDGES = ["Bridge #1120 (Clearance: 15ft)", "Bridge #5541 (Limit: 60 Tons)"]


def generate_example():
    """Generates a single synthetic query-response pair for domain adaptation."""
    
    origin = random.choice(CITIES)
    destination = random.choice([c for c in CITIES if c != origin])
    primary_route = random.choice(ROUTES)
    alt_route = random.choice([r for r in ROUTES if r != primary_route])
    
    disruption_type = random.choice(["weather", "accident", "traffic", "logistics", "facility"])
    if disruption_type == "weather":
        disruption = random.choice(WEATHER_ALERTS)
    elif disruption_type == "accident":
        disruption = random.choice(ACCIDENT_SEVERITY)
    elif disruption_type == "traffic":
        disruption = random.choice(TRAFFIC_ALERTS)
    elif disruption_type == "logistics":
        disruption = random.choice(LOGISTICS_ALERTS)
    else:
        disruption = random.choice(FACILITY_ALERTS)
        
    load = random.choice(LOAD_TYPES)
    
    # 30% chance of a structural veto on the alternate route
    is_veto = random.random() < 0.3
    
    query = f"Can we reroute the {load} shipment from {origin} to {destination} via {primary_route}? We just got a report of {disruption.lower()}."
    
    if is_veto:
        violation = random.choice(BRIDGE_VIOLATIONS)
        response = (
            f"Step 1 - Disruption: Verified {disruption.lower()} on primary route {primary_route}.\n"
            f"Step 2 - Route Analysis: Analyzed alternate route via {alt_route}.\n"
            f"Step 3 - Constraint Check: Failed. Alternate route contains {violation} which cannot support a {load} load profile.\n"
            f"Step 4 - Decision: VETO (Hold shipment at origin facility)."
        )
    else:
        safe_bridge = random.choice(SAFE_BRIDGES)
        response = (
            f"Step 1 - Disruption: Verified {disruption.lower()} on primary route {primary_route}.\n"
            f"Step 2 - Route Analysis: Analyzed alternate route via {alt_route}.\n"
            f"Step 3 - Constraint Check: Passed. Validated against National Bridge Inventory; safely accommodates {load} profile ({safe_bridge}).\n"
            f"Step 4 - Decision: APPROVED (Proceed with dispatch)."
        )

    return {
        "instruction": "Generate a constraint-compliant rerouting justification based on real-time disruptions and DOT physical constraints.",
        "input": f"{query} Context: Route options include {primary_route} and {alt_route}.",
        "output": response
    }

def main():
    dataset = []
    num_examples = 350
    
    for _ in range(num_examples):
        dataset.append(generate_example())
        
    output_file = "instruction_dataset.json"
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
            
    print(f"Successfully generated {num_examples} examples in {output_file}")
    
    # Print a sample
    print("\nSample Output:")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    main()
