import json
import random

# Domain knowledge for generating synthetic examples
CITIES = ["Chicago", "Kansas City", "Omaha", "Denver", "Salt Lake City", "Dallas", "Houston", "Atlanta", "Charlotte"]
ROUTES = ["I-70", "I-80", "I-55", "I-10", "I-25", "I-35"]
WEATHER_ALERTS = ["Heavy Snowfall (>10cm)", "Severe Icing", "Flash Flooding", "Low Visibility (<1mi)", "High Winds"]
ACCIDENT_SEVERITY = ["Major accident (Severity 4)", "Pileup (Severity 3)"]
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
    
    disruption_type = random.choice(["weather", "accident"])
    disruption = random.choice(WEATHER_ALERTS) if disruption_type == "weather" else random.choice(ACCIDENT_SEVERITY)
    
    load = random.choice(LOAD_TYPES)
    
    # 30% chance of a structural veto on the alternate route
    is_veto = random.random() < 0.3
    
    query = f"Can we reroute the {load} shipment from {origin} to {destination} via {primary_route}? We just got a report of {disruption.lower()}."
    
    if is_veto:
        violation = random.choice(BRIDGE_VIOLATIONS)
        response = (
            f"VETO: The proposed reroute for the {load} shipment from {origin} to {destination} violates DOT safety constraints. "
            f"While {primary_route} is experiencing {disruption.lower()}, the alternate route via {alt_route} contains "
            f"{violation}. You must hold the shipment at the origin facility until the {disruption_type} clears or find an alternative route "
            f"that supports a {load} load profile."
        )
    else:
        safe_bridge = random.choice(SAFE_BRIDGES)
        response = (
            f"APPROVED: Rerouting the {load} shipment from {origin} to {destination} is approved. "
            f"Historical and real-time data confirm {disruption} on {primary_route}. "
            f"The alternate route via {alt_route} has been validated against the National Bridge Inventory "
            f"and safely accommodates the {load} profile ({safe_bridge} verified). Proceed with dispatch."
        )

    return {
        "messages": [
            {"role": "system", "content": "You are the HyperLogistics Validation & Safety Agent. You provide explainable, constraint-compliant rerouting justifications for middle-mile logistics based on real-time weather, accidents, and US DOT physical constraints (bridge clearance/weight limits)."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
    }

def main():
    dataset = []
    num_examples = 100
    
    for _ in range(num_examples):
        dataset.append(generate_example())
        
    output_file = "synthetic_rerouting_dataset.jsonl"
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully generated {num_examples} examples in {output_file}")
    
    # Print a sample
    print("\nSample Output:")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    main()
