
import json
import time
import random
from pathlib import Path

# Create data directory
Path("data/simulation").mkdir(parents=True, exist_ok=True)

# Generate test data
vehicles = ["AV_001", "AV_002", "AV_003", "AV_004", "AV_005"]
data = []

for i in range(100):  # 100 data points
    for vehicle in vehicles:
        record = {
            "vehicle_id": vehicle,
            "timestamp": time.time() + i,
            "location": {
                "latitude": 37.7749 + random.uniform(-0.01, 0.01),
                "longitude": -122.4194 + random.uniform(-0.01, 0.01)
            },
            "velocity": {
                "x": random.uniform(0, 20),
                "y": random.uniform(0, 20), 
                "z": 0
            },
            "calculated_speed": random.uniform(10, 30),
            "risk_level": random.choice(["low", "medium", "high"]),
            "anomalies": []
        }
        data.append(record)

# Save data
with open("data/simulation/test_data.json", "w") as f:
    json.dump(data, f)

print(f"Generated {len(data)} test records")
