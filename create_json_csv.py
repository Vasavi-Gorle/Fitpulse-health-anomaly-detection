import pandas as pd 
import numpy as np 
import json
from datetime import datetime, timedelta


def create_sample_json_data():
    """Create sample fitness data in JSON format"""

    # Create comprehensive fitness data
    fitness_data = {
        "user_id": "user_123",
        "device": "Fitness Tracker Pro",
        "export_date": datetime.now().strftime('%Y-%m-%d'),
        "data_types": ["heart_rate", "steps", "sleep"],
        "heart_rate_data": [],
        "step_data": [],
        "sleep_data": []
    }

    # Generate heart rate data
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) 
    for i in range(60):  # 1 hour of data
        timestamp = start_time + timedelta(minutes=i)
        hr = 70 + np.random.normal(0, 8)  # More variation
        fitness_data["heart_rate_data"].append({
            "timestamp": timestamp.isoformat(), 
            "bpm": max(50, min(150, int(hr))),
            "confidence": np.random.uniform(0.8, 1.0)  # Sensor confidence
        })

    # Generate step data 
    for i in range(60):
        timestamp = start_time + timedelta(minutes=i)
        steps = np.random.poisson(12)  # Average 12 steps per minute
        fitness_data["step_data"].append({ 
            "timestamp": timestamp.isoformat(),
            "steps": steps,
            "cadence": steps * 60 if steps > 0 else 0  # Steps per minute
        })

    # Save JSON file
    with open('sample_fitness_data.json', 'w') as f: 
        json.dump(fitness_data, f, indent=2)

    print("Created sample_fitness_data.json") 
    return fitness_data


# Run the function
json_data = create_sample_json_data()
print(f"JSON contains {len(json_data['heart_rate_data'])} heart rate readings")
