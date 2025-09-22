# -------------------------------
# Import Required Libraries
# -------------------------------
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


timestamps = [datetime.now() - timedelta(minutes=10*i) for i in range(72)]
timestamps = list(reversed(timestamps))  # oldest to newest


heart_rate = np.random.randint(60, 120, size=72)  # bpm
steps = np.random.randint(0, 50, size=72)         # steps in 10 min
sleep_stage = np.random.choice(["Awake", "Light", "Deep", "REM"], size=72)



df = pd.DataFrame({
    "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "heart_rate_bpm": heart_rate,
    "steps": steps,
    "sleep_stage": sleep_stage,
})



json_file = "fitness_data_sample.json"
json_data = df.to_dict(orient="records")  # list of dictionaries
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print("JSON file created successfully:", json_file)

