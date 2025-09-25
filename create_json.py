import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


timestamps = [datetime.now() - timedelta(minutes=10*i) for i in range(250)]
timestamps = list(reversed(timestamps))  # oldest to newest


heart_rate = np.random.randint(60, 120, size=250)  # bpm
steps = np.random.randint(0, 50, size=250)         # steps in 10 min
sleep_hours = np.random.randint(0,8, size=250)



df = pd.DataFrame({
    "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "heart_rate_bpm": heart_rate,
    "steps": steps,
    "sleep_hours": sleep_hours
})



json_file = "fitness_data_sample.json"
json_data = df.to_dict(orient="records")  # list of dictionaries
with open(json_file, "w") as f:
    json.dump(json_data, f, indent=4)

print("JSON file created successfully:", json_file)

