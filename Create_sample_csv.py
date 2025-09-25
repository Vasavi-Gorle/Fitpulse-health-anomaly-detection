import pandas as pd
import numpy as np
from datetime import datetime, timedelta


timestamps = [datetime.now() - timedelta(minutes=10*i) for i in range(250)]
timestamps = list(reversed(timestamps))  # oldest to newest

heart_rate = np.random.randint(60, 120, size=250)  # bpm
steps = np.random.randint(0, 50, size=250)         # steps in 10 min interval
sleep_hours = np.random.randint(0,10, size=250)


df = pd.DataFrame({
    "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
    "heart_rate_bpm": heart_rate,
    "steps": steps,
    "sleep_hours":sleep_hours
})

csv_file = "fitness_data_sample.csv"
df.to_csv(csv_file, index=False)

print("CSV file created successfully:", csv_file)
