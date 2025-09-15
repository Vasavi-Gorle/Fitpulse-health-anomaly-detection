import pandas as pd 
import numpy as np
from datetime import datetime, timedelta

print("Let's start building our data ingestion pipeline!")


def create_sample_heart_rate_csv():
    """Create sample heart rate data in CSV format"""

start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0) 
timestamps = []
heart_rates = []
for i in range(120):
    timestamp = start_time + timedelta(minutes=i) 
    base_hr = 75 + np.sin(i/30) * 10
    noise = np.random.normal(0, 5)
    hr = max(50, min(120, base_hr + noise))

    timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S')) 
    heart_rates.append(round(hr))

df = pd.DataFrame({ 
    'timestamp': timestamps, 
    'heart_rate_bpm': heart_rates
})

df.to_csv('sample_heart_rate.csv', index=False)
print("Created sample_heart_rate.csv")
return df

hr_data = create_sample_heart_rate_csv()
print(hr_data.head())