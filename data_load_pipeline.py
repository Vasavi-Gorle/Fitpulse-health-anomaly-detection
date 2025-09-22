import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


csv_file = "data/raw/fitness_data_sample.csv"
json_file = "data/raw/fitness_data_sample.json"

try:
    df_sample = pd.read_csv(csv_file)
    print("CSV file already exists.")
except:
    print("Creating sample CSV and JSON files...")
    # Generate timestamps
    timestamps = [datetime.now() - timedelta(minutes=10*i) for i in range(72)]
    timestamps = list(reversed(timestamps))
    
    heart_rate = np.random.randint(60, 120, size=72)
    steps = np.random.randint(0, 50, size=72)
    sleep_stage = np.random.choice(["Awake", "Light", "Deep", "REM"], size=72)
    calories = np.random.uniform(0.5, 5.0, size=72).round(2)
    
    df_sample = pd.DataFrame({
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "heart_rate_bpm": heart_rate,
        "steps": steps,
        "sleep_stage": sleep_stage,
        "calories_burned": calories
    })
    
    # Save CSV and JSON
    try:
        df_sample.to_csv(csv_file, index=False)
    except:
        # If folder doesn't exist, just save in current directory
        df_sample.to_csv("fitness_data_sample.csv", index=False)
        csv_file = "fitness_data_sample.csv"

    try:
        df_sample.to_json(json_file, orient="records", indent=4)
    except:
        df_sample.to_json("fitness_data_sample.json", orient="records", indent=4)
        json_file = "fitness_data_sample.json"
    print("Sample files created.")


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        try:
            if self.filepath.endswith(".csv"):
                print("Loading CSV file...")
                return pd.read_csv(self.filepath)
            elif self.filepath.endswith(".json"):
                print("Loading JSON file...")
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format! Use .csv or .json")
        except:
            raise FileNotFoundError(f"File not found: {self.filepath}")


class Preprocessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], errors="coerce")
        self.df = self.df.dropna(subset=["timestamp"])
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.df[col].fillna(self.df[col].mean(), inplace=True)
        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print("Preprocessing complete")
        return self.df


class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def create_features(self):
        self.df["hr_rolling_avg"] = self.df["heart_rate_bpm"].rolling(window=3, min_periods=1).mean()
        self.df["activity_level"] = pd.cut(
            self.df["steps"],
            bins=[-1, 10, 30, 100],
            labels=["Low", "Medium", "High"]
        )
        print("Feature engineering complete")
        return self.df


class AnomalyDetector:
    def __init__(self, df):
        self.df = df

    def detect(self):
        self.df["anomaly"] = "Normal"
        self.df.loc[self.df["heart_rate_bpm"] > 160, "anomaly"] = "High HR"
        self.df.loc[self.df["heart_rate_bpm"] < 40, "anomaly"] = "Low HR"
        self.df.loc[(self.df["steps"] == 0) & (self.df["calories_burned"] > 3.5), "anomaly"] = "Suspicious Calories"
        print("✅ Anomaly detection complete")
        return self.df


if __name__ == "__main__":
    # Prefer CSV
    filepath = csv_file if csv_file else json_file

    loader = DataLoader(filepath)
    df = loader.load_data()
    print("\n🔹 Data Loaded:")
    print(df.head())

    pre = Preprocessor(df)
    df_clean = pre.clean_data()

    feat = FeatureEngineer(df_clean)
    df_features = feat.create_features()

    anomaly = AnomalyDetector(df_features)
    df_final = anomaly.detect()

    # Save processed data (current directory)
    df_final.to_csv("cleaned_fitness_data.csv", index=False)
    df_final.to_json("cleaned_fitness_data.json", orient="records", indent=4)

    print("\n Pipeline completed successfully!")
    print(" Processed data saved as cleaned_fitness_data.csv & .json")
