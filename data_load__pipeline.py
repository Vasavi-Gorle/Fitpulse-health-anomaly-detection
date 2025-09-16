import pandas as pd

def load_and_process(csv_file, json_file):
    df_csv = pd.read_csv(csv_file)
    df_json = pd.read_json(json_file)

    # Combine both datasets
    df = pd.concat([df_csv, df_json])

    # Fill missing values
    df.fillna(method="ffill", inplace=True)

    return df

if __name__ == "__main__":
    df = load_and_process("sample_heart_rate.csv", "sample_fitness_data.json")
    print(df.head())
