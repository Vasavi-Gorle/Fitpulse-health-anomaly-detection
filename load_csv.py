import pandas as pd

csv_file = "fitness_data_sample.csv" 
df = pd.read_csv(csv_file)


print("CSV file loaded successfully!\n")
print(df.head())  


print("\n Dataset Info:")
print(df.info())

print("\n Summary Statistics:")
print(df.describe())
