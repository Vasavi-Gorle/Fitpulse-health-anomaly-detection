import json


with open("fitness_data_sample.json", "r") as f:
    data = json.load(f)


print("Type of data loaded:", type(data))


print("\nFirst 2 entries from JSON file:")
print(json.dumps(data[:2], indent=4))


print("\nNumber of records in JSON file:", len(data))
