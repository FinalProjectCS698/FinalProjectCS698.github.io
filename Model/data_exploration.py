import pandas as pd
import os

# Set the path to your data directory
data_path = r'C:\Users\Rachel\Documents\FinalProjData'

# Dictionary to store DataFrames
dfs = {}

# List of input filenames
files = ["Condensed_HouseFly.csv", "Condensed_AnophelesMale.csv", "Condensed_AnophelesFemale.csv",
         "Condensed_BumbleBee.csv", "Condensed_AedesMale.csv", "Condensed_AedesFemale.csv"]

# Read each CSV file into a DataFrame and store in the dictionary
for file in files:
    name = os.path.splitext(file)[0]  # Extract the name without extension
    dfs[name] = pd.read_csv(os.path.join(data_path, file))

# Inspect each DataFrame in the 'dfs' dictionary
for name, df in dfs.items():
    print(f"\n{name} DataFrame:")
    print(df.info())
    print(df.describe())
    print(df['Family'].value_counts())  # Check distribution of family labels
