import pandas as pd
import os

# Set the path to your data directory
data_path = r'C:\Users\Rachel\Documents\FinalProjData'

# Load each CSV file into a separate DataFrame and remove the first column
files = [
    "Condensed_HouseFly.csv",
    "Condensed_AnophelesFemale.csv",
    "Condensed_AedesFemale.csv",
    "Condensed_AnophelesMale.csv",
    "Condensed_AedesMale.csv",
    "Condensed_BumbleBee.csv",
]

dfs = []

for file in files:
    df = pd.read_csv(os.path.join(data_path, file))
    
    # Remove the first column if it exists
    if len(df.columns) > 0:
        df = df.iloc[:, 1:]
    
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=False)

# Save the combined DataFrame to a new CSV file
combined_csv_path = os.path.join(data_path, 'Combined_Condensed_Data.csv')
combined_df.to_csv(combined_csv_path, index=False)

# Additional analysis on the combined DataFrame
print("\nCombined DataFrame:")
print(combined_df.info())
print(combined_df.describe())
print(combined_df['Family'].value_counts())  # Check distribution of family labels
