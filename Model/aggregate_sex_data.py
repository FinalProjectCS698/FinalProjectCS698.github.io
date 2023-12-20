import pandas as pd
import os

# Get the current working directory
current_directory = os.getcwd()

# Set the path to your data directory
data_path = os.path.join(current_directory, 'Data')

def combine_sex_data():
    """
    Combines selected DataFrames with the 'Sex' label into one and saves it as a new CSV file.

    Parameters:
    None

    Returns:
    None
    """
    # Load each CSV file into a separate DataFrame
    files = [
        "Condensed_AnophelesFemale_Sexed.csv",
        "Condensed_AedesFemale_Sexed.csv",
        "Condensed_AnophelesMale_Sexed.csv",
        "Condensed_AedesMale_Sexed.csv",
    ]

    dfs = {}

    for file in files:
        name = file.split('.')[0]  # Extract the name without extension
        dfs[name] = pd.read_csv(os.path.join(data_path, file))

    # Combine selected DataFrames into one
    combined_sex_df = pd.concat(dfs.values(), ignore_index=True)

    # Drop the first column
    combined_sex_df = combined_sex_df.iloc[:, 1:]

    # Save the combined DataFrame to a new CSV file
    combined_sex_csv_path = os.path.join(data_path, 'Combined_Sex_Condensed_Data.csv')
    combined_sex_df.to_csv(combined_sex_csv_path, index=False)

    # Additional analysis on the combined DataFrame
    print("\nCombined Sex DataFrame:")
    print(combined_sex_df.info())
    print(combined_sex_df.describe())
    print(combined_sex_df['Family'].value_counts())  # Check distribution of family labels

# Execute the combining function
combine_sex_data()
