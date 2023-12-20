import pandas as pd
import os

# Get the current working directory
current_directory = os.getcwd()

# Set the path to your data directory
data_path = os.path.join(current_directory, 'Data')

def label_family():
    """
    Adds a 'Family' column to each condensed CSV file based on predefined family mapping.

    Parameters:
    None

    Returns:
    None
    """
    # Define the family mapping
    family_mapping = {
        "Condensed_HouseFly.csv": "HouseFly",
        "Condensed_AnophelesFemale.csv": "Anopheles",
        "Condensed_AedesFemale.csv": "Aedes",
        "Condensed_AnophelesMale.csv": "Anopheles",
        "Condensed_AedesMale.csv": "Aedes",
        "Condensed_BumbleBee.csv": "BumbleBee",
    }

    # Add the 'Family' column to each condensed CSV file
    for filename, family in family_mapping.items():
        df = pd.read_csv(os.path.join(data_path, filename))
        df['Family'] = family
        df.to_csv(os.path.join(data_path, filename), index=False)

    print("Labeling complete.")

# Execute the labeling function
label_family()
