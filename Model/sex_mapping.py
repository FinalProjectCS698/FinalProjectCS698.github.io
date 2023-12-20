import pandas as pd
import os

# Set the path to your data directory
data_path = r'C:\Users\Rachel\Documents\FinalProjData'

def label_sex():
    """
    Adds a 'Sex' column to each condensed CSV file based on predefined sex mapping.

    Parameters:
    None

    Returns:
    None
    """
    # Define the family mapping for sex
    sex_mapping = {
        "Condensed_AnophelesFemale.csv": "Female",
        "Condensed_AedesFemale.csv": "Female",
        "Condensed_AnophelesMale.csv": "Male",
        "Condensed_AedesMale.csv": "Male",
    }

    # Create a new DataFrame for each existing one with the added 'Sex' column and a new label
    for filename, sex in sex_mapping.items():
        df = pd.read_csv(os.path.join(data_path, filename))

        # Add the 'Sex' column
        df['Sex'] = sex

        # Create a new DataFrame with a new label
        new_label = f"{filename.split('.')[0]}_Sexed.csv"
        new_label_csv_path = os.path.join(data_path, new_label)
        df.to_csv(new_label_csv_path, index=False)

    print("Labeling complete.")

# Execute the labeling function
label_sex()
