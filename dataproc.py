import pandas as pd
import os

# Set the path to your data directory
data_path = r'C:\Users\Rachel\Documents\FinalProjData'


# Function to preprocess and create condensed CSV files
def preprocess_and_save(input_filename, output_filename, selected_columns_labels):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(os.path.join(data_path, input_filename))

    # Remove the first row (header)
    df = df.iloc[1:]

    # Select specific columns by label
    df = df[selected_columns_labels]

    # Rename columns for clarity
    df.columns = ['Transit_Time', 'WBF', 'Body_Contribution', 'Wing_Contribution']

    # Create a new column for the ratio between body and wing contribution
    df['Body_Wing_Ratio'] = df['Body_Contribution'] / df['Wing_Contribution']

    # Save the condensed DataFrame to a new CSV file
    df.to_csv(os.path.join(data_path, output_filename), index=True)
    print(f'{input_filename} processed and saved as {output_filename}')

# Specify details for HouseFly.csv
input_filenameHF = "HouseFly.csv"
output_filenameHF = "Condensed_HouseFly.csv"
selected_columns_labelsHF = ['IndexTotHouseFly6', 'IndexTotHouseFly46', 'IndexTotHouseFly64', 'IndexTotHouseFly65']

# Specify details for BumbleBee.csv
input_filenameBB = "BumbleBee.csv"
output_filenameBB = "Condensed_BumbleBee.csv"
selected_columns_labelsBB = ['IndexTotBB6', 'IndexTotBB46', 'IndexTotBB64', 'IndexTotBB65']

# Specify details for AnophelesFemale.csv
input_filenameANF = "AnophelesFemale.csv"
output_filenameANF = "Condensed_AnophelesFemale.csv"
selected_columns_labelsANF = ['IndexTotAnophelesFemale6', 'IndexTotAnophelesFemale46', 'IndexTotAnophelesFemale64', 'IndexTotAnophelesFemale65']

# Specify details for AnophelesMale.csv
input_filenameANM = "AnophelesMale.csv"
output_filenameANM = "Condensed_AnophelesMale.csv"
selected_columns_labelsANM = ['IndexTotAnophelesMale6', 'IndexTotAnophelesMale46', 'IndexTotAnophelesMale64', 'IndexTotAnophelesMale65']

# Specify details for AedesFemale.csv
input_filenameAF = "AedesFemale.csv"
output_filenameAF = "Condensed_AedesFemale.csv"
selected_columns_labelsAF = ['IndexTotAedesFemale6', 'IndexTotAedesFemale46', 'IndexTotAedesFemale64', 'IndexTotAedesFemale65']

# Specify details for AedesMale.csv
input_filenameAM = "AedesMale.csv"
output_filenameAM = "Condensed_AedesMale.csv"
selected_columns_labelsAM = ['IndexTotAedesMale6', 'IndexTotAedesMale46', 'IndexTotAedesMale64', 'IndexTotAedesMale65']



# Process HouseFly.csv
preprocess_and_save(input_filenameHF, output_filenameHF, selected_columns_labelsHF)

# Process BumbleBee.csv
preprocess_and_save(input_filenameBB, output_filenameBB, selected_columns_labelsBB)

# Process AnophelesFemale.csv
preprocess_and_save(input_filenameANF, output_filenameANF, selected_columns_labelsANF)

# Process AnophelesMale.csv
preprocess_and_save(input_filenameANM, output_filenameANM, selected_columns_labelsANM)

# Process AedesFemale.csv
preprocess_and_save(input_filenameAF, output_filenameAF, selected_columns_labelsAF)

# Process AedesMale.csv
preprocess_and_save(input_filenameAM, output_filenameAM, selected_columns_labelsAM)

print("Data preprocessing complete.")
