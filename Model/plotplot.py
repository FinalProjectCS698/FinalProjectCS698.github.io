import pandas as pd
import matplotlib.pyplot as plt

# Get the current working directory
current_directory = os.getcwd()

# Set the path to your data directory
data_path = os.path.join(current_directory, 'Data')

# Load the data from the specified file path
data = pd.read_csv(os.path.join(data_path, 'Combined_Condensed_Data.csv'))

# Get unique families in the dataset
unique_families = data['Family'].unique()

# Plotting the distribution of wingbeat frequencies by family
plt.figure(figsize=(10, 6))

for family in unique_families:
    family_data = data[data['Family'] == family]['WBF']
    plt.hist(family_data, bins=30, alpha=0.5, label=family)

plt.title('Distribution of Wingbeat Frequencies by Family')
plt.xlabel('Wingbeat Frequency')
plt.ylabel('Frequency Count')
plt.legend()
plt.grid(True)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Provide the full path to the file
file_path = r'C:\Users\Rachel\Documents\FinalProjData\Combined_Sex_Condensed_Data.csv'

# Load the data from the specified file path
data = pd.read_csv(file_path)

# Plotting the distribution of wingbeat frequencies by sex
plt.figure(figsize=(12, 8))
sns.histplot(data, x='WBF', bins=30, hue='Sex', palette='coolwarm', edgecolor='black')
plt.title('Distribution of Wingbeat Frequencies by Sex')
plt.xlabel('Wingbeat Frequency')
plt.ylabel('Frequency Count')
plt.legend(title='Sex')
plt.grid(True)
plt.show()
