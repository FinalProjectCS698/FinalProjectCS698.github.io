This project involves preprocessing and analyzing entomological data related to different insect species. The following scripts should be executed in a specific order to ensure proper data processing and analysis.

Prerequisites
Before running the scripts, ensure that you have the following prerequisites installed:

- Python 3.11.4
- Required Python packages (install using pip install -r requirements.txt)

Scripts
1. data_processing.py
   Purpose: Preprocesses and condenses CSV files for various insect species.
   How to Run: python data_processing.py

2. data_exploration.py
   Purpose: Reads the condensed CSV files, inspects the data, and checks the distribution of family labels.
   How to Run: python data_exploration.py

3. family_mapping.py
   Purpose: Adds a 'Family' column to each condensed CSV file based on a predefined mapping.
   How to Run: python family_mapping.py

4. sex_mapping.py
   Purpose: Adds a 'Sex' column to selected CSV files based on a predefined mapping and saves them with new labels.
   How to Run: python sex_mapping.py

5. aggregate_family_data.py
   Purpose: Combines all condensed CSV files into one, handling missing values and providing additional analysis.
   How to Run: python aggregate_family_data.py

6. aggregate_sex_data.py
   Purpose: Combines selected CSV files related to sex into one, handling missing values and providing additional analysis.
   How to Run: python aggregate_sex_data.py

7. train_family_prediction_models.py
   Purpose: Handles missing values, trains machine learning models (Random Forest, SVM, KNN) on family prediction, and evaluates their performance.
   How to Run: python train_family_prediction_models.py

8. train_sex_prediction_models.py
   Purpose: Similar to train_family_prediction_models.py but focuses on sex prediction.
   How to Run: python train_sex_prediction_models.py

9. multioutput_prediction.py
   Purpose: Uses a multi-output Random Forest model to predict both sex and family, evaluates accuracy, and visualizes confusion matrices.
   How to Run: python multioutput_prediction.py

Note
Ensure that you have the required permissions to read and write data in the specified data directory.
Review and modify the data_path variable in each script according to your local setup.
