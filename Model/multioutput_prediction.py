import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# Get the current working directory
current_directory = os.getcwd()

# Set the path to your data directory
data_path = os.path.join(current_directory, 'Data')

# Read the combined CSV file into a DataFrame
combined_csv_path = os.path.join(data_path, 'Combined_Sex_Condensed_Data.csv')
combined_df = pd.read_csv(combined_csv_path)

# Features (X) and target variables (y_sex and y_family)
X = combined_df.drop(['Family', 'Sex'], axis=1)  
y_sex = combined_df['Sex']
y_family = combined_df['Family']

# Split the data into training (80%) and the rest (20%)
X_train, X_temp, y_sex_train, y_sex_temp, y_family_train, y_family_temp = train_test_split(
    X, y_sex, y_family, test_size=0.2, random_state=42
)

# Split the rest into validation (50%) and testing (50%)
X_val, X_test, y_sex_val, y_sex_test, y_family_val, y_family_test = train_test_split(
    X_temp, y_sex_temp, y_family_temp, test_size=0.5, random_state=42
)

# Display the shapes of the resulting sets
print("Before handling missing values:")
print("Training set - Features:", X_train.shape, " Sex Labels:", y_sex_train.shape, " Family Labels:", y_family_train.shape)
print("Testing set - Features:", X_test.shape, " Sex Labels:", y_sex_test.shape, " Family Labels:", y_family_test.shape)

# Drop rows with missing values in the training set
X_train = X_train.dropna()
y_sex_train = y_sex_train.loc[X_train.index]  # Adjust labels accordingly
y_family_train = y_family_train.loc[X_train.index]  # Adjust labels accordingly

# Drop rows with missing values in the testing set
X_test = X_test.dropna()
y_sex_test = y_sex_test.loc[X_test.index]  # Adjust labels accordingly
y_family_test = y_family_test.loc[X_test.index]  # Adjust labels accordingly

# Display the shapes of the resulting sets after handling missing values
print("\nAfter handling missing values:")
print("Training set - Features:", X_train.shape, " Sex Labels:", y_sex_train.shape, " Family Labels:", y_family_train.shape)
print("Testing set - Features:", X_test.shape, " Sex Labels:", y_sex_test.shape, " Family Labels:", y_family_test.shape)

# Initialize a multi-output Random Forest model
multi_output_rf_model = MultiOutputClassifier(RandomForestClassifier())

# Train the model on the training set
multi_output_rf_model.fit(X_train, pd.concat([y_sex_train, y_family_train], axis=1))

# Predictions on the testing set
multi_output_rf_pred = multi_output_rf_model.predict(X_test)

# Extract predictions for sex and family
y_sex_pred = multi_output_rf_pred[:, 0]
y_family_pred = multi_output_rf_pred[:, 1]

# Evaluate accuracy for sex prediction
sex_accuracy = accuracy_score(y_sex_test, y_sex_pred)
print("Sex Prediction Accuracy:", sex_accuracy)

# Evaluate accuracy for family prediction
family_accuracy = accuracy_score(y_family_test, y_family_pred)
print("Family Prediction Accuracy:", family_accuracy)

# Calculate confusion matrix for sex prediction
cm_sex = confusion_matrix(y_sex_test, y_sex_pred)

# Calculate confusion matrix for family prediction
cm_family = confusion_matrix(y_family_test, y_family_pred)

# Extract true positives from confusion matrix
tp_sex = cm_sex[1, 1]
tp_family = cm_family[1, 1]

# Print the number of true positives
print("True Positives - Sex Prediction:", tp_sex)
print("True Positives - Family Prediction:", tp_family)

# Plot confusion matrix for sex prediction
plt.figure(figsize=(8, 6))
sns.heatmap(cm_sex, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Sex Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot confusion matrix for family prediction
plt.figure(figsize=(8, 6))
sns.heatmap(cm_family, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Family Prediction")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Extract feature importances
feature_importances = multi_output_rf_model.estimators_[0].feature_importances_

# Create a DataFrame to display feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("Feature Importances for Sex Prediction:")
print(feature_importance_df)

# Evaluate precision, recall, and F1 score for sex prediction
sex_precision = precision_score(y_sex_test, y_sex_pred, average='weighted')
sex_recall = recall_score(y_sex_test, y_sex_pred, average='weighted')
sex_f1 = f1_score(y_sex_test, y_sex_pred, average='weighted')

# Print evaluation metrics for sex prediction
print("\nSex Prediction Metrics:")
print("Precision:", sex_precision)
print("Recall:", sex_recall)
print("F1 Score:", sex_f1)

# Evaluate precision, recall, and F1 score for family prediction
family_precision = precision_score(y_family_test, y_family_pred, average='weighted')
family_recall = recall_score(y_family_test, y_family_pred, average='weighted')
family_f1 = f1_score(y_family_test, y_family_pred, average='weighted')

# Print evaluation metrics for family prediction
print("\nFamily Prediction Metrics:")
print("Precision:", family_precision)
print("Recall:", family_recall)
print("F1 Score:", family_f1)

# Generate classification report for sex prediction
sex_classification_report = classification_report(y_sex_test, y_sex_pred)
print("\nSex Prediction Classification Report:")
print(sex_classification_report)

# Generate classification report for family prediction
family_classification_report = classification_report(y_family_test, y_family_pred)
print("\nFamily Prediction Classification Report:")
print(family_classification_report)
