import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set the path to your data directory
data_path = r'C:\Users\Rachel\Documents\FinalProjData'

# Read the combined CSV file into a DataFrame
combined_csv_path = os.path.join(data_path, 'Combined_Sex_Condensed_Data.csv')
combined_df = pd.read_csv(combined_csv_path)

# Features (X) and target variable (y)
X = combined_df.drop(['Family', 'Sex'], axis=1)  # Assuming 'Sex' is the target variable
y = combined_df['Sex']

# Split the data into training (80%) and the rest (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the rest into validation (50%) and testing (50%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of the resulting sets
print("Before handling missing values:")
print("Training set - Features:", X_train.shape, " Labels:", y_train.shape)
print("Testing set - Features:", X_test.shape, " Labels:", y_test.shape)

# Drop rows with missing values in the training set
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]  # Adjust labels accordingly

# Drop rows with missing values in the testing set
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]  # Adjust labels accordingly

# Display the shapes of the resulting sets after handling missing values
print("\nAfter handling missing values:")
print("Training set - Features:", X_train.shape, " Labels:", y_train.shape)
print("Testing set - Features:", X_test.shape, " Labels:", y_test.shape)

# Initialize models
rf_model = RandomForestClassifier()
svm_model = SVC()
knn_model = KNeighborsClassifier()

# Train models on the training set
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Predictions on the testing set
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Evaluate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
svm_accuracy = accuracy_score(y_test, svm_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Print accuracies
print("Random Forest Accuracy:", rf_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)

# Access feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("Feature Importances:")
print(feature_importance_df)
