import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Get the current working directory
current_directory = os.getcwd()

# Set the path to your data directory
data_path = os.path.join(current_directory, 'Data')

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

# Evaluate precision
rf_precision = precision_score(y_test, rf_pred, average='weighted')
svm_precision = precision_score(y_test, svm_pred, average='weighted')
knn_precision = precision_score(y_test, knn_pred, average='weighted')

# Evaluate recall
rf_recall = recall_score(y_test, rf_pred, average='weighted')
svm_recall = recall_score(y_test, svm_pred, average='weighted')
knn_recall = recall_score(y_test, knn_pred, average='weighted')

# Evaluate F1 score
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')

# Print evaluation metrics
print("\nRandom Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

print("\nSVM Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

print("\nKNN Metrics:")
print("Accuracy:", knn_accuracy)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1 Score:", knn_f1)

# Access feature importances from the trained Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print("Feature Importances:")
print(feature_importance_df)

####




def evaluate_model(model, X_test, y_test, model_name):
    # Predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print confusion matrix and other metrics
    print(f"Evaluation Metrics - {model_name}:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Evaluate Random Forest model
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Evaluate SVM model
evaluate_model(svm_model, X_test, y_test, "SVM")

# Evaluate KNN model
evaluate_model(knn_model, X_test, y_test, "KNN")

