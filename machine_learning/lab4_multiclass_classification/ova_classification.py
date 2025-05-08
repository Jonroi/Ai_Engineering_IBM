"""
Obesity Level Prediction using One-vs-All (OvA) Multiclass Classification
=========================================================================

This module demonstrates how to build a multiclass classification model
to predict obesity levels using the One-vs-All strategy.

Dataset: 
    Obesity level prediction dataset including various health metrics
    
Model: 
    Multiclass classification using One-vs-All Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
print("Dataset head:")
print(data.head())

# Display dataset information
print("\nChecking for missing values:")
print(data.isnull().sum())
print("\nDataset information:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())

# Visualize target distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.tight_layout()
plt.savefig('obesity_distribution.png')
plt.show()

# Data preprocessing
# Standardize continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Combine scaled features with original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Encode categorical features
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combine encoded features with dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encode target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
print("\nPreprocessed data head:")
print(prepped_data.head())

# Prepare features and target
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# ONE-VS-ALL (OVA) STRATEGY
# -------------------------------
print("\nTraining Logistic Regression with One-vs-All strategy...")
# Training logistic regression model using One-vs-All
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
model_ova.fit(X_train, y_train)

# Make predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluate model performance
accuracy = np.round(100*accuracy_score(y_test, y_pred_ova), 2)
print(f"One-vs-All (OvA) Strategy Accuracy: {accuracy}%")

# Display model parameters
print("\nModel classes:", model_ova.classes_)
print("Number of binary classifiers:", len(model_ova.classes_))

# Save the model
import joblib
joblib.dump(model_ova, 'obesity_ova_model.pkl')
print("Model saved as 'obesity_ova_model.pkl'")