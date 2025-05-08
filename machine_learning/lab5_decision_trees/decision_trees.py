"""
Decision Tree Classification for Drug Prescription

This script demonstrates the use of Decision Trees to classify which drug might be appropriate based on patient features.
Dataset: Drug200.csv - contains patient information and the drug they were prescribed.

Features:
- Age: Age of the patient
- Sex: Gender of the patient
- BP: Blood Pressure level (LOW, NORMAL, HIGH)
- Cholesterol: Cholesterol level (NORMAL, HIGH)
- Na_to_K: Sodium to Potassium ratio in blood

Target:
- Drug: Type of drug prescribed (DrugA, DrugB, DrugC, DrugX, DrugY)
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# Load the dataset
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

# Display the first few rows of the dataset
print("Dataset preview:")
print(my_data.head())

# Display dataset information
print("\nDataset information:")
my_data.info()

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 

# Check for missing values
print("\nChecking for missing values:")
print(my_data.isnull().sum())

# Analyze the distribution of target variable
category_counts = my_data['Drug'].value_counts()
print("\nDrug type distribution:")
print(category_counts)

# Create a pie chart for drug distribution
plt.figure(figsize=(10, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Drug Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.savefig('drug_distribution.png')
plt.show()

# Prepare data for modeling
y = my_data['Drug']  # Target variable
X = my_data.drop('Drug', axis=1)  # Features

# Split data into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)
print("\nData split into training and testing sets:")
print(f"Training set shape: {X_trainset.shape}")
print(f"Testing set shape: {X_testset.shape}")

# Create and train the decision tree classifier
print("\nTraining the Decision Tree model...")
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_trainset, y_trainset)

# Make predictions and evaluate the model
tree_predictions = drugTree.predict(X_testset)
accuracy = metrics.accuracy_score(y_testset, tree_predictions)
print("\nModel Evaluation:")
print(f"Decision Tree's Accuracy: {accuracy:.4f}")

# Additional evaluation metrics
print("\nClassification Report:")
print(metrics.classification_report(y_testset, tree_predictions))

# Display confusion matrix
print("\nConfusion Matrix:")
conf_matrix = metrics.confusion_matrix(y_testset, tree_predictions)
print(conf_matrix)

# Visualize the decision tree with enhanced formatting
plt.figure(figsize=(20, 10))
feature_names = X.columns
class_names = list(my_data['Drug'].unique())
plot_tree(drugTree, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Drug Prescription", fontsize=16)
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance
feature_importance = pd.Series(drugTree.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance in Decision Tree Model')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nModel training and evaluation completed successfully.")
print("Visualizations saved as 'decision_tree_visualization.png', 'drug_distribution.png', and 'feature_importance.png'")