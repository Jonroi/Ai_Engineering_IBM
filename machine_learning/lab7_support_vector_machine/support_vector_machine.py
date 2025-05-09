from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, 
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.svm import LinearSVC
import warnings

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Download the credit card fraud detection dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# Read the input data
print("Loading dataset from URL...")
raw_data = pd.read_csv(url)
print(f"Dataset loaded successfully with shape: {raw_data.shape}")
print("First 5 rows of the dataset:")
print(raw_data.head())

# get the set of distinct classes
labels = raw_data.Class.unique()

# get the count of each class
sizes = raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()

# Find features most correlated with the target
print("\nTop 6 features with highest correlation to the target variable (Class):")
correlation_values = abs(raw_data.corr()['Class']).drop('Class')
correlation_values = correlation_values.sort_values(ascending=False)[:6]
print(correlation_values)

# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we use the top 6 correlated features)
X = data_matrix[:,[3,10,12,14,16,17]]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compute sample weights to handle class imbalance
w_train = compute_sample_weight('balanced', y_train)

# Train Decision Tree model
print("\nTraining Decision Tree model...")
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)

# Train SVM model with increased iterations to ensure convergence
print("\nTraining SVM model...")
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", 
                fit_intercept=True, max_iter=10000, tol=1e-4)
svm.fit(X_train, y_train)

# Get predictions from both models
print("\nGenerating predictions...")
y_pred_dt = dt.predict_proba(X_test)[:, 1]  # Probability scores for Decision Tree
y_pred_svm = svm.decision_function(X_test)   # Decision scores for SVM

# Calculate ROC AUC scores for both models
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

# Get binary predictions for additional metrics
dt_binary_pred = dt.predict(X_test)
svm_binary_pred = svm.predict(X_test)

# Print model performance metrics
print("\n===== Model Performance Comparison =====")
print("Decision Tree ROC-AUC score: {0:.3f}".format(roc_auc_dt))
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

print("\n----- Decision Tree Performance -----")
print(f"Accuracy: {accuracy_score(y_test, dt_binary_pred):.3f}")
print("Classification Report:")
print(classification_report(y_test, dt_binary_pred))

print("\n----- SVM Performance -----")
print(f"Accuracy: {accuracy_score(y_test, svm_binary_pred):.3f}")
print("Classification Report:")
print(classification_report(y_test, svm_binary_pred))

# Plot confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm_dt = confusion_matrix(y_test, dt_binary_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
cm_svm = confusion_matrix(y_test, svm_binary_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

# Plot ROC curves for both models
plt.figure(figsize=(8, 6))
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)

plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (ROC-AUC = {roc_auc_dt:.3f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (ROC-AUC = {roc_auc_svm:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png')
plt.show()

print("\nAnalysis complete! Visualization files saved to the current directory.")