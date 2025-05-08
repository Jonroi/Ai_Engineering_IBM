import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the data
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

print("Data loaded successfully")
print(f"Dataset shape: {data.shape}")

# Standardize continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

# Identify categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

# Encode the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

# Prepare final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train models
model_ova = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
model_ova.fit(X_train, y_train)

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000, random_state=42))
model_ovo.fit(X_train, y_train)

# Feature importance for One-vs-All (OvA) model
plt.figure(figsize=(12, 8))
# Correctly extract coefficients from the OvA model
feature_importance_ova = np.mean(np.abs([estimator.coef_[0] for estimator in model_ova.estimators_]), axis=0)
sorted_idx = np.argsort(feature_importance_ova)
plt.barh(X.columns[sorted_idx], feature_importance_ova[sorted_idx])
plt.title("Feature Importance - One-vs-All (OvA)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance_ova.png')
plt.show()

# Feature importance for One-vs-One (OvO) model
# For OvO, we need to extract coefficients from each binary classifier
all_coefs = []
for estimator in model_ovo.estimators_:
    all_coefs.append(estimator.coef_[0])

# Average the absolute values of coefficients across all binary classifiers
plt.figure(figsize=(12, 8))
feature_importance_ovo = np.mean(np.abs(all_coefs), axis=0)
sorted_idx = np.argsort(feature_importance_ovo)
plt.barh(X.columns[sorted_idx], feature_importance_ovo[sorted_idx])
plt.title("Feature Importance - One-vs-One (OvO)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importance_ovo.png')
plt.show()

# Compare the top 5 most important features for both models
ova_top5_idx = np.argsort(feature_importance_ova)[-5:]
ovo_top5_idx = np.argsort(feature_importance_ovo)[-5:]

print("\nTop 5 most important features (OvA):")
for idx in reversed(ova_top5_idx):
    print(f"- {X.columns[idx]}: {feature_importance_ova[idx]:.4f}")

print("\nTop 5 most important features (OvO):")
for idx in reversed(ovo_top5_idx):
    print(f"- {X.columns[idx]}: {feature_importance_ovo[idx]:.4f}")

# Check if the top features differ between models
common_features = set([X.columns[i] for i in ova_top5_idx]).intersection(set([X.columns[i] for i in ovo_top5_idx]))
print(f"\nNumber of common top features: {len(common_features)}")
print(f"Common top features: {common_features}")