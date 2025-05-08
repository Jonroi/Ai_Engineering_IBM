import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# Load and preprocess the data
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)

print("Data loaded successfully")
print(f"Dataset shape: {data.shape}")
print("\nTarget variable distribution:")
print(data['NObeyesdad'].value_counts())

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
print(f"\nEncoded target classes: {prepped_data['NObeyesdad'].unique()}")

# Prepare final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define a function to train and evaluate models
def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    # Record start time
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Record training time
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), 
                yticklabels=sorted(y.unique()))
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    
    return accuracy, train_time, y_pred

# 1. Standard Multiclass Logistic Regression
standard_model = LogisticRegression(max_iter=1000, random_state=42)
standard_accuracy, standard_time, standard_pred = train_and_evaluate(
    standard_model, "Standard Multiclass", X_train, y_train, X_test, y_test
)

# 2. One-vs-All (OvA) approach
ova_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
ova_accuracy, ova_time, ova_pred = train_and_evaluate(
    ova_model, "One-vs-All (OvA)", X_train, y_train, X_test, y_test
)

# 3. One-vs-One (OvO) approach
ovo_model = OneVsOneClassifier(LogisticRegression(max_iter=1000, random_state=42))
ovo_accuracy, ovo_time, ovo_pred = train_and_evaluate(
    ovo_model, "One-vs-One (OvO)", X_train, y_train, X_test, y_test
)

# Compare predictions from different models
different_predictions = {}
for i, (true, std, ova, ovo) in enumerate(zip(y_test, standard_pred, ova_pred, ovo_pred)):
    if std != ova or std != ovo or ova != ovo:
        different_predictions[i] = {
            'True Class': true,
            'Standard Prediction': std,
            'OvA Prediction': ova,
            'OvO Prediction': ovo
        }

print("\nComparison of Models:")
print(f"Number of samples with different predictions: {len(different_predictions)}")

if different_predictions:
    df_diff = pd.DataFrame.from_dict(different_predictions, orient='index')
    print("\nSamples with different predictions:")
    print(df_diff.head(10))  # Show first 10 differences

# Compare performance metrics
performance_comparison = pd.DataFrame({
    'Model': ['Standard Multiclass', 'One-vs-All (OvA)', 'One-vs-One (OvO)'],
    'Accuracy': [standard_accuracy, ova_accuracy, ovo_accuracy],
    'Training Time (seconds)': [standard_time, ova_time, ovo_time]
})

print("\nPerformance Comparison:")
print(performance_comparison)

# Visualize performance comparison
plt.figure(figsize=(12, 6))

# Plot accuracy comparison
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=performance_comparison)
plt.title('Accuracy Comparison')
plt.ylim(0.9 * min(standard_accuracy, ova_accuracy, ovo_accuracy), 1.02)
plt.xticks(rotation=45)

# Plot training time comparison
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Training Time (seconds)', data=performance_comparison)
plt.title('Training Time Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

print("\nKey Differences between approaches:")
print("1. Standard Multiclass: Direct prediction of multi-class problems")
print("2. One-vs-All (OvA): Builds one binary classifier per class (class vs all other classes)")
print("3. One-vs-One (OvO): Builds binary classifiers for each pair of classes")
print("\nTheoretical differences:")
print("- OvO requires more classifiers (n*(n-1)/2 for n classes) than OvA (n classifiers)")
print("- OvO may be better for imbalanced datasets as each classifier sees a more balanced subset")
print("- OvA is generally faster for prediction but may struggle with class imbalance")
print("- Standard multiclass is usually the most efficient when the algorithm naturally supports multiclass")