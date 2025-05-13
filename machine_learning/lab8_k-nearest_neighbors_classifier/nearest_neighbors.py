# K-Nearest Neighbors Classification for Telecom Customer Categorization
# ----------------------------------------------------------------
# This script implements a K-Nearest Neighbors classifier to predict 
# customer categories from telecommunication customer data.
# The goal is to analyze which K value produces the best accuracy
# and to visualize the model's performance across different K values.

# Import necessary libraries
import numpy as np              # For numerical operations
import matplotlib.pyplot as plt # For data visualization
import pandas as pd             # For data manipulation
import seaborn as sns           # For enhanced visualizations
from sklearn.preprocessing import StandardScaler       # For feature scaling
from sklearn.model_selection import train_test_split   # For data splitting
from sklearn.neighbors import KNeighborsClassifier    # KNN algorithm
from sklearn.metrics import accuracy_score            # For model evaluation

# Load the telecommunications customer dataset
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
# Display the first few rows to understand the data structure
df.head()

# Check the distribution of customer categories
# custcat values: 1=Basic Service, 2=E-Service, 3=Plus Service, 4=Total Service
df['custcat'].value_counts()

# Create a correlation matrix to understand relationships between features
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Calculate the absolute correlation values with the target variable and sort them
# This helps identify which features have the strongest relationship with customer category
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
correlation_values

# Prepare features (X) and target variable (y)
X = df.drop('custcat', axis=1)  # All columns except the target
y = df['custcat']               # Customer category as the target

# Normalize the feature data using StandardScaler
# This ensures all features contribute equally to the distance calculations in KNN
X_norm = StandardScaler().fit_transform(X)

# Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Set the maximum number of neighbors (K) to test
Ks = 38  # We'll test K values from 1 to 37

# Initialize arrays to store accuracy values and their standard deviations
acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

# Loop through different K values to find the optimal K
for n in range(1, Ks):
    # Train the KNN model with current K value
    knn_model_n = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    
    # Make predictions on the training set
    # Note: For proper model evaluation, consider using X_test instead
    yhat = knn_model_n.predict(X_train)
    
    # Calculate accuracy for current K value
    acc[n-1] = accuracy_score(y_train, yhat)
    
    # Calculate standard deviation of the accuracy
    # This helps assess the stability of the model at this K value
    std_acc[n-1] = np.std(yhat==y_train)/np.sqrt(yhat.shape[0])

# Plot the accuracy for different K values
plt.plot(range(1,Ks), acc, 'g')
# Add a shaded area representing the standard deviation
plt.fill_between(range(1,Ks), acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

# Print the K value that gives the highest accuracy
print("The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 

# Note: In real-world applications, you should evaluate on the test set (X_test, y_test)
# instead of the training set to avoid overfitting.
# The optimal K value found above may be biased toward lower values because
# we're evaluating on the training data. Using test data would give a more
# reliable estimate of model performance.

