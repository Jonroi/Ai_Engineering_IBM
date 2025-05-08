from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

# read the input data
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
print("Dataset loaded successfully. Sample rows:")
print(raw_data.head())

# Calculate correlations with tip_amount
print("\nCalculating correlations with tip_amount...")
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')

# Find the top 3 features with highest absolute correlation to tip_amount
top_3_features = abs(correlation_values).sort_values(ascending=False)[:3]
print("\nTop 3 features with the highest correlation to tip_amount:")
print(top_3_features)

# Create a more visually appealing bar plot for all correlations
plt.figure(figsize=(10, 6))
correlation_values.sort_values().plot(kind='barh')
plt.title('Correlation of All Features with Tip Amount')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('all_correlations_with_tip.png')
plt.show()

# Create a focused bar plot for just the top 3 correlations
plt.figure(figsize=(8, 4))
top_3_features.sort_values(ascending=True).plot(kind='barh', color='green')
plt.title('Top 3 Features with Highest Correlation to Tip Amount')
plt.xlabel('Absolute Correlation Coefficient')
plt.tight_layout()
plt.savefig('top3_correlations_with_tip.png')
plt.show()

# Create scatter plots for the top 3 correlated features
plt.figure(figsize=(15, 4))
for i, feature in enumerate(top_3_features.index, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=feature, y='tip_amount', data=raw_data, alpha=0.5)
    plt.title(f"{feature} vs tip_amount")
plt.tight_layout()
plt.savefig('top3_feature_scatter_plots.png')
plt.show()

# extract the labels from the dataframe
y = raw_data[['tip_amount']].values.astype('float32')

# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)

# get the feature matrix used for training
X = proc_data.values

# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# import the Decision Tree Regression Model from scikit-learn
from sklearn.tree import DecisionTreeRegressor

# for reproducible output across multiple function calls, set random_state to a given integer value
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',
                               max_depth=8, 
                               random_state=35)

dt_reg.fit(X_train, y_train)

# run inference using the sklearn model
y_pred = dt_reg.predict(X_test)

# evaluate mean squared error on the test dataset
mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))

# Extract feature importance from the decision tree
feature_names = proc_data.columns
feature_importance = pd.Series(dt_reg.feature_importances_, index=feature_names)
top_features = feature_importance.sort_values(ascending=False)[:5]

print("\nTop 5 most important features for predicting tip amount (from Decision Tree):")
print(top_features)

# Visualize feature importance
plt.figure(figsize=(10, 6))
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance for Tip Amount Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance_for_tip.png')
plt.show()

print("\nAnalysis Summary:")
print("1. Top features by correlation coefficient: {}".format(", ".join(top_3_features.index)))
print("2. Top feature by decision tree importance: {}".format(top_features.index[0]))
print("3. Visualizations saved as PNG files in the current directory")
