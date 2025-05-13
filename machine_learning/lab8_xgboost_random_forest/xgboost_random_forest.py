# Comparative Analysis of Random Forest vs XGBoost Regression Models
# -------------------------------------------------------------
# This script compares the performance of Random Forest and XGBoost regression models
# on the California Housing dataset. The comparison includes:
#   1. Prediction accuracy (MSE and R² metrics)
#   2. Training and prediction time efficiency 
#   3. Visual comparison of prediction accuracy against actual values
# The goal is to highlight the strengths and trade-offs of these powerful ensemble methods.

# Import necessary libraries
import numpy as np              # For numerical operations and array handling
import matplotlib.pyplot as plt # For data visualization and creating comparison plots
from sklearn.datasets import fetch_california_housing  # Built-in housing price dataset
from sklearn.model_selection import train_test_split  # For creating training and testing datasets
from sklearn.ensemble import RandomForestRegressor    # Random Forest implementation
from xgboost import XGBRegressor                      # XGBoost implementation
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation
import time                     # For benchmarking model training and prediction times

# Load the California Housing dataset
# This dataset contains information about housing in California with target variable being median house value
data = fetch_california_housing()
X, y = data.data, data.target  # X: features matrix, y: target house values

# Feature descriptions in the California Housing dataset:
# MedInc: median income in block group
# HouseAge: median house age in block group
# AveRooms: average number of rooms per household
# AveBedrms: average number of bedrooms per household
# Population: block group population
# AveOccup: average number of household members
# Latitude: block group latitude
# Longitude: block group longitude

# Split data into training (80%) and test (20%) sets
# random_state ensures reproducibility of the data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize both regression models with identical parameters for fair comparison
n_estimators=100  # Number of trees in both ensemble models
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)  # Random Forest model
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)         # XGBoost model

# Training phase with performance timing
# --------------------------------------
# Random Forest training with timing
start_time_rf = time.time()
rf.fit(X_train, y_train)  # Train Random Forest model
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf  # Calculate training time in seconds

# XGBoost training with timing
start_time_xgb = time.time()
xgb.fit(X_train, y_train)  # Train XGBoost model
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb  # Calculate training time in seconds

# Prediction phase with performance timing
# -------------------------------------
# Random Forest prediction with timing
start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)  # Make predictions using Random Forest
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf  # Calculate prediction time in seconds

# XGBoost prediction with timing
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)  # Make predictions using XGBoost
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb  # Calculate prediction time in seconds

# Model evaluation using standard regression metrics
# -------------------------------------------------
# Mean Squared Error (MSE): Average of squared differences between predicted and actual values
# Lower MSE indicates better model performance
mse_rf = mean_squared_error(y_test, y_pred_rf)   # MSE for Random Forest
mse_xgb = mean_squared_error(y_test, y_pred_xgb) # MSE for XGBoost

# R-squared (R²): Proportion of variance in dependent variable explained by independent variables
# Higher R² (closer to 1) indicates better model performance
r2_rf = r2_score(y_test, y_pred_rf)   # R² for Random Forest
r2_xgb = r2_score(y_test, y_pred_xgb) # R² for XGBoost

# Display performance metrics
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')

# Calculate standard deviation of test target values
# Used for visualization of prediction error boundaries
std_y = np.std(y_test)

# NOTE: The following section duplicates the prediction and evaluation code
# This duplication is unnecessary and could be removed, as we've already:
# 1. Generated predictions using both models
# 2. Calculated MSE and R² metrics
# 3. Measured prediction times

# Duplicate prediction code - could be removed
start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf

# Duplicate XGBoost prediction code - could be removed
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb

# Duplicate metric calculations - could be removed
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Display performance metrics (duplicated output)
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')

# Display timing comparison between models
# Both training and prediction times are important factors when choosing a model
print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')

# Visual comparison of model predictions
# ------------------------------------
# Create a side-by-side comparison of how each model's predictions compare to actual values
plt.figure(figsize=(14, 6))  # Create a wide figure to accommodate two subplots

# Left subplot: Random Forest prediction visualization
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue", ec='k')
# Diagonal line representing perfect predictions (actual = predicted)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Model")
# Upper bound of standard deviation band
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
# Lower bound of standard deviation band
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1)
plt.ylim(0, 6)  # Set consistent y-axis limits for both plots
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.legend()

# Right subplot: XGBoost prediction visualization
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
# Scatter plot of actual vs. predicted values
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange", ec='k')
# Diagonal line representing perfect predictions (actual = predicted)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Perfect Model")
# Upper bound of standard deviation band
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
# Lower bound of standard deviation band
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1)
plt.ylim(0, 6)  # Set consistent y-axis limits for both plots
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual House Values")
plt.legend()

# Adjust spacing between subplots for better readability
plt.tight_layout()

# Display the visualization
# Notes on interpretation:
# - Points closer to the diagonal line indicate more accurate predictions
# - Points within the red dashed lines are within one standard deviation
# - The spread/dispersion of points shows the overall prediction error distribution
plt.show()

