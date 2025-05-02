# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data directly from the URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Display a sample of the dataset
print("Dataset sample (5 random rows):")
print(df.sample(5))

# Explore the data with statistical summary
print("\nStatistical summary of the data:")
print(df.describe())

# Check the data types and missing values
print("\nData types and non-null counts:")
print(df.info())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualize the distribution of CO2 emissions
plt.figure(figsize=(10, 6))
sns.histplot(df['CO2EMISSIONS'], kde=True)
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Explore correlations between features and target variable
print("\nCorrelation with CO2EMISSIONS:")
correlations = df.corr()['CO2EMISSIONS'].sort_values(ascending=False)
print(correlations)

# Visualize correlations with heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Create scatter plots for key features vs CO2 emissions
features_to_plot = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                   'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=feature, y='CO2EMISSIONS', data=df)
    plt.title(f'{feature} vs CO2EMISSIONS')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Based on the exploration, ENGINESIZE has a strong correlation with CO2EMISSIONS
# Continue with the linear regression model using ENGINESIZE as the predictor

# Extract features and target variable
X = df['ENGINESIZE'].values.reshape(-1, 1)  # Feature: Engine size
y = df['CO2EMISSIONS'].values  # Target: CO2 emissions

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model coefficients
print(f"\nLinear Regression Model:")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Linear Regression: Predicting CO2 Emissions from Engine Size')
plt.legend()

# Add model equation to the plot
equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes)

# Show the plot
plt.grid(True)
plt.show()

# Calculate and print the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Try multiple features to see if we can improve the model
print("\nExploring multiple features for prediction...")
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']
X_multi = df[features].values
X_train_multi, X_test_multi, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Create and train the multiple linear regression model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train)

# Make predictions with the multiple feature model
y_pred_multi = model_multi.predict(X_test_multi)

# Calculate and print the model performance for multiple features
mse_multi = mean_squared_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)
print(f"\nMultiple Feature Model Performance:")
print(f"Features used: {features}")
print(f"Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.4f}")
print(f"Mean Squared Error: {mse_multi:.4f}")
print(f"R² Score: {r2_multi:.4f}")
print(f"Improvement over single feature model: {((r2_multi - r2) / r2) * 100:.2f}%")