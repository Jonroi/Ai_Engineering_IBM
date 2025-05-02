#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CO2 Emissions Prediction with Multiple Linear Regression

This module demonstrates the application of simple and multiple linear regression
for predicting CO2 emissions of vehicles based on engine size and fuel consumption data.
The analysis includes model training, evaluation, comparison, and visualization.

Dataset: Fuel consumption and CO2 emissions for various vehicle models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_and_prepare_data():
    """
    Load fuel consumption dataset from URL and prepare it for analysis.
    
    The dataset contains model-specific fuel consumption ratings and CO2 emissions
    for vehicles available for retail sale in Canada.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test arrays for model training and testing
    """
    # Load data directly from the URL
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df = pd.read_csv(url)

    # Display a sample of the dataset for exploratory analysis
    print("Dataset sample (5 random rows):")
    print(df.sample(5))

    # Calculate and display feature correlations with target variable
    print("\nCorrelation with CO2EMISSIONS:")
    correlations = df.corr()['CO2EMISSIONS'].sort_values(ascending=False)
    print(correlations)

    # Extract features and target variable
    # We select ENGINESIZE and FUELCONSUMPTION_COMB_MPG as predictors based on domain knowledge
    X = df[['ENGINESIZE', 'FUELCONSUMPTION_COMB_MPG']].values  # Independent variables
    y = df['CO2EMISSIONS'].values  # Dependent variable (target)

    # Split the data into training (80%) and testing (20%) sets with fixed random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def exercise1_engine_size_regression(X_train, y_train):
    """
    Exercise 1: Simple Linear Regression using Engine Size
    
    Train a linear regression model to predict CO2 emissions based solely on engine size
    and visualize the relationship between these variables.
    
    Args:
        X_train (numpy.ndarray): Training data features array
        y_train (numpy.ndarray): Training data target values
        
    Returns:
        tuple: Trained regressor, coefficient, and intercept
    """
    print("\n----- Exercise 1: Linear Regression with Engine Size -----")
    
    # Extract engine size feature from the training data
    X_train_1 = X_train[:,0]
    
    # Initialize and train the linear regression model
    regressor_1 = linear_model.LinearRegression()
    regressor_1.fit(X_train_1.reshape(-1, 1), y_train)
    
    # Extract model parameters
    coef_1 = regressor_1.coef_
    intercept_1 = regressor_1.intercept_
    print('Coefficients: ', coef_1)
    print('Intercept: ', intercept_1)
    
    # Visualize training data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_1, y_train, color='blue', alpha=0.6, label='Training data')
    plt.plot(X_train_1, coef_1[0] * X_train_1 + intercept_1, '-r', 
             label=f'y = {coef_1[0]:.2f}x + {intercept_1:.2f}')
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.title("Training Data: Engine Size vs. CO2 Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return regressor_1, coef_1, intercept_1


def exercise2_fuel_consumption_regression(X_train, y_train):
    """
    Exercise 2: Simple Linear Regression using Fuel Consumption MPG
    
    Train a linear regression model to predict CO2 emissions based solely on fuel consumption
    and visualize the relationship between these variables.
    
    Args:
        X_train (numpy.ndarray): Training data features array
        y_train (numpy.ndarray): Training data target values
        
    Returns:
        tuple: Trained regressor, coefficient, and intercept
    """
    print("\n----- Exercise 2: Linear Regression with Fuel Consumption MPG -----")
    
    # Extract fuel consumption feature from the training data
    X_train_2 = X_train[:,1]
    
    # Initialize and train the linear regression model
    regressor_2 = linear_model.LinearRegression()
    regressor_2.fit(X_train_2.reshape(-1, 1), y_train)
    
    # Extract model parameters
    coef_2 = regressor_2.coef_
    intercept_2 = regressor_2.intercept_
    print('Coefficients: ', coef_2)
    print('Intercept: ', intercept_2)
    
    # Visualize training data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_2, y_train, color='blue', alpha=0.6, label='Training data')
    plt.plot(X_train_2, coef_2[0] * X_train_2 + intercept_2, '-r', 
             label=f'y = {coef_2[0]:.2f}x + {intercept_2:.2f}')
    plt.xlabel("Combined Fuel Consumption (MPG)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.title("Training Data: Fuel Consumption vs. CO2 Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return regressor_2, coef_2, intercept_2


def exercise3_test_engine_model(regressor_1, coef_1, intercept_1, X_test, y_test):
    """
    Exercise 3: Evaluate the Engine Size Model on Test Data
    
    Test the engine size regression model on unseen data and calculate performance metrics
    to evaluate its predictive capability.
    
    Args:
        regressor_1: Trained linear regression model for engine size
        coef_1: Model coefficient
        intercept_1: Model intercept
        X_test: Test data features
        y_test: Test data targets
        
    Returns:
        tuple: Mean squared error and R-squared score
    """
    print("\n----- Exercise 3: Testing the Model with Engine Size -----")
    
    # Extract engine size feature from test data
    X_test_1 = X_test[:,0]
    
    # Make predictions on test data
    y_pred_1 = regressor_1.predict(X_test_1.reshape(-1, 1))
    
    # Calculate performance metrics
    mse_1 = mean_squared_error(y_test, y_pred_1)
    r2_1 = r2_score(y_test, y_pred_1)
    
    # Visualize test data predictions against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_1, y_test, color='blue', alpha=0.6, label='Actual values')
    plt.plot(X_test_1, coef_1[0] * X_test_1 + intercept_1, '-r', 
             label=f'Predicted: y = {coef_1[0]:.2f}x + {intercept_1:.2f}')
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.title("Test Data: Engine Size vs. CO2 Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Display performance metrics
    print("Model 1 (Engine Size) Evaluation:")
    print(f"Mean Squared Error: {mse_1:.4f}")
    print(f"R² Score: {r2_1:.4f}")
    
    return mse_1, r2_1


def exercise4_test_fuel_model(regressor_2, coef_2, intercept_2, X_test, y_test):
    """
    Exercise 4: Evaluate the Fuel Consumption Model on Test Data
    
    Test the fuel consumption regression model on unseen data and calculate performance metrics
    to evaluate its predictive capability.
    
    Args:
        regressor_2: Trained linear regression model for fuel consumption
        coef_2: Model coefficient
        intercept_2: Model intercept
        X_test: Test data features
        y_test: Test data targets
        
    Returns:
        tuple: Mean squared error and R-squared score
    """
    print("\n----- Exercise 4: Testing the Model with Fuel Consumption MPG -----")
    
    # Extract fuel consumption feature from test data
    X_test_2 = X_test[:,1]
    
    # Make predictions on test data
    y_pred_2 = regressor_2.predict(X_test_2.reshape(-1, 1))
    
    # Calculate performance metrics
    mse_2 = mean_squared_error(y_test, y_pred_2)
    r2_2 = r2_score(y_test, y_pred_2)
    
    # Visualize test data predictions against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_2, y_test, color='blue', alpha=0.6, label='Actual values')
    plt.plot(X_test_2, coef_2[0] * X_test_2 + intercept_2, '-r', 
             label=f'Predicted: y = {coef_2[0]:.2f}x + {intercept_2:.2f}')
    plt.xlabel("Combined Fuel Consumption (MPG)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.title("Test Data: Fuel Consumption vs. CO2 Emissions")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Display performance metrics
    print("Model 2 (Fuel Consumption MPG) Evaluation:")
    print(f"Mean Squared Error: {mse_2:.4f}")
    print(f"R² Score: {r2_2:.4f}")
    
    return mse_2, r2_2


def exercise5_multiple_regression(X_train, X_test, y_train, y_test, r2_1, r2_2):
    """
    Exercise 5: Multiple Linear Regression with Both Features
    
    Train a multiple linear regression model using both engine size and fuel consumption
    to predict CO2 emissions, and compare its performance with single-feature models.
    
    Args:
        X_train: Training data features
        X_test: Test data features
        y_train: Training data targets
        y_test: Test data targets
        r2_1: R-squared score of engine size model
        r2_2: R-squared score of fuel consumption model
        
    Returns:
        tuple: Trained model, mean squared error, and R-squared score
    """
    print("\n----- Exercise 5: Multiple Linear Regression -----")
    
    # Initialize and train the multiple linear regression model
    regressor_multi = linear_model.LinearRegression()
    regressor_multi.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred_multi = regressor_multi.predict(X_test)
    
    # Display model parameters
    print('Multiple Regression Coefficients: ', regressor_multi.coef_)
    print('Multiple Regression Intercept: ', regressor_multi.intercept_)
    
    # Calculate performance metrics
    mse_multi = mean_squared_error(y_test, y_pred_multi)
    r2_multi = r2_score(y_test, y_pred_multi)
    
    # Display performance metrics
    print("Multiple Regression Model Evaluation:")
    print(f"Mean Squared Error: {mse_multi:.4f}")
    print(f"R² Score: {r2_multi:.4f}")
    
    # Compare models based on R-squared scores
    print("\n----- Model Comparison -----")
    print(f"Model 1 (Engine Size) R²: {r2_1:.4f}")
    print(f"Model 2 (Fuel Consumption MPG) R²: {r2_2:.4f}")
    print(f"Multiple Regression R²: {r2_multi:.4f}")
    
    # Calculate improvement percentage over best single-feature model
    best_single_r2 = max(r2_1, r2_2)
    improvement = ((r2_multi - best_single_r2) / best_single_r2) * 100
    print(f"Multiple regression improves over best single-feature model by: {improvement:.2f}%")
    
    # Create 3D visualization of the multiple regression model
    visualize_3d_regression(regressor_multi, X_test, y_test)
    
    return regressor_multi, mse_multi, r2_multi


def visualize_3d_regression(model, X_test, y_test):
    """
    Create a 3D visualization of the multiple regression plane and actual data points.
    
    This function creates a 3D scatter plot of test data points and overlays
    the regression plane determined by the multiple linear regression model.
    
    Args:
        model: Trained multiple linear regression model
        X_test: Test data features
        y_test: Test data targets
    
    Returns:
        None (displays the visualization)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for visualization
    x_min, x_max = X_test[:, 0].min(), X_test[:, 0].max()
    y_min, y_max = X_test[:, 1].min(), X_test[:, 1].max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    
    # Calculate z-values for the regression plane
    z_grid = model.intercept_ + model.coef_[0] * x_grid + model.coef_[1] * y_grid
    
    # Plot the regression plane with transparency
    surf = ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, cmap='viridis')
    
    # Plot the actual data points
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Engine Size (L)')
    ax.set_ylabel('Fuel Consumption (MPG)')
    ax.set_zlabel('CO2 Emissions (g/km)')
    ax.set_title('Multiple Linear Regression: 3D Visualization')
    
    # Add color bar to represent z-values
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the complete linear regression analysis workflow.
    
    This function orchestrates the entire process of data preparation, model training,
    evaluation, and visualization for both simple and multiple linear regression models.
    """
    # Load and prepare the dataset
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Exercise 1: Train and visualize simple linear regression with engine size
    regressor_1, coef_1, intercept_1 = exercise1_engine_size_regression(X_train, y_train)
    
    # Exercise 2: Train and visualize simple linear regression with fuel consumption
    regressor_2, coef_2, intercept_2 = exercise2_fuel_consumption_regression(X_train, y_train)
    
    # Exercise 3: Evaluate engine size model on test data
    mse_1, r2_1 = exercise3_test_engine_model(regressor_1, coef_1, intercept_1, X_test, y_test)
    
    # Exercise 4: Evaluate fuel consumption model on test data
    mse_2, r2_2 = exercise4_test_fuel_model(regressor_2, coef_2, intercept_2, X_test, y_test)
    
    # Exercise 5: Train, evaluate, and visualize multiple linear regression model
    regressor_multi, mse_multi, r2_multi = exercise5_multiple_regression(
        X_train, X_test, y_train, y_test, r2_1, r2_2)


# Execute the main function when the script is run directly
if __name__ == "__main__":
    main()