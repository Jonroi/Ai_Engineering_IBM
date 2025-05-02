"""
Fuel Consumption CO2 Emissions Linear Regression
================================================

This module demonstrates how to build a simple linear regression model
to predict CO2 emissions based on engine size and other vehicle features.

Dataset: 
    Fuel consumption and CO2 emissions data for various vehicle models
    
Model: 
    Linear regression for predicting CO2 emissions
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_data(url):
    """
    Load the fuel consumption dataset from a URL.
    
    Args:
        url (str): URL to the dataset CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    return pd.read_csv(url)


def explore_data(df):
    """
    Explore the dataset by displaying samples, statistics, and checking for missing values.
    
    Args:
        df (pandas.DataFrame): The dataset to explore
        
    Returns:
        None: Outputs information about the dataset
    """
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


def visualize_distribution(df, column, title=None):
    """
    Visualize the distribution of a specific column in the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset containing the column
        column (str): The name of the column to visualize
        title (str, optional): The title for the plot. Defaults to None.
        
    Returns:
        None: Displays the histogram plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(f'{column}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def explore_correlations(df, target_column):
    """
    Explore and visualize the correlations between features and the target variable.
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): The name of the target variable column
        
    Returns:
        pandas.Series: Sorted correlations with the target variable
    """
    print(f"\nCorrelation with {target_column}:")
    correlations = df.corr()[target_column].sort_values(ascending=False)
    print(correlations)
    
    # Visualize correlations with heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlations


def plot_feature_relationships(df, features, target):
    """
    Create scatter plots for multiple features vs the target variable.
    
    Args:
        df (pandas.DataFrame): The dataset containing the features and target
        features (list): List of feature column names to plot
        target (str): The name of the target column
        
    Returns:
        None: Displays the scatter plots
    """
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(x=feature, y=target, data=df)
        plt.title(f'{feature} vs {target}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Train a simple linear regression model.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred) - Model and data splits
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(model, X_test, y_test, y_pred, feature_names=None):
    """
    Evaluate and visualize the performance of a linear regression model.
    
    Args:
        model (LinearRegression): The trained model
        X_test (numpy.ndarray): Test feature matrix
        y_test (numpy.ndarray): True target values
        y_pred (numpy.ndarray): Predicted target values
        feature_names (list, optional): Names of the features. Defaults to None.
        
    Returns:
        tuple: (mse, r2) - Mean squared error and R-squared score
    """
    # Print model coefficients
    print(f"\nLinear Regression Model:")
    if feature_names and len(model.coef_) == len(feature_names):
        for feature, coef in zip(feature_names, model.coef_):
            print(f"Coefficient for {feature}: {coef:.4f}")
    else:
        print(f"Coefficient(s): {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2


def visualize_regression_results(X_test, y_test, y_pred, model, feature_name='Feature'):
    """
    Visualize the results of a simple linear regression model.
    
    Args:
        X_test (numpy.ndarray): Test feature matrix
        y_test (numpy.ndarray): True target values
        y_pred (numpy.ndarray): Predicted target values
        model (LinearRegression): The trained model
        feature_name (str, optional): Name of the feature. Defaults to 'Feature'.
        
    Returns:
        None: Displays the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
    plt.xlabel(feature_name)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.title(f'Linear Regression: Predicting CO2 Emissions from {feature_name}')
    plt.legend()

    # Add model equation to the plot
    if X_test.shape[1] == 1:  # Only for single-feature models
        equation = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes)

    # Show the plot
    plt.grid(True)
    plt.show()


# Main execution section
if __name__ == "__main__":
    # Load data directly from the URL
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    df = load_data(url)

    # Explore the dataset
    explore_data(df)

    # Visualize the distribution of CO2 emissions
    visualize_distribution(df, 'CO2EMISSIONS', 'Distribution of CO2 Emissions')

    # Explore correlations between features and target variable
    correlations = explore_correlations(df, 'CO2EMISSIONS')

    # Create scatter plots for key features vs CO2 emissions
    features_to_plot = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                       'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
    plot_feature_relationships(df, features_to_plot, 'CO2EMISSIONS')

    # Based on the exploration, ENGINESIZE has a strong correlation with CO2EMISSIONS
    # Continue with the linear regression model using ENGINESIZE as the predictor

    # Simple Linear Regression
    # -----------------------
    # Extract features and target variable
    X = df['ENGINESIZE'].values.reshape(-1, 1)  # Feature: Engine size
    y = df['CO2EMISSIONS'].values  # Target: CO2 emissions

    # Train the model and get predictions
    model, _, X_test, _, y_test, y_pred = train_linear_regression(X, y)

    # Evaluate the simple model
    mse, r2 = evaluate_model(model, X_test, y_test, y_pred)

    # Visualize the regression results
    visualize_regression_results(X_test, y_test, y_pred, model, 'Engine Size')

    # Multiple Linear Regression
    # -------------------------
    print("\nExploring multiple features for prediction...")
    features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']
    X_multi = df[features].values

    # Train the multiple feature model and get predictions
    model_multi, _, X_test_multi, _, y_test, y_pred_multi = train_linear_regression(X_multi, y)

    # Evaluate the multiple feature model
    mse_multi, r2_multi = evaluate_model(model_multi, X_test_multi, y_test, y_pred_multi, features)

    # Report improvement
    improvement = ((r2_multi - r2) / r2) * 100
    print(f"Improvement over single feature model: {improvement:.2f}%")