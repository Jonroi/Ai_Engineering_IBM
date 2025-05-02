"""
Customer Churn Prediction using Logistic Regression
===================================================

This module demonstrates how to build a logistic regression model
to predict customer churn based on various features.

Dataset: 
    Customer churn data including tenure, age, income and other features
    
Model: 
    Binary classification using Logistic Regression
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def load_data():
    """
    Load and prepare the customer churn dataset.
    
    Returns:
        pandas.DataFrame: Prepared dataframe with selected features 
                          and converted target variable
    """
    # churn_df = pd.read_csv("ChurnData.csv")
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
    churn_df = pd.read_csv(url)
    
    return churn_df


def extract_features(df, features, target_col='churn'):
    """
    Extract features and target variable from the dataframe.
    
    Args:
        df (pandas.DataFrame): The prepared churn dataframe
        features (list): List of feature column names to use
        target_col (str, optional): Name of the target column. Defaults to 'churn'.
        
    Returns:
        tuple: (X, y) where X is the feature array and y is the target array
    """
    # Extract features (X) from the dataset
    X = np.asarray(df[features])
    
    # Extract target variable (y) - whether the customer churned or not
    y = np.asarray(df[target_col])
    
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=4):
    """
    Preprocess the data by normalizing features and splitting into train/test sets.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 4.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and testing sets
    """
    # Normalize the feature data using StandardScaler to improve model performance
    X_norm = StandardScaler().fit(X).transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a logistic regression model on the given data.
    
    Args:
        X_train (numpy.ndarray): Training feature matrix
        y_train (numpy.ndarray): Training target vector
        
    Returns:
        LogisticRegression: Trained logistic regression model
    """
    model = LogisticRegression(C=0.01).fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using log loss metric.
    
    Args:
        model (LogisticRegression): Trained logistic regression model
        X_test (numpy.ndarray): Test feature matrix
        y_test (numpy.ndarray): Test target vector
        
    Returns:
        float: Log loss value
    """
    yhat = model.predict(X_test)
    yhat_prob = model.predict_proba(X_test)
    logloss = log_loss(y_test, yhat_prob)
    
    return logloss, yhat


def visualize_coefficients(model, feature_names):
    """
    Visualize the coefficients of the logistic regression model.
    
    Args:
        model (LogisticRegression): Trained logistic regression model
        feature_names (list): Names of the features
        
    Returns:
        None: Displays the plot
    """
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    coefficients.sort_values().plot(kind='barh')
    plt.title("Feature Coefficients in Logistic Regression Churn Model")
    plt.xlabel("Coefficient Value")
    plt.show()


def run_experiment(df, features, title):
    """
    Run a logistic regression experiment with the given features.
    
    Args:
        df (pandas.DataFrame): The dataset
        features (list): List of feature column names to use
        title (str): Title of the experiment
        
    Returns:
        float: Log loss value
    """
    print(f"\n{title}")
    print(f"Features used: {features}")
    
    # Extract features and target
    X, y = extract_features(df, features)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    logloss, yhat = evaluate_model(model, X_test, y_test)
    print(f"Log Loss: {logloss}")
    
    return logloss


# Main execution
if __name__ == "__main__":
    # Load the dataset
    df = load_data()
    
    # Ensure 'churn' is integer type
    df['churn'] = df['churn'].astype('int')
    
    # Original features
    original_features = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
    
    # Original model
    run_experiment(df, original_features, "Original Model")
    
    # Practice Exercises
    print("\n\nPractice Exercises")
    print("=================\n")
    
    # a. Add 'callcard' feature
    features_a = original_features + ['callcard']
    logloss_a = run_experiment(df, features_a, "a. Add 'callcard' feature")
    
    # b. Add 'wireless' feature
    features_b = original_features + ['wireless']
    logloss_b = run_experiment(df, features_b, "b. Add 'wireless' feature")
    
    # c. Add both 'callcard' and 'wireless' features
    features_c = original_features + ['callcard', 'wireless']
    logloss_c = run_experiment(df, features_c, "c. Add 'callcard' and 'wireless' features")
    
    # d. Remove 'equip' feature
    features_d = [f for f in original_features if f != 'equip']
    logloss_d = run_experiment(df, features_d, "d. Remove 'equip' feature")
    
    # e. Remove 'income' and 'employ' features
    features_e = [f for f in original_features if f not in ['income', 'employ']]
    logloss_e = run_experiment(df, features_e, "e. Remove 'income' and 'employ' features")
    
    # Summary of results
    print("\nSummary of Results")
    print("=================")
    print(f"Original model log loss: {logloss_a}")
    print(f"a. Add 'callcard' feature: {logloss_a}")
    print(f"b. Add 'wireless' feature: {logloss_b}")
    print(f"c. Add both features: {logloss_c}")
    print(f"d. Remove 'equip' feature: {logloss_d}")
    print(f"e. Remove 'income' and 'employ' features: {logloss_e}")
