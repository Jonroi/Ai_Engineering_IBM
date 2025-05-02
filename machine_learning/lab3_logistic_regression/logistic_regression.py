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

    # Select relevant features and convert target to integer type
    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
    churn_df['churn'] = churn_df['churn'].astype('int')
    
    return churn_df


def extract_features(df):
    """
    Extract features and target variable from the dataframe.
    
    Args:
        df (pandas.DataFrame): The prepared churn dataframe
        
    Returns:
        tuple: (X, y) where X is the feature array and y is the target array
    """
    # Extract features (X) from the dataset
    X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    
    # Extract target variable (y) - whether the customer churned or not
    y = np.asarray(df['churn'])
    
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
    model = LogisticRegression().fit(X_train, y_train)
    return model


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


# Main execution
if __name__ == "__main__":
    # Load and prepare the dataset
    churn_df = load_data()
    
    # Extract features and target
    X, y = extract_features(churn_df)
    print("First 5 feature rows:")
    print(X[0:5])
    print("\nFirst 5 target values:")
    print(y[0:5])
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train model
    LR = train_model(X_train, y_train)
    
    # Make predictions
    yhat = LR.predict(X_test)
    print("\nFirst 10 predictions:")
    print(yhat[:10])
    
    # Visualize feature coefficients
    visualize_coefficients(LR, churn_df.columns[:-1])
    
    # Note: For model evaluation, you could add:
    # from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    # print("Accuracy:", accuracy_score(y_test, yhat))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, yhat))
    # print("Classification Report:\n", classification_report(y_test, yhat))
