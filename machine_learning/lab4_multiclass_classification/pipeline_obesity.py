import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

def obesity_risk_pipeline(data_path, test_size=0.2):
    # Load data
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{data_path}'")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    
    # Converting to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    
    # Combining with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column
    
    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    
    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Preparing final dataset
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']
   
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    try:
        # Training and evaluation
        model = LogisticRegression(multi_class='multinomial', max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        
        # Return model and data for further use
        return model, X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in training or evaluation: {e}")
        return None, None, None, None, None

# Execute the function when the script is run directly
if __name__ == "__main__":
    # Use the URL for the obesity dataset (found in other scripts)
    data_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
    
    print(f"Downloading obesity dataset from: {data_path}")
    
    # Run the pipeline
    model, X_train, X_test, y_train, y_test = obesity_risk_pipeline(data_path)
    
    # Save the trained model if successful
    if model is not None:
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'obesity_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model successfully saved to '{model_path}'")
            
            # Also save to the current directory for convenience
            current_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'obesity_model.pkl')
            with open(current_model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model also saved to '{current_model_path}'")
        except Exception as e:
            print(f"Error saving model: {e}")