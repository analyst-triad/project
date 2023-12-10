import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import sys

def train_and_evaluate(metric_file):
    # Load data
    data = pd.read_csv("preprocess_data.csv")

    # Define features (X) and target variable (y)
    X = data[['Hour', 'Machine_ID', 'Sensor_ID']]
    y = data['Reading']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Load the model using MLflow (replace "best_model/" with the actual path)
    loaded_model = mlflow.sklearn.load_model("best_model/")

    # Calculate Mean Squared Error
    y_pred = loaded_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Log MSE to the specified JSON file
    mse_dict = {'mse': mse}
    with open(metric_file, 'w') as json_file:
        json.dump(mse_dict, json_file)

if __name__ == "__main__":
    # Check if the correct number of command line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <metric_file>")
        sys.exit(1)

    # Get the metric file name from the command line arguments
    metric_file = sys.argv[1]

    # Call the train_and_evaluate function with the provided metric file name
    train_and_evaluate(metric_file)
