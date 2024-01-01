import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json
import shutil

# Load the model using MLflow
shared_model_path = "best_model/"
loaded_model = mlflow.sklearn.load_model(shared_model_path)

# Load data
data = pd.read_csv("preprocess_data.csv")

# Define features (X) and target variable (y)
X = data[['Hour', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

# Check if metrics difference exceeds the threshold
DIFF_THRESHOLD = 0.001
previous_metrics_file = 'current_metrics.json'
current_metric_file = 'new_metrics.json'

try:
    with open(previous_metrics_file, 'r') as json_file:
        previous_mse_dict = json.load(json_file)
        previous_mse = previous_mse_dict.get('mse', 0)
except (FileNotFoundError, json.JSONDecodeError):
    previous_mse = 0
    
    
try:
    with open(current_metric_file, 'r') as json_file:
        current_mse_dict = json.load(json_file)
        current_mse = current_mse_dict.get('mse', 0)
except (FileNotFoundError, json.JSONDecodeError):
    current_mse = 0

metrics_difference = current_mse - previous_mse

if metrics_difference > DIFF_THRESHOLD:
    print("Metrics difference exceeds the threshold. Retraining needed.")
    loaded_model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = loaded_model.predict(X_test)
    # Evaluate the new model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on the test set: {mse}")
    # Log MSE to a JSON file
    mse_dict = {'mse': mse}
    with open('current_metrics.json', 'w') as json_file:
        json.dump(mse_dict, json_file)
    
    # Log the retrained model using MLflow
    mlflow.sklearn.save_model(loaded_model, "new_model")
    
    # Source and destination paths
    source_folder = "new_model/"
    destination_folder = "best_model/"

    # Copy the contents of the source folder to the destination folder
    shutil.rmtree(destination_folder)  # Remove existing destination folder and its contents
    shutil.copytree(source_folder, destination_folder)
    shutil.rmtree(source_folder)
    
else:
    print("Metrics difference is within the threshold. No retraining needed.")
