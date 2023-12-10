import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import json

# Load data
data = pd.read_csv("preprocess_data.csv")

# Define features (X) and target variable (y)
X = data[['Hour', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Load the model using MLflow
shared_model_path = "best_model/"
loaded_model = mlflow.sklearn.load_model(shared_model_path)

# Calculate Mean Squared Error
y_pred = loaded_model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# Log MSE to a JSON file
mse_dict = {'mse': mse}
with open('current_metrics.json', 'w') as json_file:
    json.dump(mse_dict, json_file)
