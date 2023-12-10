import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load data
data = pd.read_csv("preprocess_data.csv")

# Define features (X) and target variable (y)
X = data[['Hour', 'Machine_ID', 'Sensor_ID']]
y = data['Reading']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define lists of hyperparameters to tune
n_estimators_list = [50, 100, 150]
max_depth_list = [5, 10, 15]

# Iterate over hyperparameter combinations
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Build the RandomForestRegressor with the current hyperparameters
            rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            # Fit the model
            rf_model.fit(X_train, y_train)

            # Make predictions
            predictions = rf_model.predict(X_test)

            # Log metrics
            mse = mean_squared_error(y_test, predictions)
            mlflow.log_metric("mse", mse)

            # Log the model
            mlflow.sklearn.log_model(rf_model, "random_forest_model")
