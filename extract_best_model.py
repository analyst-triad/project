import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import json
import shutil

# Set the experiment name and model name
experiment_name = "Default"  # Replace with your actual experiment name
model_name = "production_model"  # Replace with your desired model name

# Get the experiment ID
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)

if experiment:
    experiment_id = experiment.experiment_id
else:
    print(f"Experiment '{experiment_name}' not found.")
    # Handle the case where the experiment doesn't exist or has no runs
    # You might want to create the experiment or take appropriate action
    exit()

# Search for runs in the experiment
runs = client.search_runs(experiment_ids=[experiment_id], filter_string="", order_by=["metrics.mse ASC"])

# Iterate through runs to find the best one
best_run = None
best_mse = float('inf')  # Initialize with a large value

for run in runs:
    mse = run.data.metrics.get("mse")
    if mse is not None and mse < best_mse:
        best_mse = mse
        best_run = run

# Check if any runs were found
if best_run is not None:
    print(f"Best Run ID: {best_run.info.run_id}")
    print(f"Best MSE: {best_mse}")

    # Retrieve the best model URI
    best_model_uri = best_run.info.artifact_uri + "/random_forest_model"

    try:
        # Register the best model in the Model Registry
        mlflow.register_model("runs:/" + best_run.info.run_id + "/random_forest_model", model_name)
        print(f"Best model registered as '{model_name}' in the Model Registry.")
    except MlflowException as e:
        print(f"Error registering the best model: {e}")

else:
    print("No runs found in the experiment.")


# Save the best model locally
local_model_path = "best_model"
mlflow.sklearn.save_model(best_run.info.artifact_uri + "/random_forest_model", local_model_path)

