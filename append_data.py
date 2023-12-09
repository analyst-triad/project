from random_data import generate_dummy_data
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import numpy as np
import dvc.api

# Function to generate dummy data and append to existing CSV
def generate_and_append_data(file_path, num_machines=5, num_sensors=3, freq='H'):
    try:
        existing_data = pd.read_csv(file_path)
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=['Timestamp', 'Machine_ID', 'Sensor_ID', 'Reading'])

    if not existing_data.empty and 'Timestamp' in existing_data.columns:
        existing_data['Timestamp'] = pd.to_datetime(existing_data['Timestamp'])
        start_date = existing_data['Timestamp'].max() + timedelta(hours=1)
    else:
        start_date = datetime.now()

    end_date = start_date + timedelta(days=1)

    new_data = generate_dummy_data(start_date, end_date, num_machines, num_sensors, freq)
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    updated_data.to_csv(file_path, index=False)


# DVC remote URL
REMOTE_URL = 'gdrive://1v0x1hF9Ta4TYldCm5qDLhWmlCD4zYwzW/sensor_data.csv'



if __name__ == "__main__":
    data_file_path = 'sensor_data.csv'
    generate_and_append_data(data_file_path)
    
    