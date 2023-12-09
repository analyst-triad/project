import pandas as pd
from datetime import datetime, timedelta
import subprocess
import numpy as np
import dvc.api

# Set seed for reproducibility
np.random.seed(42)

REMOTE_URL = 'gdrive://1v0x1hF9Ta4TYldCm5qDLhWmlCD4zYwzW/sensor_data.csv'

# Function to generate random datetime within a given range
def random_dates(start_date, end_date, n=10):
    date_range = (end_date - start_date).days
    random_dates = [start_date + timedelta(days=np.random.randint(date_range)) for _ in range(n)]
    return sorted(random_dates)

# Function to generate dummy data
def generate_dummy_data(start_date, end_date, num_machines=5, num_sensors=3, freq='H'):
    machine_ids = [f'Machine_{i}' for i in range(1, num_machines + 1)]
    sensor_ids = [f'Sensor_{j}' for j in range(1, num_sensors + 1)]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    data = {'Timestamp': [], 'Machine_ID': [], 'Sensor_ID': [], 'Reading': []}

    for date in dates:
        for machine_id in machine_ids:
            for sensor_id in sensor_ids:
                data['Timestamp'].append(date)
                data['Machine_ID'].append(machine_id)
                data['Sensor_ID'].append(sensor_id)
                # Simulate sensor readings as random values
                data['Reading'].append(np.random.normal(loc=100, scale=20))

    return pd.DataFrame(data)


if __name__ == "__main__":
    data_file_path = 'sensor_data.csv'
    # Define date range for dummy data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)

    # Generate dummy data
    dummy_data = generate_dummy_data(start_date, end_date, num_machines=5, num_sensors=3)

    # Save dummy data to CSV file
    dummy_data.to_csv('sensor_data.csv', index=False)
