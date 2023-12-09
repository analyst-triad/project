import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Load data from CSV file
data = pd.read_csv('sensor_data.csv')

# Convert 'Timestamp' column to datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract 'Hour' from 'Timestamp' for potential use as a feature
data['Hour'] = data['Timestamp'].dt.hour

# Label encode 'Machine_ID' and 'Sensor_ID' columns for numeric representation
le_machine = LabelEncoder()
data['Machine_ID'] = le_machine.fit_transform(data['Machine_ID'])

le_sensor = LabelEncoder()
data['Sensor_ID'] = le_sensor.fit_transform(data['Sensor_ID'])

# Standardize numerical columns ('Hour', 'Machine_ID', 'Sensor_ID', 'Reading') using StandardScaler
scaler = StandardScaler()
data[['Hour', 'Machine_ID', 'Sensor_ID', 'Reading']] = scaler.fit_transform(data[['Hour', 'Machine_ID', 'Sensor_ID', 'Reading']])

data.to_csv("preprocess_data.csv")