This project showcases a comprehensive end-to-end MLOps pipeline for predictive maintenance, encompassing data collection, preprocessing, model training, deployment, and concept drift monitoring.

## Overview

The project focuses on creating a robust system for predictive maintenance using simulated sensor data from industrial machines. It demonstrates the integration of various tools and methodologies to streamline the development and deployment of machine learning models 
for predictive maintenance.

## Steps to Run

### 1. Append Data and run locally

To generate and append data:

```bash
python append_data.py
# To preprocess the data:
python preprocess.py
# To train the model:
python train.py
# To run the Flask app for real-time predictions:
python app.py
```
### 2. To run using Docker Conatiner
- Pull the docker Image:
```bash
docker pull analysts/project_flask_app:latest
```
- Run the Container:
```bash
docker run -p 5001:5001 analysts/project_flask_app:latest
```
