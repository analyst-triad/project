from flask import Flask, render_template, request
import mlflow.pyfunc
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the model using MLflow
shared_model_path = "best_model/"
loaded_model = mlflow.pyfunc.load_model(shared_model_path)

# Define a route to handle file uploads and display predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Read the uploaded file into a DataFrame
            user_data = pd.read_csv(uploaded_file)
            
            user_data = user_data[['Hour', 'Machine_ID', 'Sensor_ID']]
            
            
            # Use the loaded model to make predictions
            predictions = loaded_model.predict(user_data)
            
            # Display the predictions
            result = pd.DataFrame({'Prediction': predictions})
            return render_template('result.html', result=result.to_html())
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
