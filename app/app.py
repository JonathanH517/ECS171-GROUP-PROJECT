from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.model_selection import GridSearchCV
import joblib

from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('tuned_RF.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html file in a folder named templates

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            model_year_input = int(request.form['modelYear'])
            mileage_input = int(request.form['mileage'])
            horsepower_input = int(request.form['horsepower'])


            # Prepare the feature vector for prediction
            input_features = [[model_year_input, mileage_input, horsepower_input]]
            scaled_input_features = scaler.fit_transform(input_features)
            # Make the prediction
            prediction = model.predict(input_features)

            # Return the result
            return render_template('index.html', prediction_text=f'Predicted Car Price: ${prediction[0]:,.2f}')
        except ValueError:
            return render_template('index.html', prediction_text='Please enter valid numbers.')

if __name__ == "__main__":
    app.run(debug=True)
