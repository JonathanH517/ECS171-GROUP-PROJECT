from flask import Flask, request, render_template
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('tuned_RF.joblib')

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

            car_age = 2024 - model_year_input

            # Prepare the feature vector for prediction
            input_features = [[car_age, mileage_input, horsepower_input]]

            # Make the prediction
            prediction = model.predict(input_features)

            # Return the result
            return render_template('index.html', prediction_text=f'Predicted Car Price: ${prediction[0]:,.2f}')
        except ValueError:
            return render_template('index.html', prediction_text='Please enter valid numbers.')

if __name__ == "__main__":
    app.run(debug=True)
