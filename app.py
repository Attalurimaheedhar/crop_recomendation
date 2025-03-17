from flask import Flask, request, render_template
import pickle
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'crop_recommendation_model.pkl')
app = Flask(__name__)

def load_model():
    """Loads the trained ML model."""
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

model = load_model()

@app.route('/')
def home():
    return render_template('recomend.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form inputs and convert them to float
        nitrogen = float(request.form.get('nitrogen', 0))
        phosphorus = float(request.form.get('phosphorus', 0))
        potassium = float(request.form.get('potassium', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        ph = float(request.form.get('ph', 0))
        rainfall = float(request.form.get('rainfall', 0))

        # Validate input ranges (optional, adjust as needed)
        if not (0 <= nitrogen <= 200 and 0 <= phosphorus <= 200 and 
                0 <= potassium <= 200 and 0 <= temperature <= 70 and 
                0 <= humidity <= 100 and 0 <= ph <= 14 and 
                0 <= rainfall <= 4000):
            return render_template('recomend.html', prediction_text="Invalid input values.")

        # Prepare data for prediction
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Get prediction from the ML model
        prediction = model.predict(input_data)

        return render_template('recomend.html', prediction_text=f'Recommended Crop: {prediction[0]}')

    except Exception as e:
        return render_template('recomend.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=5000,debug=True)
