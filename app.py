from flask import Flask, render_template, request
import numpy as np
import pickle
import gzip

# Load model and LabelEncoder from compressed file
with gzip.open('model1.pkl', 'rb') as f:
    model, le = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Homepage route
@app.route('/')
def index():
    return render_template('crop4.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def recommend_crop():
    try:
        # Get form values and convert to float
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temprature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Format input as array for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict and decode label
        encoded_prediction = model.predict(input_data)[0]
        predicted_crop = le.inverse_transform([encoded_prediction])[0].capitalize()

        return render_template('crop4.html', result=predicted_crop)

    except Exception as e:
        return render_template('crop4.html', result=f"Error: {e}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
