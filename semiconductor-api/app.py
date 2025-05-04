from flask import Flask, request, jsonify, render_template 
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the pre-trained ANN model
model_path = 'ANNPassFailClassifierforSemiconductorManufacturingProcess.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Required file not found: {model_path}")

model = tf.keras.models.load_model(model_path)

# Class label mapping (if needed, update accordingly)
class_labels = ['False', 'True']  # Modify if necessary

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data
        input_data = request.get_json()
        
        # Extract features from the incoming request data
        features = np.array(input_data['features'])

        # Predict the output using the ANN model
        prediction = model.predict(features.reshape(1, -1))

        # Determine the predicted class (Pass/Fail)
        predicted_class = class_labels[np.argmax(prediction)]  # Convert prediction to a label
        
        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_data', methods=['POST'])
def submit_data():
    # Get the JSON data sent from the frontend
    data = request.get_json()
    received_data = data.get('data', [])
    print("Received Data:", received_data)

    # Send the received data to the prediction route
    prediction_response = predict()

    return jsonify(prediction_response)

if __name__ == '__main__':
    app.run(debug=True)
