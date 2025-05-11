from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained ANN model
model_path = 'ANNPassFailClassifierforSemiconductorManufacturingProcess.h5'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Required file not found: {model_path}")

model = tf.keras.models.load_model(model_path)

# Class label mapping (if needed, update accordingly)
class_labels = ['False', 'True']  # Modify if necessary

# Pydantic model to parse the request data
class PredictRequest(BaseModel):
    features: list

# Initialize templates for rendering HTML (equivalent to Flask's render_template)
templates = Jinja2Templates(directory="templates")

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Extract features from the incoming request data
        features = np.array(request.features)

        # Predict the output using the ANN model
        prediction = model.predict(features.reshape(1, -1))

        # Determine the predicted class (Pass/Fail)
        predicted_class = class_labels[np.argmax(prediction)]  # Convert prediction to a label
        
        return {"prediction": predicted_class}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # This will render the index.html file, like Flask's render_template
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit_data")
async def submit_data(request: PredictRequest):
    try:
        # Get the features from the incoming request
        received_data = request.features
        print("Received Data:", received_data)

        # Send the received data to the prediction route (reuse /predict logic)
        prediction_response = await predict(request)

        return prediction_response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
