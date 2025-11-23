from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastapi.middleware.cors import CORSMiddleware  # Added for CORS

app = FastAPI(title="AgriYield Predictor", description="Predict crop yield using ML model")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model and preprocessors (fitted on full training data)
model = joblib.load("../models/random_forest_model.pkl")
crop_encoder = joblib.load("../models/crop_encoder.pkl")  # Load pre-fitted encoder
soil_encoder = joblib.load("../models/soil_encoder.pkl")  # Load pre-fitted encoder
scaler = joblib.load("../models/scaler.pkl")  # Load pre-fitted scaler

# Define input model (based on your dataset features)
class PredictionInput(BaseModel):
    crop_type: str  # e.g., "Wheat", "Corn"
    soil_type: str  # e.g., "Peaty", "Loamy"
    soil_pH: float
    temperature: float
    humidity: float
    wind_speed: float
    N: float  # Nitrogen
    P: float  # Phosphorus
    K: float  # Potassium

# Define output model
class PredictionOutput(BaseModel):
    predicted_yield: float

# No need to fit encoders/scaler here anymore

@app.post("/predict", response_model=PredictionOutput)
def predict_yield(input_data: PredictionInput):
    try:
        # Encode categoricals
        crop_encoded = crop_encoder.transform([input_data.crop_type])[0]
        soil_encoded = soil_encoder.transform([input_data.soil_type])[0]
        
        # Prepare numerical array and scale
        numericals = np.array([[input_data.soil_pH, input_data.temperature, input_data.humidity, 
                                input_data.wind_speed, input_data.N, input_data.P, input_data.K]])
        numericals_scaled = scaler.transform(numericals)
        
        # Combine features (order: crop_encoded, soil_encoded, soil_pH, temp, humidity, wind, N, P, K)
        features = np.array([[crop_encoded, soil_encoded, numericals_scaled[0][0], numericals_scaled[0][1], 
                              numericals_scaled[0][2], numericals_scaled[0][3], numericals_scaled[0][4], 
                              numericals_scaled[0][5], numericals_scaled[0][6]]])
        
        # Predict
        prediction = model.predict(features)[0]
        
        return PredictionOutput(predicted_yield=round(prediction, 2))
    except ValueError as e:  # For encoder/scaler issues
        raise HTTPException(status_code=400, detail=f"Invalid input value: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    return {"message": "AgriYield Predictor API is running"}