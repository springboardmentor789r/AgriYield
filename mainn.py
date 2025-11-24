from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# --------------------------------------
# Load Models
# --------------------------------------
regression_model = joblib.load("regression_model.pkl")
ts_model = joblib.load("time_series_model.pkl")

# --------------------------------------
# Input Schema for Regression
# --------------------------------------
class RegressionInput(BaseModel):
    Crop_Type: str
    Soil_Type: str
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float

# --------------------------------------
# REGRESSION PREDICTION ENDPOINT
# --------------------------------------
@app.post("/predict/regression")
def predict_regression(data: RegressionInput):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Encoding based on your training
    crop_map = {
        "Wheat": 0, "Corn": 1, "Rice": 2, "Barley": 3, "Soybean": 4,
        "Cotton": 5, "Sugarcane": 6, "Tomato": 7, "Potato": 8, "Sunflower": 9
    }

    soil_map = {
        "Peaty": 0, "Loamy": 1, "Sandy": 2, "Saline": 3, "Clay": 4
    }

    df["Crop_Type"] = df["Crop_Type"].map(crop_map)
    df["Soil_Type"] = df["Soil_Type"].map(soil_map)

    # Validate missing encodings
    if df["Crop_Type"].isnull().any():
        return {"error": "Invalid Crop_Type entered"}
    if df["Soil_Type"].isnull().any():
        return {"error": "Invalid Soil_Type entered"}

    # Make Prediction
    prediction = regression_model.predict(df)[0]

    return {"Predicted_Crop_Yield": float(prediction)}

# --------------------------------------
# TIME-SERIES PREDICTION ENDPOINT
# --------------------------------------
@app.get("/predict/timeseries")
def predict_timeseries(days: int = 30):
    """
    Forecast the total crop yield for next 'days' days using ARIMA model.
    """
    forecast = ts_model.forecast(steps=days)
    forecast_list = forecast.tolist()

    return {
        "Forecast_Horizon_Days": days,
        "Predicted_Values": forecast_list
    }
