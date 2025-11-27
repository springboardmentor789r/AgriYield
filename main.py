from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI()

# --------------------------------------------------
# CORS (Allow React Frontend)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
regression_model = joblib.load("regression_model.pkl")
ts_model = joblib.load("time_series_model.pkl")

# --------------------------------------------------
# Input Schema for Regression Model
# --------------------------------------------------
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

# --------------------------------------------------
# REGRESSION ENDPOINT
# --------------------------------------------------
@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    df = pd.DataFrame([data.dict()])

    # Encoding maps
    crop_map = {
        "Wheat": 0, "Corn": 1, "Rice": 2, "Barley": 3, "Soybean": 4,
        "Cotton": 5, "Sugarcane": 6, "Tomato": 7, "Potato": 8, "Sunflower": 9
    }

    soil_map = {
        "Peaty": 0, "Loamy": 1, "Sandy": 2, "Saline": 3, "Clay": 4
    }

    df["Crop_Type"] = df["Crop_Type"].map(crop_map)
    df["Soil_Type"] = df["Soil_Type"].map(soil_map)

    if df["Crop_Type"].isnull().any():
        return {"error": "Invalid Crop Type"}
    if df["Soil_Type"].isnull().any():
        return {"error": "Invalid Soil Type"}

    prediction = regression_model.predict(df)[0]

    return {"Predicted_Yield": float(prediction)}

# --------------------------------------------------
# TIME-SERIES (WITH DATE RANGE)
# --------------------------------------------------
@app.get("/predict/timeseries")
def predict_timeseries(start_date: str, end_date: str):

    # Parse dates
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except:
        return {"error": "Date format must be YYYY-MM-DD"}

    # Calculate number of days
    days = (end - start).days

    if days <= 0:
        return {"error": "Invalid date range"}

    forecast = ts_model.forecast(steps=days)

    return {
        "Start_Date": start_date,
        "End_Date": end_date,
        "Total_Days": days,
        "Predicted_Values": forecast.tolist()
    }

# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "Backend is running successfully!"}
Values": forecast_list
    }

