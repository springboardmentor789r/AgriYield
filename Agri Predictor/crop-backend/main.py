from datetime import date
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------
app = FastAPI(title="Crop Yield Forecast API")

# Allow your React frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# LOAD PROPHET MODEL
# --------------------------------------------------------
model = joblib.load("Prophet.pkl")

# EXACT COLUMN NAMES USED DURING TRAINING
REGRESSORS = [
    "Soil_pH",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "N",
    "P",
    "K",
    "Soil_Quality"
]

# --------------------------------------------------------
# REQUEST MODEL (VERY IMPORTANT: EXACT SAME NAMES)
# --------------------------------------------------------
class YieldRequest(BaseModel):
    ds: date
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float
    actual_yield: Optional[float] = None


# --------------------------------------------------------
# RESPONSE MODEL
# --------------------------------------------------------
class YieldResponse(BaseModel):
    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float
    abs_error: Optional[float] = None
    mape: Optional[float] = None


# --------------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------------
@app.post("/predict", response_model=YieldResponse)
def predict_yield(payload: YieldRequest):
    """
    Predict yield using Prophet + regressors.
    """

    # Build a single-row DataFrame with EXACT column names
    row = pd.DataFrame([{
        "ds": payload.ds,
        "Soil_pH": payload.Soil_pH,
        "Temperature": payload.Temperature,
        "Humidity": payload.Humidity,
        "Wind_Speed": payload.Wind_Speed,
        "N": payload.N,
        "P": payload.P,
        "K": payload.K,
        "Soil_Quality": payload.Soil_Quality
    }])

    # Prophet prediction
    forecast = model.predict(row)
    pred = forecast.iloc[0]

    yhat = float(pred["yhat"])
    yhat_lower = float(pred["yhat_lower"])
    yhat_upper = float(pred["yhat_upper"])

    # Optional real-time validation
    abs_error = None
    mape = None

    if payload.actual_yield is not None:
        abs_error = float(payload.actual_yield - yhat)
        if payload.actual_yield != 0:
            mape = float(abs(abs_error) / abs(payload.actual_yield) * 100)

    return YieldResponse(
        ds=payload.ds,
        yhat=yhat,
        yhat_lower=yhat_lower,
        yhat_upper=yhat_upper,
        abs_error=abs_error,
        mape=mape
    )
