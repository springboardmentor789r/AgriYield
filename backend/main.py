# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd

# app = FastAPI(
#     title="Crop Yield Prediction API",
#     description="API for Random Forest Crop Prediction & Prophet Time-Series Forecasting",
#     version="2.0"
# )

# # ----------------------------------------------------
# # LOAD TRAINED MODELS
# # ----------------------------------------------------
# rf_model = joblib.load("SavedModel.pkl")                 # Full RF pipeline
# feature_order = joblib.load("featuresaved.pkl")          # For RF model

# prophet_model = joblib.load("prophet_model.pkl")        # NEW Prophet model
# prophet_scaler = joblib.load("prophet_scaler.pkl")       # NEW scaler

# # Prophet regressors
# prophet_regressors = [
#     "Temperature", "Humidity", "Wind_Speed", "Soil_pH",
#     "N", "P", "K", "Soil_Quality"
# ]

# # REQUEST MODELS

# class CropInput(BaseModel):
#     Crop_Type: str
#     Soil_Type: str
#     Soil_pH: float
#     Wind_Speed: float
#     Temperature: float
#     N: float
#     P: float
#     K: float
#     Soil_Quality: float
#     Humidity: float


# class ProphetNextNDays(BaseModel):
#     days: int
#     Temperature: float
#     Humidity: float
#     Wind_Speed: float
#     Soil_pH: float
#     N: float
#     P: float
#     K: float
#     Soil_Quality: float


# class ProphetDateRange(BaseModel):
#     start_date: str
#     end_date: str
#     Temperature: float
#     Humidity: float
#     Wind_Speed: float
#     Soil_pH: float
#     N: float
#     P: float
#     K: float
#     Soil_Quality: float


# # ENDPOINT 1 — RANDOM FOREST PREDICTION
# @app.post("/predict_rf")
# def predict_rf(data: CropInput):
    
#     input_df = pd.DataFrame([data.dict()])

#     # reorder columns
#     input_df = input_df[feature_order]

#     prediction = rf_model.predict(input_df)[0]

#     return {"Predicted_Crop_Yield (tonnes/ha)": float(prediction)}


# # ENDPOINT 2A — PROPHET FORECAST FOR NEXT N DAYS
# @app.post("/forecast_next_days")
# def forecast_next_days(req: ProphetNextNDays):

#     reg = {
#         "Temperature": req.Temperature,
#         "Humidity": req.Humidity,
#         "Wind_Speed": req.Wind_Speed,
#         "Soil_pH": req.Soil_pH,
#         "N": req.N,
#         "P": req.P,
#         "K": req.K,
#         "Soil_Quality": req.Soil_Quality
#     }

#     # Scale regressors → returns numpy array
#     reg_scaled = prophet_scaler.transform(pd.DataFrame([reg]))

#     # Convert scaled array back to DataFrame with column names
#     reg_scaled_df = pd.DataFrame(reg_scaled, columns=prophet_regressors)

#     # Create future DF
#     future = prophet_model.make_future_dataframe(periods=req.days)

#     # Assign same scaled regressor values for all future rows
#     for col in prophet_regressors:
#         future[col] = reg_scaled_df[col].iloc[0]

#     forecast = prophet_model.predict(future).tail(req.days)

#     avg_yield = forecast["yhat"].mean()

#     return {
#         "daily_predicted_yield": forecast[["ds", "yhat"]].to_dict(orient="records"),
#         "average_predicted_yield": round(float(avg_yield), 3)
#     }



# # ENDPOINT 2B — PROPHET FORECAST FOR DATE RANGE
# @app.post("/forecast_range")
# def forecast_range(req: ProphetDateRange):

#     date_range = pd.date_range(req.start_date, req.end_date)

#     reg = {
#         "Temperature": req.Temperature,
#         "Humidity": req.Humidity,
#         "Wind_Speed": req.Wind_Speed,
#         "Soil_pH": req.Soil_pH,
#         "N": req.N,
#         "P": req.P,
#         "K": req.K,
#         "Soil_Quality": req.Soil_Quality
#     }

#     # Scale values → numpy array
#     reg_scaled = prophet_scaler.transform(pd.DataFrame([reg]))

#     # Convert back to DataFrame
#     reg_scaled_df = pd.DataFrame(reg_scaled, columns=prophet_regressors)

#     future = pd.DataFrame({"ds": date_range})

#     # Assign scaled values
#     for col in prophet_regressors:
#         future[col] = reg_scaled_df[col].iloc[0]

#     forecast = prophet_model.predict(future)

#     avg_yield = forecast["yhat"].mean()

#     return {
#         "dates_and_yield": forecast[["ds", "yhat"]].to_dict(orient="records"),
#         "average_predicted_yield": round(float(avg_yield), 3)
#     }


# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import traceback
from typing import List

app = FastAPI(title="AgriYield API", version="1.0",
              description="RandomForest prediction + Prophet time-series forecasting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or replace "*" with frontend URL: "http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------
# Load models (make sure files exist)
# ---------------------------
try:
    rf_model = joblib.load("SavedModel.pkl")
    feature_order = joblib.load("featuresaved.pkl")
    prophet_model = joblib.load("prophet_model.pkl")
    prophet_scaler = joblib.load("prophet_scaler.pkl")
except Exception as e:
    # If models are not found or invalid, raise on startup.
    raise RuntimeError("Failed loading models: " + str(e))

# Prophet regressors used at training (order must match scaler)
prophet_regressors = [
    "Temperature", "Humidity", "Wind_Speed", "Soil_pH",
    "N", "P", "K", "Soil_Quality"
]

# ---------------------------
# Request schemas
# ---------------------------
class CropInput(BaseModel):
    Crop_Type: str = Field(..., example="Rice")
    Soil_Type: str = Field(..., example="Loamy")
    Soil_pH: float = Field(..., example=6.5)
    Wind_Speed: float = Field(..., example=3.5)
    Temperature: float = Field(..., example=28.0)
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=40.0)
    K: float = Field(..., example=35.0)
    Soil_Quality: float = Field(..., example=8.0)
    Humidity: float = Field(..., example=65.0)

class ProphetNextNDays(BaseModel):
    days: int = Field(..., example=7, gt=0)
    Temperature: float = Field(..., example=28.0)
    Humidity: float = Field(..., example=65.0)
    Wind_Speed: float = Field(..., example=3.5)
    Soil_pH: float = Field(..., example=6.5)
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=40.0)
    K: float = Field(..., example=35.0)
    Soil_Quality: float = Field(..., example=8.0)

class ProphetDateRange(BaseModel):
    start_date: str = Field(..., example="2026-01-01")
    end_date: str = Field(..., example="2026-01-10")
    Temperature: float = Field(..., example=28.0)
    Humidity: float = Field(..., example=65.0)
    Wind_Speed: float = Field(..., example=3.5)
    Soil_pH: float = Field(..., example=6.5)
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=40.0)
    K: float = Field(..., example=35.0)
    Soil_Quality: float = Field(..., example=8.0)

# ---------------------------
# Helper function
# ---------------------------
def _scale_regressors_single(reg_dict):
    """
    Accepts a dict of regressors and returns a DataFrame with scaled values (single row).
    """
    df_in = pd.DataFrame([reg_dict])
    scaled = prophet_scaler.transform(df_in)  # returns numpy array
    scaled_df = pd.DataFrame(scaled, columns=prophet_regressors)
    return scaled_df.iloc[0]  # pandas Series

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def root():
    return {"message": "AgriYield API running."}


@app.post("/predict_rf")
def predict_rf(data: CropInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        # reorder to training feature order
        input_df = input_df[feature_order]
        pred = rf_model.predict(input_df)[0]
        return {"predicted_crop_yield_tonnes_per_ha": float(pred)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/forecast_next_days")
def forecast_next_days(req: ProphetNextNDays):
    """
    Forecast next N days. We repeat the same regressor values for all future dates.
    (If you trained Prophet without regressors, you can ignore scaling and pass future only.)
    """
    try:
        # Scale provided regressor set (single row)
        reg = {
            "Temperature": req.Temperature,
            "Humidity": req.Humidity,
            "Wind_Speed": req.Wind_Speed,
            "Soil_pH": req.Soil_pH,
            "N": req.N,
            "P": req.P,
            "K": req.K,
            "Soil_Quality": req.Soil_Quality
        }
        reg_scaled_series = _scale_regressors_single(reg)

        # Create future frame
        future = prophet_model.make_future_dataframe(periods=req.days, freq="D")
        # assign same scaled values to each future row
        for col in prophet_regressors:
            future[col] = reg_scaled_series[col]

        fc = prophet_model.predict(future).tail(req.days)
        daily = fc[["ds", "yhat"]].rename(columns={"yhat": "predicted_yield"})
        avg = float(daily["predicted_yield"].mean())
        # round results to reasonable precision (2 or 3 decimals)
        daily["predicted_yield"] = daily["predicted_yield"].round(3)
        return {
            "daily_predicted_yield": daily.to_dict(orient="records"),
            "average_predicted_yield": round(avg, 3)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/forecast_range")
def forecast_range(req: ProphetDateRange):
    """
    Forecast for a custom date range (start_date to end_date inclusive).
    """
    try:
        # construct date range
        dates = pd.date_range(start=req.start_date, end=req.end_date, freq="D")
        reg = {
            "Temperature": req.Temperature,
            "Humidity": req.Humidity,
            "Wind_Speed": req.Wind_Speed,
            "Soil_pH": req.Soil_pH,
            "N": req.N,
            "P": req.P,
            "K": req.K,
            "Soil_Quality": req.Soil_Quality
        }
        reg_scaled_series = _scale_regressors_single(reg)
        future = pd.DataFrame({"ds": dates})
        for col in prophet_regressors:
            future[col] = reg_scaled_series[col]
        fc = prophet_model.predict(future)
        daily = fc[["ds", "yhat"]].rename(columns={"yhat": "predicted_yield"})
        daily["predicted_yield"] = daily["predicted_yield"].round(3)
        avg = float(daily["predicted_yield"].mean())
        return {
            "dates_and_yield": daily.to_dict(orient="records"),
            "average_predicted_yield": round(avg, 3)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")
