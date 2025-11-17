from fastapi import APIRouter
from app.schemas import TimeSeriesForecast

import pandas as pd
import joblib
from typing import List, Optional

# Prophet
from prophet import Prophet

router = APIRouter()

# ===============================
#   PROPHET FORECAST FUNCTION
# ===============================
def generate_prophet_forecast(request: TimeSeriesForecast) -> List[dict]:
    # Convert crop_type to lowercase and remove spaces if needed
    crop_name = request.crop_type.lower().replace(" ", "_")
    model_path = f"models/{crop_name}_prophet.pkl"
    
    # Load Prophet model
    model: Prophet = joblib.load(model_path)

    start = pd.to_datetime(request.from_date)
    end = pd.to_datetime(request.to_date)
    forecast_index = pd.date_range(start=start, end=end, freq="D")

    # Build future DataFrame
    future = pd.DataFrame(forecast_index, columns=["ds"])

    # Add regressors (only if provided)
    regressors = {
        "Temperature": request.temperature,
        "Humidity": request.humidity,
        "Wind_Speed": request.windspeed,
        "Soil_pH": request.soilph,
        "N": request.n,
        "P": request.p,
        "K": request.k,
        "Soil_Quality": request.soilquality,
    }

    for col, val in regressors.items():
        if val is not None:
            future[col] = [val] * len(future)

    # Forecast
    forecast = model.predict(future)

    # Format result
    result = [{"date": str(row["ds"].date()), "predicted_yield": float(row["yhat"])}
              for _, row in forecast.iterrows()]
    return result

# ===============================
#   TIME SERIES ROUTE
# ===============================
@router.post("/predict/timeseries")
def predict_timeseries(data: TimeSeriesForecast):
    # Only Prophet is supported
    forecast = generate_prophet_forecast(data)

    return {
        "crop_type": data.crop_type,
        "model_used": "prophet",
        "from_date": data.from_date,
        "to_date": data.to_date,
        "forecast": forecast
    }
