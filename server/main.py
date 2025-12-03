from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import date, timedelta
import pandas as pd
import joblib
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AgriYield API with Time Series Logging")

# ----------------- Middleware -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    print(f"{request.method} {request.url.path} completed in {duration:.3f}s")
    return response

# ----------------- Load Models -----------------
prophet_model = joblib.load(r"C:\Users\ADMIN\Desktop\AgriYieldPredictorproject\server\models\prophet_model.pkl")
catboost_model = joblib.load(r"C:\Users\ADMIN\Desktop\AgriYieldPredictorproject\server\models\CatBoost.pkl")

# ----------------- Schemas -----------------
class RegressionInput(BaseModel):
    soil_type: str
    crop_type: str
    soil_pH: float
    temperature: float
    humidity: float
    wind_speed: float
    n: float
    p: float
    k: float
    soil_quality: float

class TimeSeriesRangeInput(BaseModel):
    start_date: date
    end_date: date
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float

# ----------------- CatBoost Prediction -----------------
@app.post("/predict_regression")
def predict_regression(data: RegressionInput):
    df = pd.DataFrame([data.dict()])
    prediction = catboost_model.predict(df)[0]
    return {"predicted_crop_yield": float(prediction)}

@app.post("/predict_timeseries")
def predict_timeseries(data: TimeSeriesRangeInput):
    # Generate full date range
    date_range = pd.date_range(start=data.start_date, end=data.end_date)
    
    # Create dataframe for all dates
    df = pd.DataFrame({
        "ds": date_range,
        "Soil_pH": data.Soil_pH,
        "Temperature": data.Temperature,
        "Humidity": data.Humidity,
        "Wind_Speed": data.Wind_Speed,
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "Soil_Quality": data.Soil_Quality
    })

    # Predict using Prophet
    forecast = prophet_model.predict(df)

    # Extract predictions
    predicted_values = forecast["yhat"].tolist()
    
    # Calculate average
    avg_pred = sum(predicted_values) / len(predicted_values)

    return {
        "predicted_crop_yield": round(float(avg_pred), 2)
    }
# ----------------- Health Check -----------------
@app.get("/")
def root():
    return {"message": "AgriYield API Running Successfully"}
