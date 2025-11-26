from fastapi import FastAPI, HTTPException
from app.models import CropYieldRequest, CropYieldResponse, TSForecastRequest, TSForecastResponse
from app.predict import predict_crop_yield
from app.ts_forecast import simple_last_value_forecast, simple_ar_forecast

app = FastAPI(
    title="Crop Yield & Forecasting API",
    version="1.0.0",
    description="API for crop yield prediction and time-series forecasting."
)

@app.get("/")
def root():
    return {"message": "FastAPI is running successfully!"}

# ------------------- Crop Yield Endpoint -------------------
@app.post("/predict/crop", response_model=CropYieldResponse)
def predict_crop(payload: CropYieldRequest):
    try:
        prediction = predict_crop_yield(payload.dict())
        return {"predicted_yield": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Time Series Forecast Endpoint -------------------
@app.post("/forecast/ts", response_model=TSForecastResponse)
def forecast_ts(req: TSForecastRequest):

    values = [point.value for point in req.series]

    # Simple AR if enough data else fallback
    forecast = (
        simple_ar_forecast(values, req.horizon)
        if len(values) > 5
        else simple_last_value_forecast(values, req.horizon)
    )

    return TSForecastResponse(forecast=forecast, horizon=req.horizon)
