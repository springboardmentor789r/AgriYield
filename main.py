from fastapi import FastAPI
from pydantic import BaseModel, validator
import joblib
import pandas as pd
from catboost import CatBoostRegressor

app = FastAPI(title = "Crop Yield Prediction API")

model_path = "regression_models\catboost.cbm"
model = CatBoostRegressor()
model.load_model(model_path)

ALLOWED_CROPS = {
    'barley', 'corn', 'cotton', 'potato', 'rice', 
    'soybean', 'sugarcane', 'sunflower', 'tomato', 'wheat'
}

ALLOWED_SOILS = {'clay', 'loamy', 'peaty', 'saline', 'sandy'}



class CropYieldPrediction(BaseModel):
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

    @validator("Crop_Type")
    def validate_crop(cls, value):
        value_lower = value.strip().lower()
        if value_lower not in ALLOWED_CROPS:
            raise ValueError(f"Crop_Type '{value}' is not supported. Allowed: {ALLOWED_CROPS}")
        return value_lower

    @validator("Soil_Type")
    def validate_soil(cls, value):
        value_lower = value.strip().lower()
        if value_lower not in ALLOWED_SOILS:
            raise ValueError(f"Soil_Type '{value}' is not supported. Allowed: {ALLOWED_SOILS}")
        return value_lower
    
class CropYieldForecast(BaseModel):
    Crop_Type : str
    Months: int

    @validator("Crop_Type")
    def validate_crop(cls, value):
        value_lower = value.strip().lower()
        if value_lower not in ALLOWED_CROPS:
            raise ValueError(f"Crop_Type '{value}' is not supported.Allowed: {ALLOWED_CROPS}")
        return value_lower
    @validator("Months")
    def validate_months(cls, value):
        if value < 1:
            raise ValueError("Months must be at least 1")
        return value




@app.post("/predict")
def predict_yield(data: CropYieldPrediction):
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]

    return {
        "Predicted_Yield " : round(float(prediction), 2),
        "Input_Data" : data
    }

@app.post("/forecast")
def forecast_crop(request: CropYieldForecast):
    crop = request.Crop_Type
    months = request.Months
    model_path = f"prophet_models\\{crop}_prophet.pkl"
    model = joblib.load(model_path)


    future = pd.DataFrame(pd.date_range(start=pd.Timestamp.today(), periods=months, freq='ME'),columns=['ds'])

    forecast = model.predict(future)

    result = []
    for _, row in forecast.iterrows():
        result.append({
            "Month": row['ds'].strftime("%Y-%m"),
            "Predicted_Yield": round(float(row['yhat']), 2)
        })

    return {
        "Crop": crop,
        "Forecast_Horizon_Months": months,
        "Forecast": result
    } 




