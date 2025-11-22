# from fastapi import FastAPI
# from pydantic import BaseModel,validator
# import joblib
# import pandas as pd
# from catboost import CatBoostRegressor

# app = FastAPI(title = "Crop Yield Prediction API")

# model_path = "regression_models\catboost.cbm"
# model = CatBoostRegressor()
# model.load_model(model_path)

# ALLOWED_CROPS = {
#     'barley', 'corn', 'cotton', 'potato', 'rice', 
#     'soybean', 'sugarcane', 'sunflower', 'tomato', 'wheat'
# }

# ALLOWED_SOILS = {'clay', 'loamy', 'peaty', 'saline', 'sandy'}



# class CropYieldPrediction(BaseModel):
#     Crop_Type: str
#     Soil_Type: str
#     Soil_pH: float
#     Temperature: float
#     Humidity: float
#     Wind_Speed: float
#     N: float
#     P: float
#     K: float
#     Soil_Quality: float

#     @validator("Crop_Type")
#     def validate_crop(cls, value):
#         value_lower = value.strip().lower()
#         if value_lower not in ALLOWED_CROPS:
#             raise ValueError(f"Crop_Type '{value}' is not supported. Allowed: {ALLOWED_CROPS}")
#         return value_lower

#     @validator("Soil_Type")
#     def validate_soil(cls, value):
#         value_lower = value.strip().lower()
#         if value_lower not in ALLOWED_SOILS:
#             raise ValueError(f"Soil_Type '{value}' is not supported. Allowed: {ALLOWED_SOILS}")
#         return value_lower
    
# class CropYieldForecast(BaseModel):
#     Crop_Type: str
#     Months: int
#     Soil_pH: float
#     Temperature: float
#     Humidity: float
#     Wind_Speed: float
#     N: float
#     P: float
#     K: float
#     Soil_Quality: float


#     @validator("Crop_Type")
#     def validate_crop(cls, value):
#         value_lower = value.strip().lower()
#         if value_lower not in ALLOWED_CROPS:
#             raise ValueError(f"Crop_Type '{value}' is not supported.Allowed: {ALLOWED_CROPS}")
#         return value_lower
    
#     @validator("Months")
#     def validate_months(cls, value):
#         if value < 1:
#             raise ValueError("Months must be at least 1")
#         return value



# @app.post("/predict")
# def predict_yield(data: CropYieldPrediction):
#     input_data = pd.DataFrame([data.dict()])
#     prediction = model.predict(input_data)[0]

#     return {
#         "Predicted_Yield " : round(float(prediction), 2),
#         "Input_Data" : data
#     }

# @app.post("/forecast")
# def forecast_crop_with_regressors(request: CropYieldForecast):
#     crop = request.Crop_Type
#     months = request.Months

#     model_path = f"prophet_reg_models/{crop}_prophet.pkl"
#     model = joblib.load(model_path)

#     future_dates = pd.date_range(start=pd.Timestamp.today(), periods=months, freq='ME')
#     future_df = pd.DataFrame({"ds": future_dates})

#     # Repeat each regressor for all months
#     for col in ["Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K", "Soil_Quality"]:
#         future_df[col] = [getattr(request, col)] * months

#     # Predict
#     forecast = model.predict(future_df)

#     result = []
#     for _, row in forecast.iterrows():
#         result.append({
#             "Month": row['ds'].strftime("%Y-%m"),
#             "Predicted_Yield": round(float(row['yhat']), 2)
#         })

#     return {
#         "Crop": crop,
#         "Forecast_Horizon_Months": months,
#         "Forecast": result
#     }


from fastapi import FastAPI, Request
from pydantic import BaseModel, validator
import joblib
import pandas as pd
from catboost import CatBoostRegressor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time

app = FastAPI(title="Crop Yield Prediction API")

# ---------------------------
# 1. CORS Middleware
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 2. Logging Middleware
# ---------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} completed in {duration:.3f}s")
    return response

# ---------------------------
# 3. Validation Error Handler
# ---------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input data",
            "details": exc.errors(),
        },
    )

# ---------------------------
# Load ML Model
# ---------------------------
model_path = "regression_models/catboost.cbm"
model = CatBoostRegressor()
model.load_model(model_path)

ALLOWED_CROPS = {'barley','corn','cotton','potato','rice','soybean','sugarcane','sunflower','tomato','wheat'}
ALLOWED_SOILS = {'clay','loamy','peaty','saline','sandy'}

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
            raise ValueError(f"Crop_Type '{value}' is not supported")
        return value_lower

    @validator("Soil_Type")
    def validate_soil(cls, value):
        value_lower = value.strip().lower()
        if value_lower not in ALLOWED_SOILS:
            raise ValueError(f"Soil_Type '{value}' is not supported")
        return value_lower

class CropYieldForecast(BaseModel):
    Crop_Type: str
    Months: int
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
            raise ValueError(f"Crop_Type '{value}' is not supported")
        return value_lower

    @validator("Months")
    def validate_months(cls, value):
        if value < 1:
            raise ValueError("Months must be at least 1")
        return value

@app.post("/predict")
def predict_yield(data: CropYieldPrediction):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]

    return {
        "Predicted_Yield": round(float(prediction), 2),
        "Input_Data": data
    }

@app.post("/forecast")
def forecast_crop_with_regressors(request: CropYieldForecast):
    crop = request.Crop_Type
    months = request.Months

    model_path = f"prophet_reg_models/{crop}_prophet.pkl"
    model = joblib.load(model_path)

    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=months, freq='ME')
    future_df = pd.DataFrame({"ds": future_dates})

    for col in ["Soil_pH","Temperature","Humidity","Wind_Speed","N","P","K","Soil_Quality"]:
        future_df[col] = getattr(request, col)

    forecast = model.predict(future_df)

    result = [
        {"Month": row["ds"].strftime("%Y-%m"), "Predicted_Yield": round(float(row["yhat"]), 2)}
        for _, row in forecast.iterrows()
    ]

    return {
        "Crop": crop,
        "Forecast_Horizon_Months": months,
        "Forecast": result
    }
