


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
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request Logging Middleware
# ---------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} completed in {duration:.3f}s")
    return response

# ---------------------------
# Validation Error Handler
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
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request Logging Middleware
# ---------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"{request.method} {request.url.path} completed in {duration:.3f}s")
    return response

# ---------------------------
# Validation Error Handler
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
# Load CatBoost Model
# ---------------------------
model_path = "regression_models/catboost.cbm"
model = CatBoostRegressor()
model.load_model(model_path)

ALLOWED_CROPS = {
    'barley', 'corn', 'cotton', 'potato', 'rice',
    'soybean', 'sugarcane', 'sunflower', 'tomato', 'wheat'
}

ALLOWED_SOILS = {'clay', 'loamy', 'peaty', 'saline', 'sandy'}

# ---------------------------
# Request Models
# ---------------------------
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

    @validator("Crop_Type")
    def validate_crop(cls, value):
        v = value.strip().lower()
        if v not in ALLOWED_CROPS:
            raise ValueError(f"Crop_Type '{value}' is not supported")
        return v

    @validator("Soil_Type")
    def validate_soil(cls, value):
        v = value.strip().lower()
        if v not in ALLOWED_SOILS:
            raise ValueError(f"Soil_Type '{value}' is not supported")
        return v


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

    @validator("Crop_Type")
    def validate_crop(cls, value):
        v = value.strip().lower()
        if v not in ALLOWED_CROPS:
            raise ValueError(f"Crop_Type '{value}' is not supported")
        return v

    @validator("Months")
    def validate_months(cls, value):
        if value < 1:
            raise ValueError("Months must be at least 1")
        return value

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def predict_yield(data: CropYieldPrediction):

    # Auto-calculate soil quality
    soil_quality = (data.N + data.P + data.K) / 3

    # Build DataFrame
    df_dict = data.dict()
    df_dict["Soil_Quality"] = soil_quality

    df = pd.DataFrame([df_dict])

    prediction = model.predict(df)[0]

    return {
        "Predicted_Yield": round(float(prediction), 2)
    }

# ---------------------------
# Forecast Endpoint
# ---------------------------
@app.post("/forecast")
def forecast_crop_with_regressors(request: CropYieldForecast):

    crop = request.Crop_Type
    months = request.Months

    # Auto soil quality
    soil_quality = (request.N + request.P + request.K) / 3

    # Load Prophet regressor model
    model_path = f"prophet_reg_models/{crop}_prophet.pkl"
    model = joblib.load(model_path)

    # Build future regressor DF
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=months, freq='ME')
    future_df = pd.DataFrame({"ds": future_dates})

    for col in ["Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K"]:
        future_df[col] = getattr(request, col)

    future_df["Soil_Quality"] = soil_quality

    forecast = model.predict(future_df)

    result = [
        {
            "Month": row["ds"].strftime("%Y-%m"),
            "Predicted_Yield": round(float(row["yhat"]), 2)
        }
        for _, row in forecast.iterrows()
    ]

    return {
        "Crop": crop,
        "Forecast_Horizon_Months": months,
        "Forecast": result
    }
