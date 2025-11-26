from pydantic import BaseModel
from typing import List

# ---------------- Crop Yield Request ----------------
class CropYieldRequest(BaseModel):
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

class CropYieldResponse(BaseModel):
    predicted_yield: float

# ---------------- Time Series Forecast ----------------
class TSDataPoint(BaseModel):
    date: str
    value: float

class TSForecastRequest(BaseModel):
    series: List[TSDataPoint]
    horizon: int = 7

class TSForecastResponse(BaseModel):
    forecast: List[float]
    horizon: int
