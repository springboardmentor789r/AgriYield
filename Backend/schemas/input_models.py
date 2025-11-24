from pydantic import BaseModel
from datetime import date
from typing import List


class CropYieldInput(BaseModel): 
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

class ForecastInput(BaseModel):
    start_date: date
    periods: int
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float