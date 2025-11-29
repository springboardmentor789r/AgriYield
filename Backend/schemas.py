from pydantic import BaseModel
from typing import Optional

class RegressionInput(BaseModel):
    # Numerical Fields (using exact column names from your model input)
    temperature: float
    humidity: float
    soilph: float
    windspeed: float
    soilquality: float
    
    # Nutrient Fields (must match lowercase model columns)
    n: float
    p: float
    k: float
    
    # Categorical Fields (must match model columns)
    croptype: str
    soiltype: str

# -------------------------------
# Time Series Forecast Schema (Prophet only)
# -------------------------------
class TimeSeriesForecast(BaseModel):
    crop_type: str          # e.g., "rice", "wheat"
    from_date: str          # YYYY-MM-DD
    to_date: str            # YYYY-MM-DD

    # Optional Prophet regressors
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soilph: Optional[float] = None
    windspeed: Optional[float] = None
    n: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    soilquality: Optional[float] = None
