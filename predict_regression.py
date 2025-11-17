from fastapi import APIRouter
from app.schemas import RegressionInput
import joblib
import pandas as pd
import os

router = APIRouter()

# Directory where crop models are saved
MODEL_DIR = "models"

@router.post("/predict")
def predict_crop_yield(request: RegressionInput):
    crop = request.croptype.lower()
    model_path = os.path.join(MODEL_DIR, f"{crop}_catboost_no_date.pkl")

    if not os.path.exists(model_path):
        available = [f.replace('_catboost_no_date.pkl','') for f in os.listdir(MODEL_DIR) if f.endswith('_catboost_no_date.pkl')]
        return {"error": f"No model found for crop '{request.croptype}'. Available crops: {available}"}

    # Load model
    model = joblib.load(model_path)

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "croptype": crop,
        "soiltype": request.soiltype,
        "temperature": request.temperature,
        "humidity": request.humidity,
        "windspeed": request.windspeed,
        "soilph": request.soilph,
        "n": request.n,
        "p": request.p,
        "k": request.k,
        "soilquality": request.soilquality
    }])

    # Predict
    pred = model.predict(input_df)[0]

    return {
        "Crop": request.croptype,
        "Predicted_Yield": float(pred)
    }
