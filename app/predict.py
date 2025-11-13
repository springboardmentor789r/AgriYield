import os
import pandas as pd
import joblib

# Path to your saved full pipeline
PIPELINE_PATH = "saved_models\crop_yield_catboost_model.pkl"

# Load pipeline ONCE at startup (best practice)
print("🔄 Loading crop yield prediction pipeline...")
pipe = joblib.load(PIPELINE_PATH)
print("✅ Pipeline loaded successfully!")

def predict_crop_yield(payload: dict) -> float:
    """
    payload: dictionary of crop features
    """
    df = pd.DataFrame([payload])
    pred = pipe.predict(df)[0]
    return float(pred)
