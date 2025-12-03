# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)   # allow all origins for dev

# ---- LOAD ALL MODELS ----
model   = pickle.load(open("trained_model.pkl", "rb"))
encoder = pickle.load(open("loo_encoder.pkl", "rb"))
scaler  = pickle.load(open("minmax_scaler.pkl", "rb"))

# REQUIRED FEATURE ORDER
features = [
    'Date','Crop_Type','Soil_Type','Soil_pH','Temperature',
    'Humidity','Wind_Speed','N','P','K','Soil_Quality'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("\n📩 Incoming Payload:", data)

        # ----- Build DF -----
        row = {f: data.get(f, None) for f in features}
        df = pd.DataFrame([row])

        # ----- DATE -----
        try:
            if not df.loc[0, "Date"]:
                df.loc[0, "Date"] = pd.Timestamp.now()
            df["Date"] = pd.to_datetime(df["Date"])
        except:
            df["Date"] = pd.Timestamp.now()

        df["Date"] = df["Date"].astype("int64") // 10**9

        # ----- HANDLE CATEGORIES -----
        df["Crop_Type"] = df["Crop_Type"].astype(str).str.lower()
        df["Soil_Type"] = df["Soil_Type"].astype(str).str.lower()

        print("🔤 Before Encoding:\n", df[["Crop_Type", "Soil_Type"]])

        # encoder expects 2 columns
        try:
            df[["Crop_Type","Soil_Type"]] = encoder.transform(
                df[["Crop_Type","Soil_Type"]]
            )
        except Exception as e:
            print("⚠ ENCODER ERROR:", e)
            return jsonify({"error": "Encoder failed", "detail": str(e)}), 400

        print("🔤 After Encoding:\n", df[["Crop_Type", "Soil_Type"]])

        # ----- NUMERIC -----
        num_cols = ['Soil_pH','Temperature','Humidity','Wind_Speed','N','P','K','Soil_Quality']
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        # ----- SCALE -----
        df[num_cols] = scaler.transform(df[num_cols])

        # ----- PREDICT -----
        pred = float(model.predict(df)[0])
        print("🔥 MODEL PREDICTION:", pred)

        result = {
            "predicted_q_per_ha": round(pred, 3),
            "predicted_kg_per_ha": round(pred * 100, 1)
        }

        print("📤 Sending Back:", result)
        return jsonify(result), 200

    except Exception as e:
        print("\n🚨 SERVER ERROR:", e)
        print(traceback.format_exc())
        return jsonify({"error": "server error", "detail": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀 SERVER READY → http://127.0.0.1:5000/predict\n")
    app.run(debug=True, port=5000)
