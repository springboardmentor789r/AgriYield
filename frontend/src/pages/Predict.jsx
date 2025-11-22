// src/pages/Predict.jsx
import { useState } from "react";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000";

const cropTypes = [
  "Wheat",
  "Corn",
  "Rice",
  "Barley",
  "Soybean",
  "Cotton",
  "Sugarcane",
  "Tomato",
  "Potato",
  "Sunflower",
];

const soilTypes = ["Sandy", "Clay", "Loamy", "Peaty", "Saline"];

export default function Predict() {
  const [form, setForm] = useState({
    Crop_Type: "Wheat",
    Soil_Type: "Sandy",
    Soil_pH: 6.5,
    Temperature: 25,
    Humidity: 60,
    Wind_Speed: 10,
    N: 40,
    P: 30,
    K: 30,
    Soil_Quality: 50,
  });

  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]:
        name === "Crop_Type" || name === "Soil_Type"
          ? value
          : value === ""
          ? ""
          : Number(value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const payload = {
        Crop_Type: form.Crop_Type,
        Soil_Type: form.Soil_Type,
        Soil_pH: Number(form.Soil_pH),
        Wind_Speed: Number(form.Wind_Speed),
        Temperature: Number(form.Temperature),
        N: Number(form.N),
        P: Number(form.P),
        K: Number(form.K),
        Soil_Quality: Number(form.Soil_Quality),
        Humidity: Number(form.Humidity),
      };

      const res = await axios.post(`${API_BASE}/predict_rf`, payload);

      // ✅ Handle all possible keys from backend
      const value =
        res.data.predicted_crop_yield_tonnes_per_ha ??
        res.data.Predicted_Crop_Yield ??
        res.data.prediction;

      if (value === undefined || value === null) {
        throw new Error("Prediction value missing in response");
      }

      setPrediction(Number(value));
    } catch (err) {
      console.error(err);
      setError("Could not get prediction. Please check backend and inputs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 h-full overflow-y-auto px-6 py-6 bg-gradient-to-b from-emerald-50 to-emerald-100">
      <div className="max-w-6xl mx-auto space-y-6">
        <header>
          <h1 className="text-3xl md:text-4xl font-bold text-emerald-900">
            Crop Yield Prediction
          </h1>
          <p className="mt-2 text-emerald-800/80 max-w-3xl">
            Provide crop, soil and environmental conditions to estimate yield
            (tonnes per hectare) using the trained model.
          </p>
        </header>

        <div className="bg-white rounded-3xl shadow-xl shadow-emerald-900/5 p-6 md:p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Crop & Soil type */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-emerald-900">
                  Crop Type
                </label>
                <select
                  name="Crop_Type"
                  value={form.Crop_Type}
                  onChange={handleChange}
                  className="w-full rounded-2xl border border-emerald-100 bg-emerald-50/60 px-4 py-2.5 text-emerald-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
                >
                  {cropTypes.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-emerald-900">
                  Soil Type
                </label>
                <select
                  name="Soil_Type"
                  value={form.Soil_Type}
                  onChange={handleChange}
                  className="w-full rounded-2xl border border-emerald-100 bg-emerald-50/60 px-4 py-2.5 text-emerald-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
                >
                  {soilTypes.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Numeric inputs */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[
                ["Soil_pH", "Soil pH", "6.5"],
                ["Soil_Quality", "Soil Quality Index", "50"],
                ["Humidity", "Humidity (%)", "60"],
                ["Temperature", "Temperature (°C)", "25"],
                ["Wind_Speed", "Wind Speed (km/h)", "10"],
                ["N", "Nitrogen (N) kg/ha", "40"],
                ["P", "Phosphorus (P) kg/ha", "30"],
                ["K", "Potassium (K) kg/ha", "30"],
              ].map(([name, label, placeholder]) => (
                <div key={name} className="space-y-2">
                  <label className="block text-sm font-medium text-emerald-900">
                    {label}
                  </label>
                  <input
                    type="number"
                    step="any"
                    name={name}
                    value={form[name]}
                    onChange={handleChange}
                    placeholder={placeholder}
                    className="w-full rounded-2xl border border-emerald-100 bg-emerald-50/60 px-4 py-2.5 text-emerald-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
                  />
                </div>
              ))}
            </div>

            <div className="flex items-center gap-4">
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center justify-center rounded-full bg-emerald-600 px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-emerald-600/30 hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed transition-transform transform hover:-translate-y-0.5"
              >
                {loading ? "Predicting..." : "Predict Yield"}
              </button>

              {error && (
                <p className="text-sm text-red-600 font-medium">{error}</p>
              )}
            </div>
          </form>

          <div className="mt-8 pt-6 border-t border-emerald-100">
            <h2 className="text-lg font-semibold text-emerald-900">
              Predicted Yield
            </h2>
            {prediction !== null ? (
              <p className="mt-3 text-3xl font-semibold text-emerald-700">
                {prediction.toFixed(2)}{" "}
                <span className="text-base font-normal text-emerald-800/80">
                  tonnes / hectare
                </span>
              </p>
            ) : (
              <p className="mt-2 text-sm text-emerald-900/70">
                Enter the field conditions and click{" "}
                <span className="font-semibold">Predict Yield</span> to see the
                result.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
