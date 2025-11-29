import { useState } from "react";

export default function CatBoostPage() {
  const [formData, setFormData] = useState({
    Crop_Type: "",
    Soil_Type: "",
    Soil_Quality: "",
    Soil_pH: "",
    Temperature: "",
    Humidity: "",
    Wind_Speed: "",
    N: "",
    P: "",
    K: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    const payload = {
      ...formData,
      Soil_Quality: parseFloat(formData.Soil_Quality),
      Soil_pH: parseFloat(formData.Soil_pH),
      Temperature: parseFloat(formData.Temperature),
      Humidity: parseFloat(formData.Humidity),
      Wind_Speed: parseFloat(formData.Wind_Speed),
      N: parseFloat(formData.N),
      P: parseFloat(formData.P),
      K: parseFloat(formData.K),
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/predict-catboost", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setPrediction(data.predicted_crop_yield);
    } catch (err) {
      setError("Failed to fetch prediction. Check backend and input.");
    } finally {
      setLoading(false);
    }
  };

  const numericKeys = [
    "Soil_Quality",
    "Soil_pH",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "N",
    "P",
    "K",
  ];

  return (
    <section className="bg-slate-900/70 border border-slate-800 rounded-2xl shadow-xl shadow-black/40 p-6">
      <h2 className="text-xl font-semibold mb-1">CatBoost Crop Yield</h2>
      <p className="text-sm text-slate-400 mb-5">
        Enter crop, soil, and weather details to get a single yield prediction.
      </p>

      <form
        onSubmit={handleSubmit}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-4"
      >
        {Object.keys(formData).map((key) => (
          <div key={key} className="flex flex-col gap-1">
            <label className="text-xs font-medium text-slate-300">
              {key.replace(/_/g, " ")}
            </label>
            <input
              name={key}
              value={formData[key]}
              onChange={handleChange}
              type={numericKeys.includes(key) ? "number" : "text"}
              step="any"
              required
              className="w-full rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/70 focus:border-emerald-500/70"
              placeholder={key.replace(/_/g, " ")}
            />
          </div>
        ))}

        <div className="sm:col-span-2 lg:col-span-3 flex justify-start">
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center rounded-full bg-emerald-500 px-5 py-2 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/40 hover:bg-emerald-400 disabled:opacity-60 disabled:cursor-not-allowed transition"
          >
            {loading ? "Predicting..." : "Submit"}
          </button>
        </div>
      </form>

      {prediction !== null && (
        <div className="mt-3 rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-300">
          Crop Yield:{" "}
          <span className="font-semibold">{prediction.toFixed(4)}</span>
        </div>
      )}

      {error && (
        <div className="mt-3 rounded-xl border border-rose-500/50 bg-rose-500/10 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
    </section>
  );
}
