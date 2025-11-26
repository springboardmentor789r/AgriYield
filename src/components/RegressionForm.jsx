import React, { useState } from "react";
import { predictCrop } from "../services/api";

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];
const SOIL_OPTIONS = ["peaty","loamy","clay","sandy","saline"];

export default function RegressionForm({ setResult, setLoading }) {
  const [form, setForm] = useState({
    temperature: 25,
    humidity: 60,
    soilph: 6.5,
    windspeed: 10,
    soilquality: 50,
    n: 50,
    p: 40,
    k: 30,
    croptype: "Wheat",
    soiltype: "peaty",
  });
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading?.(true);
    try {
      const payload = {
        ...form,
        temperature: parseFloat(form.temperature),
        humidity: parseFloat(form.humidity),
        soilph: parseFloat(form.soilph),
        windspeed: parseFloat(form.windspeed),
        soilquality: parseFloat(form.soilquality),
        n: parseFloat(form.n),
        p: parseFloat(form.p),
        k: parseFloat(form.k),
      };

      const res = await predictCrop(payload);
      setResult(res);
    } catch (err) {
      setError(err?.response?.data || err.message);
      setResult(null);
    } finally {
      setLoading?.(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
      {[ 
        { key: "temperature", label: "Temperature (°C)" },
        { key: "humidity", label: "Humidity (%)" },
        { key: "soilph", label: "Soil pH" },
        { key: "windspeed", label: "Wind Speed (Km/h)" },
        { key: "soilquality", label: "Soil Quality Index" },

        // 🔥 UPDATED LABELS HERE
        { key: "n", label: "Nitrogen (Kg/ha)" },
        { key: "p", label: "Phosphorus (Kg/ha)" },
        { key: "k", label: "Potassium (Kg/ha)" },

      ].map(f => (
        <label key={f.key} className="flex flex-col">
          <span className="text-sm text-gray-600">{f.label}</span>
          <input className="input" name={f.key} value={form[f.key]} onChange={handleChange} />
        </label>
      ))}

      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Crop Type</span>
        <select name="croptype" className="input" value={form.croptype} onChange={handleChange}>
          {CROP_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>

      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Soil Type</span>
        <select name="soiltype" className="input" value={form.soiltype} onChange={handleChange}>
          {SOIL_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>

      <div className="col-span-2 flex gap-3 mt-2">
        <button type="submit" className="btn-primary">Predict</button>

        <button
          type="button"
          onClick={() => {
            setForm({
              temperature: 25, humidity: 60, soilph: 6.5, windspeed: 10,
              soilquality: 50, n: 50, p: 40, k: 30, croptype: "Wheat", soiltype: "peaty"
            });
            setResult(null);
            setError(null);
          }}
          className="px-4 py-2 border rounded"
        >
          Reset
        </button>
      </div>

      {error && <div className="col-span-2 text-red-600">{JSON.stringify(error)}</div>}
    </form>
  );
}
