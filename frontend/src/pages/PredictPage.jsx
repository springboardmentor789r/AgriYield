import { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import { predictYield } from "../services/api";

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];
const SOIL_OPTIONS = ["Loamy", "Clay", "Sandy", "Saline", "Peaty"];

export default function PredictPage() {
  const [form, setForm] = useState({
    Crop_Type: "rice",
    Soil_Type: "Loamy",
    Soil_pH: 6.5,
    Temperature: 28,
    Humidity: 80,
    Wind_Speed: 5,
    N: 100,
    P: 40,
    K: 50,
    Soil_Quality: 0.8,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function update(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: isNaN(value) ? value : Number(value) }));
  }

  async function submit(e) {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await predictYield(form);
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto py-12 px-6">
      <div className="bg-white p-8 rounded-xl shadow">
        <h2 className="text-2xl font-bold mb-4">Predict Crop Yield</h2>

        <form onSubmit={submit} className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SelectField label="Crop Type" name="Crop_Type" options={CROP_OPTIONS} value={form.Crop_Type} onChange={update} />
          <SelectField label="Soil Type" name="Soil_Type" options={SOIL_OPTIONS} value={form.Soil_Type} onChange={update} />
          <InputField label="Soil pH" name="Soil_pH" value={form.Soil_pH} onChange={update} />
          <InputField label="Temperature" name="Temperature" value={form.Temperature} onChange={update} />
          <InputField label="Humidity" name="Humidity" value={form.Humidity} onChange={update} />
          <InputField label="Wind Speed" name="Wind_Speed" value={form.Wind_Speed} onChange={update} />
          <InputField label="Nitrogen (N)" name="N" value={form.N} onChange={update} />
          <InputField label="Phosphorus (P)" name="P" value={form.P} onChange={update} />
          <InputField label="Potassium (K)" name="K" value={form.K} onChange={update} />
          <InputField label="Soil Quality" name="Soil_Quality" value={form.Soil_Quality} onChange = {update} />

          <button className="col-span-2 py-2 px-4 bg-emerald-600 text-white rounded" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {result && (
          <div className="mt-6 p-4 bg-emerald-50 rounded border-l-4 border-emerald-600">
            <h3 className="font-semibold">Predicted Yield:</h3>
            <p className="text-xl">{result.Predicted_Yield ?? result.prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
}
