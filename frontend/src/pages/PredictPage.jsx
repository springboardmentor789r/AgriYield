import { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import { predictYield } from "../services/api";
import { validateForm } from "../utils/validation";

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];
const SOIL_OPTIONS = ["Loamy", "Clay", "Sandy", "Saline", "Peaty"];

export default function PredictPage() {
  const [form, setForm] = useState({
    Crop_Type: "",
    Soil_Type: "",
    Soil_pH: "",
    Temperature: "",
    Humidity: "",
    Wind_Speed: "",
    N: "",
    P: "",
    K: "",
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  function update(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));

    // Validate on change if already submitted
    if (submitted) {
      const newErrors = validateForm({ ...form, [name]: value });
      setErrors(newErrors);
    }
  }

  async function submit(e) {
    e.preventDefault();
    setSubmitted(true);

    const newErrors = validateForm(form);
    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    // Convert numeric fields before sending to API
    const payload = { ...form };
    const numericFields = ["Soil_pH","Temperature","Humidity","Wind_Speed","N","P","K"];
    numericFields.forEach(field => {
      payload[field] = Number(payload[field]);
    });

    setLoading(true);
    try {
      const res = await predictYield(payload);
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

          <SelectField
            label="Crop Type"
            name="Crop_Type"
            options={CROP_OPTIONS}
            value={form.Crop_Type}
            onChange={update}
            error={errors.Crop_Type}
          />

          <SelectField
            label="Soil Type"
            name="Soil_Type"
            options={SOIL_OPTIONS}
            value={form.Soil_Type}
            onChange={update}
            error={errors.Soil_Type}
          />

          <InputField label="Soil pH" name="Soil_pH" value={form.Soil_pH} onChange={update} error={errors.Soil_pH} />
          <InputField label="Temperature" name="Temperature" value={form.Temperature} onChange={update} unit="°C" error={errors.Temperature} />
          <InputField label="Humidity" name="Humidity" value={form.Humidity} onChange={update} unit="%" error={errors.Humidity} />
          <InputField label="Wind Speed" name="Wind_Speed" value={form.Wind_Speed} onChange={update} unit="km/h" error={errors.Wind_Speed} />
          <InputField label="N" name="N" value={form.N} onChange={update} unit="ppm" error={errors.N} />
          <InputField label="P" name="P" value={form.P} onChange={update} unit="ppm" error={errors.P} />
          <InputField label="K" name="K" value={form.K} onChange={update} unit="ppm" error={errors.K} />

          <button className="col-span-2 py-2 px-4 bg-emerald-600 text-white rounded" disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {result && (
          <div className="mt-6 p-4 bg-emerald-50 rounded border-l-4 border-emerald-600">
            <h3 className="font-semibold">Predicted Yield:</h3>
            <p className="text-xl">{result.Predicted_Yield}</p>
          </div>
        )}
      </div>
    </div>
  );
}
