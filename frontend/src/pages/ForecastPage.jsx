import React, { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import ForecastChart from "../components/ForecastChart";
import { validateForm } from "../utils/validation";
import { forecastYield } from "../services/api";

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];

export default function ForecastPage() {
  const [form, setForm] = useState({
    Crop_Type: "",
    Months: "",
    Soil_pH: "",
    Temperature: "",
    Humidity: "",
    Wind_Speed: "",
    N: "",
    P: "",
    K: "",
  });

  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [forecast, setForecast] = useState(null);

  function update(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));

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

    const res = await forecastYield(form);
    setForecast(res);
  }

  return (
    <div className="max-w-5xl mx-auto py-12 px-6">
      <div className="bg-white p-8 rounded-xl shadow">
        <h2 className="text-2xl font-bold mb-4">Forecast Yield</h2>

        <form onSubmit={submit} className="grid grid-cols-1 md:grid-cols-3 gap-4">

          <SelectField label="Crop Type" name="Crop_Type"
            value={form.Crop_Type} options={CROP_OPTIONS}
            onChange={update} error={errors.Crop_Type} />

          <InputField label="Months" name="Months" value={form.Months}
            onChange={update} error={errors.Months} />

          <InputField label="Soil pH" name="Soil_pH" value={form.Soil_pH}
            onChange={update} error={errors.Soil_pH} />

          <InputField label="Temperature" name="Temperature" value={form.Temperature}
            onChange={update} unit="°C" error={errors.Temperature} />

          <InputField label="Humidity" name="Humidity" value={form.Humidity}
            onChange={update} unit="%" error={errors.Humidity} />

          <InputField label="Wind Speed" name="Wind_Speed" value={form.Wind_Speed}
            onChange={update} unit="km/h" error={errors.Wind_Speed} />

          <InputField label="N" name="N" value={form.N}
            onChange={update} unit="ppm" error={errors.N} />

          <InputField label="P" name="P" value={form.P}
            onChange={update} unit="ppm" error={errors.P} />

          <InputField label="K" name="K" value={form.K}
            onChange={update} unit="ppm" error={errors.K} />

          <button className="col-span-3 py-2 px-4 bg-sky-600 text-white rounded">
            Forecast
          </button>
        </form>

        {forecast && (
          <>
            <table className="w-full mt-6 border">
              <thead>
                <tr className="bg-slate-100">
                  <th className="p-2">Month</th>
                  <th className="p-2">Predicted Yield</th>
                </tr>
              </thead>
              <tbody>
                {forecast.Forecast.map((f, i) => (
                  <tr key={i} className="border-t">
                    <td className="p-2">{f.Month}</td>
                    <td className="p-2">{f.Predicted_Yield}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-6">
              <ForecastChart forecast={forecast.Forecast} />
            </div>
          </>
        )}

      </div>
    </div>
  );
}
