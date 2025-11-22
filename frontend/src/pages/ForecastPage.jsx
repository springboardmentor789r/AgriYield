import React, { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import ForecastChart from "../components/ForecastChart";
import { forecastYield } from "../services/api";

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];

export default function ForecastPage() {
  const [form, setForm] = useState({
    Crop_Type: "rice",
    Months: 6,
    Soil_pH: 6.5,
    Temperature: 28,
    Humidity: 80,
    Wind_Speed: 5,
    N: 100,
    P: 40,
    K: 50,
    Soil_Quality: 0.8,
  });

  const [forecast, setForecast] = useState(null);

  function update(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: isNaN(value) ? value : Number(value) }));
  }

  async function submit(e) {
    e.preventDefault();
    const res = await forecastYield(form);
    setForecast(res);
  }

  return (
    <div className="max-w-5xl mx-auto py-12 px-6">
      <div className="bg-white p-8 rounded-xl shadow">
        <h2 className="text-2xl font-bold mb-4">Forecast Yield</h2>

        <form onSubmit={submit} className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <SelectField label="Crop Type" name="Crop_Type" options={CROP_OPTIONS} value={form.Crop_Type} onChange={update} />
          <InputField label="Months" name="Months" value={form.Months} onChange={update} />
          <InputField label="Soil pH" name="Soil_pH" value={form.Soil_pH} onChange={update} />
          <InputField label="Temperature" name="Temperature" value={form.Temperature} onChange={update} />
          <InputField label="Humidity" name="Humidity" value={form.Humidity} onChange={update} />
          <InputField label="Wind Speed" name="Wind_Speed" value={form.Wind_Speed} onChange={update} />
          <InputField label="N" name="N" value={form.N} onChange={update} />
          <InputField label="P" name="P" value={form.P} onChange={update} />
          <InputField label="K" name="K" value={form.K} onChange={update} />
          <InputField label="Soil Quality" name="Soil_Quality" value={form.Soil_Quality} onChange={update} />

          <button className="col-span-3 py-2 px-4 bg-sky-600 text-white rounded">Forecast</button>
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
