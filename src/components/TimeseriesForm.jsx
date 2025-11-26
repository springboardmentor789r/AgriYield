import React, { useState } from "react";
import { predictTimeseries } from "../services/api";

export default function TimeSeriesForm({ setResult, setLoading }) {
  const [form, setForm] = useState({
    crop_type: "rice",
    from_date: "2025-11-18",
    to_date: "2025-11-25",
    temperature: 30,
    humidity: 75,
    soilph: 6.5,
    windspeed: 5,
    n: 10,
    p: 5,
    k: 5,
    soilquality: 7,
  });

  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading?.(true);
    setError(null);

    try {
      const res = await predictTimeseries(form);
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

      {/* Crop Type */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Crop Type</span>
        <select
          name="crop_type"
          className="input"
          value={form.crop_type}
          onChange={handleChange}
        >
          <option value="rice">Rice</option>
          <option value="wheat">Wheat</option>
          <option value="corn">Corn</option>
          <option value="cotton">Cotton</option>
          <option value="potato">Potato</option>
          <option value="soybean">Soybean</option>
          <option value="sugarcane">Sugarcane</option>
          <option value="sunflower">Sunflower</option>
          <option value="tomato">Tomato</option>
          <option value="barley">Barley</option>
        </select>
      </label>

      {/* From Date */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">From Date</span>
        <input
          type="date"
          name="from_date"
          className="input"
          value={form.from_date}
          onChange={handleChange}
        />
      </label>

      {/* To Date */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">To Date</span>
        <input
          type="date"
          name="to_date"
          className="input"
          value={form.to_date}
          onChange={handleChange}
        />
      </label>

      {/* Temperature */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Temperature (°C)</span>
        <input
          name="temperature"
          className="input"
          value={form.temperature}
          onChange={handleChange}
        />
      </label>

      {/* Humidity */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Humidity (%)</span>
        <input
          name="humidity"
          className="input"
          value={form.humidity}
          onChange={handleChange}
        />
      </label>

      {/* Soil pH */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Soil pH</span>
        <input
          name="soilph"
          className="input"
          value={form.soilph}
          onChange={handleChange}
        />
      </label>

      {/* Wind Speed */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Wind Speed (Km/h)</span>
        <input
          name="windspeed"
          className="input"
          value={form.windspeed}
          onChange={handleChange}
        />
      </label>

      {/* Nitrogen */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Nitrogen (Kg/ha)</span>
        <input
          name="n"
          className="input"
          value={form.n}
          onChange={handleChange}
        />
      </label>

      {/* Phosphorus */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Phosphorus (Kg/ha)</span>
        <input
          name="p"
          className="input"
          value={form.p}
          onChange={handleChange}
        />
      </label>

      {/* Potassium */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Potassium (Kg/ha)</span>
        <input
          name="k"
          className="input"
          value={form.k}
          onChange={handleChange}
        />
      </label>

      {/* Soil Quality Index */}
      <label className="flex flex-col">
        <span className="text-sm text-gray-600">Soil Quality Index</span>
        <input
          name="soilquality"
          className="input"
          value={form.soilquality}
          onChange={handleChange}
        />
      </label>

      {/* Submit Button */}
      <div className="col-span-2 flex gap-3 mt-2">
        <button type="submit" className="btn-primary">Forecast</button>
      </div>

      {error && <div className="col-span-2 text-red-600">{JSON.stringify(error)}</div>}
    </form>
  );
}
