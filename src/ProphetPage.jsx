import { useState } from "react";

export default function ProphetPage() {
  const [startDate, setStartDate] = useState("");
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setForecast(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict-prophet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_date: startDate }),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setForecast(data);
    } catch (err) {
      setError("Failed to fetch forecast. Check backend and input.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="bg-slate-900/70 border border-slate-800 rounded-2xl shadow-xl shadow-black/40 p-6">
      <h2 className="text-xl font-semibold mb-1">Prophet 3‑Month Forecast</h2>
      <p className="text-sm text-slate-400 mb-4">
        Choose a start date to generate daily crop yield forecast for 90 days.
      </p>

      <form
        onSubmit={handleSubmit}
        className="flex flex-wrap items-end gap-3 mb-4"
      >
        <div className="flex flex-col gap-1">
          <label
            htmlFor="start-date"
            className="text-xs font-medium text-slate-300"
          >
            Start date
          </label>
          <input
            id="start-date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            required
            className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="inline-flex items-center justify-center rounded-full bg-emerald-500 px-5 py-2 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/40 hover:bg-emerald-400 disabled:opacity-60 disabled:cursor-not-allowed transition"
        >
          {loading ? "Loading..." : "Submit"}
        </button>
      </form>

      {forecast && (
        <div className="mt-3 max-h-80 overflow-auto rounded-xl border border-slate-800">
          <table className="min-w-full text-sm text-left text-slate-200">
            <thead className="sticky top-0 bg-slate-900">
              <tr>
                <th className="px-4 py-2 border-b border-slate-800">Date</th>
                <th className="px-4 py-2 border-b border-slate-800">
                  Crop Yield
                </th>
              </tr>
            </thead>
            <tbody>
              {forecast.map((item, idx) => (
                <tr
                  key={idx}
                  className={idx % 2 === 0 ? "bg-slate-900/40" : "bg-slate-900/20"}
                >
                  <td className="px-4 py-2 border-b border-slate-800">
                    {item.date}
                  </td>
                  <td className="px-4 py-2 border-b border-slate-800">
                    {item.predicted_crop_yield.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
