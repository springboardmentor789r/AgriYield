import { useState } from "react";
import { api } from "../api";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend,
} from "chart.js";
import "chartjs-adapter-date-fns";
import { Line } from "react-chartjs-2";

ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  TimeScale,
  Tooltip,
  Legend
);

const initialRegressors = {
  Temperature: "",
  Humidity: "",
  Wind_Speed: "",
  Soil_pH: "",
  N: "",
  P: "",
  K: "",
  Soil_Quality: "",
};

export default function Forecast() {
  const [mode, setMode] = useState("next"); // 'next' | 'range'

  const [nextDays, setNextDays] = useState(7);
  const [range, setRange] = useState({
    start_date: "",
    end_date: "",
  });

  const [regressors, setRegressors] = useState(initialRegressors);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  function handleRegressorChange(e) {
    const { name, value } = e.target;
    setRegressors((prev) => ({ ...prev, [name]: value }));
  }

  function buildRegressorPayload() {
    return {
      Temperature: parseFloat(regressors.Temperature),
      Humidity: parseFloat(regressors.Humidity),
      Wind_Speed: parseFloat(regressors.Wind_Speed),
      Soil_pH: parseFloat(regressors.Soil_pH),
      N: parseFloat(regressors.N),
      P: parseFloat(regressors.P),
      K: parseFloat(regressors.K),
      Soil_Quality: parseFloat(regressors.Soil_Quality),
    };
  }

  async function handleNextDaysForecast(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const payload = {
        days: Number(nextDays),
        ...buildRegressorPayload(),
      };
      const res = await api.post("/forecast_next_days", payload);
      setResult({ type: "next", data: res.data });
    } catch (err) {
      console.error(err);
      alert("Error fetching next-days forecast.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRangeForecast(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const payload = {
        start_date: range.start_date,
        end_date: range.end_date,
        ...buildRegressorPayload(),
      };
      const res = await api.post("/forecast_range", payload);
      setResult({ type: "range", data: res.data });
    } catch (err) {
      console.error(err);
      alert("Error fetching range forecast.");
    } finally {
      setLoading(false);
    }
  }

  // Build chart data from API result
  let chartData = null;
  if (result && result.data) {
    const list =
      result.type === "next"
        ? result.data.daily_predicted_yield
        : result.data.dates_and_yield;

    const labels = list.map((d) => d.ds);
    const values = list.map((d) => d.predicted_yield);

    chartData = {
      labels,
      datasets: [
        {
          label: "Predicted Yield (t/ha)",
          data: values,
          borderColor: "#16a34a",
          backgroundColor: "rgba(34,197,94,0.15)",
          tension: 0.25,
          fill: true,
          pointRadius: 2,
        },
      ],
    };
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: "time",
        time: {
          unit: "day",
        },
        grid: { display: false },
      },
      y: {
        beginAtZero: false,
        grid: { color: "rgba(148,163,184,0.3)" },
        title: {
          display: true,
          text: "Yield (t/ha)",
        },
      },
    },
    plugins: {
      legend: {
        display: true,
        position: "bottom",
      },
    },
  };

  return (
    <div className="space-y-6 pb-8">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h2 className="text-2xl font-semibold text-emerald-900">
            Time Series Forecasting
          </h2>
          <p className="text-sm text-slate-600 max-w-2xl">
            Use the trained Prophet model to forecast daily crop yield based on
            a fixed set of climate and soil conditions. Choose either the next
            N days or a custom date range.
          </p>
        </div>
      </header>

      {/* Toggle buttons */}
      <div className="inline-flex rounded-full bg-emerald-100 p-1">
        <button
          onClick={() => setMode("next")}
          className={`px-4 py-1.5 text-xs sm:text-sm rounded-full transition ${
            mode === "next"
              ? "bg-white text-emerald-700 shadow"
              : "text-emerald-700/70"
          }`}
        >
          Next N Days
        </button>
        <button
          onClick={() => setMode("range")}
          className={`px-4 py-1.5 text-xs sm:text-sm rounded-full transition ${
            mode === "range"
              ? "bg-white text-emerald-700 shadow"
              : "text-emerald-700/70"
          }`}
        >
          Date Range
        </button>
      </div>

      <section className="bg-white rounded-3xl shadow-soft border border-emerald-50 p-5 sm:p-6 space-y-6">
        {/* Regressor inputs (same for both modes) */}
        <div>
          <h3 className="text-sm font-semibold text-emerald-900 mb-2">
            Climate &amp; Soil Conditions
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            These values will be used for all forecasted days.
          </p>
          <div className="grid md:grid-cols-4 gap-4">
            {[
              ["Temperature", "Temperature (°C)"],
              ["Humidity", "Humidity (%)"],
              ["Wind_Speed", "Wind Speed (km/h)"],
              ["Soil_pH", "Soil pH"],
              ["Soil_Quality", "Soil Quality Index"],
              ["N", "N (kg/ha)"],
              ["P", "P (kg/ha)"],
              ["K", "K (kg/ha)"],
            ].map(([key, label]) => (
              <div key={key}>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  {label}
                </label>
                <input
                  type="number"
                  step="any"
                  name={key}
                  value={regressors[key]}
                  onChange={handleRegressorChange}
                  className="w-full rounded-xl border border-emerald-100 bg-white px-3 py-2 text-xs sm:text-sm focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                  required
                />
              </div>
            ))}
          </div>
        </div>

        {/* Mode specific section */}
        {mode === "next" ? (
          <form onSubmit={handleNextDaysForecast} className="space-y-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Next N Days
                </label>
                <input
                  type="number"
                  min={1}
                  name="days"
                  value={nextDays}
                  onChange={(e) => setNextDays(e.target.value)}
                  className="w-32 rounded-xl border border-emerald-100 bg-white px-3 py-2 text-sm focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center rounded-full bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow-soft hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed transition"
              >
                {loading ? "Forecasting..." : "Forecast Next Days"}
              </button>
            </div>
          </form>
        ) : (
          <form onSubmit={handleRangeForecast} className="space-y-4">
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Start Date
                </label>
                <input
                  type="date"
                  name="start_date"
                  value={range.start_date}
                  onChange={(e) =>
                    setRange((prev) => ({ ...prev, start_date: e.target.value }))
                  }
                  className="rounded-xl border border-emerald-100 bg-white px-3 py-2 text-sm focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                  required
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  End Date
                </label>
                <input
                  type="date"
                  name="end_date"
                  value={range.end_date}
                  onChange={(e) =>
                    setRange((prev) => ({ ...prev, end_date: e.target.value }))
                  }
                  className="rounded-xl border border-emerald-100 bg-white px-3 py-2 text-sm focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center rounded-full bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow-soft hover:bg-emerald-700 disabled:opacity-60 disabled:cursor-not-allowed transition"
              >
                {loading ? "Forecasting..." : "Forecast Range"}
              </button>
            </div>
          </form>
        )}

        {/* Results area */}
        <div className="grid lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-6 items-start">
          <div className="h-60 sm:h-72 lg:h-80 rounded-2xl border border-emerald-100 bg-emerald-50/40 p-3">
            {chartData ? (
              <Line data={chartData} options={chartOptions} />
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-slate-500">
                Run a forecast to see the yield trend chart here.
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-emerald-100 bg-white p-4 sm:p-5">
            <h3 className="text-sm font-semibold text-emerald-900 mb-2">
              Numeric Summary
            </h3>
            {result && result.data ? (
              <>
                <p className="text-sm text-slate-600 mb-3">
                  Average predicted yield for this period:
                </p>
                <p className="text-2xl font-semibold text-emerald-700 mb-3">
                  {result.data.average_predicted_yield.toFixed(2)}{" "}
                  <span className="text-sm text-emerald-600">t/ha</span>
                </p>
                <div className="max-h-48 overflow-y-auto text-xs border-t border-emerald-100 pt-2 space-y-1">
                  {(
                    result.type === "next"
                      ? result.data.daily_predicted_yield
                      : result.data.dates_and_yield
                  ).map((row, idx) => (
                    <div
                      key={idx}
                      className="flex justify-between py-0.5 text-slate-600"
                    >
                      <span>{row.ds}</span>
                      <span className="font-medium">
                        {row.predicted_yield.toFixed(2)} t/ha
                      </span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-500">
                After running a forecast you will see the average yield and the
                predicted value for each day in this panel.
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
