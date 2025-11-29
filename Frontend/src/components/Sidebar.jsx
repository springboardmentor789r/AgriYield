import React from "react";

export default function Sidebar({ route, setRoute }) {
  return (
    <aside className="w-72 p-6 bg-gradient-to-b from-green-600 to-green-700 text-white min-h-screen">
      <div className="mb-8">
        <div className="w-14 h-14 rounded-full bg-white/20 flex items-center justify-center font-bold text-xl">AY</div>
        <h1 className="text-2xl font-bold mt-4">AgriYield</h1>
        <p className="text-sm text-green-100/80">Crop Yield Forecast</p>
      </div>

      <nav className="space-y-2">
        <button
          onClick={() => setRoute("regression")}
          className={`w-full text-left px-3 py-2 rounded ${route==="regression" ? "bg-white/20" : "bg-white/5"}`}
        >
          📊 Regression
        </button>

        <button
          onClick={() => setRoute("timeseries")}
          className={`w-full text-left px-3 py-2 rounded ${route==="timeseries" ? "bg-white/20" : "bg-white/5"}`}
        >
          ⏳ Time Series Forecast
        </button>
      </nav>

      <div className="mt-8 text-sm text-green-100/80">
        <div><strong>Server</strong></div>
        <div>Backend: <span className="font-mono">http://127.0.0.1:8000</span></div>
        <div>Frontend: <span className="font-mono">http://localhost:5173</span></div>
      </div>
    </aside>
  );
}
