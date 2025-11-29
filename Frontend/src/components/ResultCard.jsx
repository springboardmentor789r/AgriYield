import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  // Detect regression vs time series
  const isTimeSeries = Array.isArray(result.forecast);

  // REGRESSION: Predicted_Yield comes as a single number
  const avgYield = isTimeSeries
    ? result.forecast.length
      ? (
          result.forecast.reduce((sum, i) => sum + i.predicted_yield, 0) /
          result.forecast.length
        ).toFixed(2)
      : null
    : result.Predicted_Yield
    ? result.Predicted_Yield.toFixed(2)
    : null;

  return (
    <div className="mt-6 p-4 bg-green-50 rounded-lg shadow">
      <h3 className="text-xl font-semibold mb-3">Prediction Summary</h3>

      {/* -------- Average Predicted Yield -------- */}
      <div className="mb-4 p-3 bg-white rounded shadow-sm">
        <h4 className="font-medium">Average Predicted Yield:</h4>
        <p className="text-lg font-bold mt-1">
          {avgYield ? `${avgYield} t/ha` : "No data"}
        </p>
      </div>

      {/* -------- Time Series Forecast Table -------- */}
      {isTimeSeries && (
        <div className="bg-white rounded-lg p-4 shadow">
          <h4 className="font-medium mb-2">Daily Forecast:</h4>

          {result.forecast.map((item, index) => (
            <div
              key={index}
              className="flex justify-between border-b py-2 text-sm"
            >
              <span>{item.date}</span>
              <span>{item.predicted_yield.toFixed(2)} t/ha</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
