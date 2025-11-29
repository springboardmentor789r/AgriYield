import React, { useState } from "react";
import TimeSeriesForm from "../components/TimeSeriesForm";
import ResultCard from "../components/ResultCard";
import TimeSeriesChart from "../components/TimeSeriesChart";

export default function TimeSeriesPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <>
      <h2 className="text-2xl font-semibold mb-4">Time Series Forecast</h2>

      <div className="card">
        <TimeSeriesForm setResult={setResult} setLoading={setLoading} />

        {loading && (
          <div className="mt-4">
            <div className="spinner" />
          </div>
        )}

        {result && (
          <>
            <ResultCard result={result} />

            <div className="mt-6">
              <TimeSeriesChart forecast={result.forecast} />
            </div>
          </>
        )}
      </div>
    </>
  );
}
