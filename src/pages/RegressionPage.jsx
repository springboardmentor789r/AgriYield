import React, { useState } from "react";
import RegressionForm from "../components/RegressionForm";
import ResultCard from "../components/ResultCard";

export default function RegressionPage(){
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  return (
    <>
      <h2 className="text-2xl font-semibold mb-4">Predict Crop Yield</h2>
      <div className="card">
        <RegressionForm setResult={setResult} setLoading={setLoading} />
        {loading && <div className="mt-4"><div className="spinner" /></div>}
        {result && <ResultCard result={result} />}
      </div>
    </>
  );
}
