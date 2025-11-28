// src/pages/Home.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

import ProphetChart from "../assets/prophet_forecast.png";
import ComparisonChart from "../assets/comparison.png";

const Home = () => {
  const navigate = useNavigate();

  const handleStartPrediction = () => {
    navigate("/predict");
  };

  const handleOpenForecast = () => {
    navigate("/forecast");
  };

  return (
    <div className="flex-1 bg-gradient-to-b from-emerald-50 to-emerald-100 min-h-screen px-4 py-6 md:px-8 lg:px-12 overflow-y-auto">
      <div className="max-w-6xl mx-auto space-y-8 lg:space-y-10">
        {/* Top label */}
        <div className="mt-2">
          <span className="inline-flex items-center rounded-full bg-emerald-100 px-3 py-1 text-xs md:text-sm font-medium text-emerald-800">
            ● SMART FARMING · DATA DRIVEN
          </span>
        </div>

        {/* Hero + actions */}
        <div className="space-y-6">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-semibold leading-tight text-emerald-950 font-[Poppins]">
            Predict Crop Yield &{" "}
            <span className="text-emerald-700">Plan Your Farm</span>{" "}
            Production with AI Power
          </h1>

          <p className="max-w-3xl text-sm md:text-base text-emerald-900/80 leading-relaxed">
            AgriYield Predictor combines traditional agronomy with machine
            learning. Enter soil and weather conditions to estimate yield for a
            specific field, or use the time-series module to understand yield
            trends over days and seasons.
          </p>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleStartPrediction}
              className="inline-flex items-center justify-center rounded-full bg-emerald-700 px-6 py-2.5 text-sm md:text-base font-medium text-white shadow-md shadow-emerald-700/30 hover:bg-emerald-800 transition-colors"
            >
              Start Yield Prediction
            </button>
            <button
              onClick={handleOpenForecast}
              className="inline-flex items-center justify-center rounded-full border border-emerald-300 bg-white/70 px-6 py-2.5 text-sm md:text-base font-medium text-emerald-800 hover:bg-emerald-50 transition-colors"
            >
              Open Time Forecast
            </button>
          </div>
        </div>

        {/* Main content: left info cards + right charts */}
        <div className="grid gap-8 xl:gap-10 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] items-start">
          {/* LEFT SIDE – Info cards */}
          <div className="space-y-4 md:space-y-5">
            <div className="grid gap-4 md:grid-cols-2">
              {/* Card 1 */}
              <div className="rounded-2xl bg-white/80 shadow-sm shadow-emerald-100 border border-emerald-100 px-4 py-4 md:px-5 md:py-5">
                <h3 className="text-sm md:text-base font-semibold text-emerald-900 mb-1.5">
                  About AgriYield Predictor
                </h3>
                <p className="text-xs md:text-sm text-emerald-900/80 leading-relaxed">
                  AI-powered tool to help farmers, agronomists, and students
                  explore how soil, weather and nutrients influence crop yield.
                </p>
              </div>

              {/* Card 2 */}
              <div className="rounded-2xl bg-white/80 shadow-sm shadow-emerald-100 border border-emerald-100 px-4 py-4 md:px-5 md:py-5">
                <h3 className="text-sm md:text-base font-semibold text-emerald-900 mb-1.5">
                  Why Crop Yield Matters
                </h3>
                <p className="text-xs md:text-sm text-emerald-900/80 leading-relaxed">
                  Yield estimation supports better planning for inputs, storage,
                  and market decisions, reducing risk and improving profit.
                </p>
              </div>
            </div>

            {/* Card 3 */}
            <div className="rounded-2xl bg-white/80 shadow-sm shadow-emerald-100 border border-emerald-100 px-4 py-4 md:px-5 md:py-5">
              <h3 className="text-sm md:text-base font-semibold text-emerald-900 mb-1.5">
                Climate + Soil Factors
              </h3>
              <p className="text-xs md:text-sm text-emerald-900/80 leading-relaxed">
                Temperature, humidity, wind speed, soil type / quality and
                N-P-K nutrients together shape final production for crops such
                as wheat, rice, corn, tomato, potato and more.
              </p>
            </div>
          </div>

          {/* RIGHT SIDE – Demo Snapshot */}
          <div className="rounded-3xl bg-white/90 shadow-md shadow-emerald-100 border border-emerald-100 p-4 md:p-5 lg:p-6">
            <div className="flex items-center justify-between mb-3 md:mb-4">
              <div>
                <p className="text-[11px] md:text-xs font-semibold tracking-wide text-emerald-600 uppercase">
                  Demo Snapshot
                </p>
                <p className="text-xs md:text-sm font-medium text-emerald-950">
                  Yield vs Time (t/ha)
                </p>
              </div>
              <span className="rounded-full bg-emerald-50 px-3 py-1 text-[11px] md:text-xs font-medium text-emerald-700 border border-emerald-100">
                AI Forecast
              </span>
            </div>

            <div className="space-y-4 md:space-y-5">
              {/* Prophet chart */}
              <div className="rounded-2xl bg-emerald-50/70 border border-emerald-100 overflow-hidden">
                <img
                  src={ProphetChart}
                  alt="Prophet crop yield forecast chart"
                  className="w-full h-44 md:h-52 lg:h-56 object-contain bg-white"
                />
              </div>

              {/* Comparison chart */}
              <div className="rounded-2xl bg-emerald-50/70 border border-emerald-100 overflow-hidden">
                <img
                  src={ComparisonChart}
                  alt="Comparison of ARIMA, ARIMAX and Prophet forecasts"
                  className="w-full h-44 md:h-52 lg:h-56 object-contain bg-white"
                />
              </div>

              {/* Caption + Kaggle link */}
              <p className="text-[10px] md:text-xs text-emerald-800/80 leading-relaxed">
                Charts are generated using the{" "}
                <a
                  href="https://www.kaggle.com/datasets/madhankumar789/crop-yield-and-environmental-factors-2014-2023/data"
                  target="_blank"
                  rel="noreferrer"
                  className="underline underline-offset-2 text-emerald-700 hover:text-emerald-900"
                >
                  Crop Yield (2014–2023) dataset from Kaggle.
                </a>
                <br /> Use the <span className="font-medium">Time Forecast</span>{" "}
                section to generate live predictions from your own inputs.
              </p>

              {/* Tiny footer note inside card */}
              <p className="text-[9px] md:text-[10px] text-emerald-700/80 pt-1 border-t border-emerald-100 mt-1">
                Built by <span className="font-semibold">Farhan Abid</span>
                 <br />Live project for Infosys Springboard Virtual
                Internship&nbsp;6.0&nbsp;(2025).
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
