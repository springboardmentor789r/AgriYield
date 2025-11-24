import React from 'react';
import { Sprout, Brain, LineChart, Thermometer, Droplets, Wind, CloudRain, TrendingUp } from 'lucide-react';
import FactorCard from '../components/FactorCard';
import StatCard from '../components/StatCard';
import { useNavigate } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div>
      <section className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-800 mb-6">
              Predict Your Crop Yields with <span className="text-green-600">AI Precision</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Leverage advanced machine learning models to forecast agricultural yields based on soil conditions, weather patterns, and crop types.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button onClick={() => navigate('/catboost')} className="bg-green-600 hover:bg-green-700 text-white font-semibold px-8 py-4 rounded-lg shadow-lg">
                Try CatBoost Predictor
              </button>
              <button onClick={() => navigate('/prophet')} className="bg-teal-600 hover:bg-teal-700 text-white font-semibold px-8 py-4 rounded-lg shadow-lg">
                Try Time Series Forecast
              </button>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mt-16">
            <div className="bg-white rounded-xl p-8 shadow-xl">
              <div className="flex items-center mb-4">
                <Brain className="w-12 h-12 text-green-600 mr-4" />
                <h3 className="text-2xl font-bold text-gray-800">CatBoost Model</h3>
              </div>
              <p className="text-gray-600 mb-4">Gradient boosting model analyzing environmental factors to predict yield.</p>
            </div>

            <div className="bg-white rounded-xl p-8 shadow-xl">
              <div className="flex items-center mb-4">
                <LineChart className="w-12 h-12 text-teal-600 mr-4" />
                <h3 className="text-2xl font-bold text-gray-800">Prophet Forecast</h3>
              </div>
              <p className="text-gray-600 mb-4">Time-series forecasting with confidence intervals for planning and trends.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Key Factors */}
      <section className="py-16 px-4 bg-white">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-gray-800 mb-12">Key Factors We Analyze</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <FactorCard icon={<Thermometer />} title="Temperature" description="Optimal heat conditions" />
            <FactorCard icon={<Droplets />} title="Humidity" description="Moisture levels" />
            <FactorCard icon={<Wind />} title="Wind Speed" description="Air circulation" />
            <FactorCard icon={<CloudRain />} title="Soil Quality" description="pH and nutrients" />
            <FactorCard icon={<Sprout />} title="Nitrogen (N)" description="Essential nutrient" />
            <FactorCard icon={<Sprout />} title="Phosphorus (P)" description="Root development" />
            <FactorCard icon={<Sprout />} title="Potassium (K)" description="Overall health" />
            <FactorCard icon={<TrendingUp />} title="Crop Type" description="Species-specific" />
          </div>
        </div>
      </section>

      <section className="py-16 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 text-center">
            <StatCard number="97%" label="Prediction Accuracy" />
            <StatCard number="10" label="Crop Types Supported" />
            <StatCard number="8" label="Environmental Factors" />
          </div>
        </div>
      </section>
    </div>
  );
}
