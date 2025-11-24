import React, { useState } from 'react';
import axios from '../api/axiosIns';
import InputField from '../components/InputField';

export default function ProphetPage() {
  const [formData, setFormData] = useState({
    start_date: '',
    periods: '30',
    Soil_pH: '',
    Temperature: '',
    Humidity: '',
    Wind_Speed: '',
    N: '',
    P: '',
    K: '',
    Soil_Quality: ''
  });

  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const payload = {
        start_date: formData.start_date,
        periods: parseInt(formData.periods, 10),
        Soil_pH: formData.Soil_pH ? parseFloat(formData.Soil_pH) : null,
        Temperature: formData.Temperature ? parseFloat(formData.Temperature) : null,
        Humidity: formData.Humidity ? parseFloat(formData.Humidity) : null,
        Wind_Speed: formData.Wind_Speed ? parseFloat(formData.Wind_Speed) : null,
        N: formData.N ? parseFloat(formData.N) : null,
        P: formData.P ? parseFloat(formData.P) : null,
        K: formData.K ? parseFloat(formData.K) : null,
        Soil_Quality: formData.Soil_Quality ? parseFloat(formData.Soil_Quality) : null,
      };

      const res = await axios.post('/forecast', payload);
      // Expecting backend to return { forecast: [...] }
      setForecast(res.data?.forecast ?? null);
    } catch (err) {
      console.error(err);
      setError('Failed to generate forecast. Please check the backend or network.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">Prophet Time Series Forecast</h1>
          <p className="text-gray-600 text-lg">Generate future yield predictions with confidence intervals</p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <div className="space-y-6">
            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <InputField label="Start Date" name="start_date" type="date" value={formData.start_date} onChange={handleChange} />
              <InputField label="Forecast Periods (days)" name="periods" type="number" value={formData.periods} onChange={handleChange} />
            </div>

            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Environmental Conditions</h3>
              <div className="grid md:grid-cols-2 gap-6">
                {['Soil_pH','Temperature','Humidity','Wind_Speed','N','P','K','Soil_Quality'].map((key) => (
                  <InputField key={key} label={key} name={key} value={formData[key]} onChange={handleChange} />
                ))}
              </div>
            </div>

            <button onClick={handleSubmit} disabled={loading} className="w-full bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-semibold py-4 rounded-lg shadow-lg disabled:opacity-50">
              {loading ? 'Generating Forecast...' : 'Generate Forecast'}
            </button>
          </div>

          {error && <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded"><p className="text-red-700">{error}</p></div>}
        </div>

        {forecast && (
          <div className="bg-white rounded-2xl shadow-2xl p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Forecast Results</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted Yield</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Lower Bound</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Upper Bound</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {forecast.slice(0, 10).map((row, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{new Date(row.ds).toLocaleDateString()}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-teal-600">{row.yhat.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{row.yhat_lower.toFixed(2)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{row.yhat_upper.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {forecast.length > 10 && <p className="text-center text-gray-600 mt-4">Showing first 10 of {forecast.length} forecasted days</p>}
          </div>
        )}
      </div>
    </div>
  );
}
