import React, { useState } from 'react';
import axios from '../api/axiosIns';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import { Thermometer, Droplets, Wind, TrendingUp } from 'lucide-react';

export default function CatBoostPage() {
    const [formData, setFormData] = useState({
        Crop_Type: '',
        Soil_Type: '',
        Soil_pH: '',
        Temperature: '',
        Humidity: '',
        Wind_Speed: '',
        N: '',
        P: '',
        K: '',
        Soil_Quality: ''
    });

    const CropTypes = ['Corn', 'Barley', 'Soybean', 'Cotton', 'Tomato', 'Potato', 'Sunflower', 'Wheat', 'Sugarcane', 'Rice']
    const SoilTypes = ['Peaty', 'Loamy', 'Sandy', 'Saline', 'Clay']

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = (e) => setFormData(prev => ({ ...prev, [e.target.name]: e.target.value }));

    const handleSubmit = async () => {
        setLoading(true);
        setError(null);

        try {
            // convert numeric fields
            const payload = {
                ...formData,
                Soil_pH: formData.Soil_pH ? parseFloat(formData.Soil_pH) : null,
                Temperature: formData.Temperature ? parseFloat(formData.Temperature) : null,
                Humidity: formData.Humidity ? parseFloat(formData.Humidity) : null,
                Wind_Speed: formData.Wind_Speed ? parseFloat(formData.Wind_Speed) : null,
                N: formData.N ? parseFloat(formData.N) : null,
                P: formData.P ? parseFloat(formData.P) : null,
                K: formData.K ? parseFloat(formData.K) : null,
                Soil_Quality: formData.Soil_Quality ? parseFloat(formData.Soil_Quality) : null,
            };

            const res = await axios.post('/predict_yield', payload);
            setPrediction(res.data?.Predicted_Crop_Yield ?? null);
        } catch (err) {
            console.error('API error', err);
            setError('Failed to get prediction. Please check the backend or network.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen py-12 px-4">
            <div className="max-w-4xl mx-auto">
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-gray-800 mb-4">CatBoost Yield Predictor</h1>
                    <p className="text-gray-600 text-lg">Enter your farm's environmental conditions to get an instant yield prediction</p>
                </div>

                <div className="bg-white rounded-2xl shadow-2xl p-8">
                    <div className="space-y-6">
                        <div className="grid md:grid-cols-2 gap-6">
                            <SelectField label="Crop Type" name="Crop_Type" value={formData.Crop_Type} onChange={handleChange} options={CropTypes}/>

                            <SelectField label="Soil Type" name="Soil_Type" value={formData.Soil_Type} onChange={handleChange} options={SoilTypes}/>

                            <InputField label="Soil pH" name="Soil_pH" type="number" step="0.1" value={formData.Soil_pH} onChange={handleChange} placeholder="6.0 - 8.0" />
                            <InputField label="Temperature (°C)" name="Temperature" type="number" step="0.1" value={formData.Temperature} onChange={handleChange} placeholder="15 - 35" icon={<Thermometer className="w-5 h-5 text-gray-400" />} />
                            <InputField label="Humidity (%)" name="Humidity" type="number" step="0.1" value={formData.Humidity} onChange={handleChange} placeholder="40 - 90" icon={<Droplets className="w-5 h-5 text-gray-400" />} />
                            <InputField label="Wind Speed (km/h)" name="Wind_Speed" type="number" step="0.1" value={formData.Wind_Speed} onChange={handleChange} placeholder="0 - 50" icon={<Wind className="w-5 h-5 text-gray-400" />} />

                            <InputField label="Nitrogen (N)" name="N" type="number" step="0.1" value={formData.N} onChange={handleChange} placeholder="Nitrogen content" />
                            <InputField label="Phosphorus (P)" name="P" type="number" step="0.1" value={formData.P} onChange={handleChange} placeholder="Phosphorus content" />
                            <InputField label="Potassium (K)" name="K" type="number" step="0.1" value={formData.K} onChange={handleChange} placeholder="Potassium content" />
                            <InputField label="Soil Quality" name="Soil_Quality" type="number" step="0.1" value={formData.Soil_Quality} onChange={handleChange} placeholder="Quality index" />
                        </div>

                        <button onClick={handleSubmit} disabled={loading} className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold py-4 rounded-lg shadow-lg disabled:opacity-50">
                            {loading ? 'Calculating...' : 'Predict Crop Yield'}
                        </button>
                    </div>

                    {error && <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded"><p className="text-red-700">{error}</p></div>}

                    {prediction !== null && (
                        <div className="mt-8 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-500 p-6 rounded-xl">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-gray-600 mb-2">Predicted Crop Yield</p>
                                    <p className="text-4xl font-bold text-green-700">{Number(prediction).toFixed(2)} tons/hectare</p>
                                </div>
                                <TrendingUp className="w-16 h-16 text-green-600" />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
