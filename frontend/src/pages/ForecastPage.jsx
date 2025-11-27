import React, { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import ForecastChart from "../components/ForecastChart";
import { validateForm } from "../utils/validation";
import { forecastYield } from "../services/api";
// Assuming lucide-react or similar is available for icons
import { TrendingUp, Calendar, FlaskConical, CloudRain, Clock, Loader2 } from 'lucide-react'; 

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];

// Helper component for the page header
const PageHeader = () => (
  <header className="mb-8 border-b border-sky-100 pb-4 animate-slideDown">
    <h1 className="text-3xl font-bold text-sky-700 flex items-center gap-2">
      <TrendingUp size={28} className="text-sky-500" />
      Multi-Month Yield Forecast
    </h1>
    <p className="text-gray-500 mt-1.5 text-base">
      Predict future crop yield trends based on current farm and climate conditions.
    </p>
  </header>
);

// Helper component for the results section
const ForecastResults = ({ forecast }) => (
  <div className="md:col-span-2 lg:col-span-3">
    <div className="bg-white p-6 rounded-xl shadow-lg border border-sky-50 animate-fadeIn h-full">
      <h3 className="text-xl font-semibold text-sky-700 mb-4 flex items-center gap-2 border-b pb-2">
        <Clock size={20} />
        Yield Trend Visualizer
      </h3>
      
      {/* Chart Display */}
      <div className="h-64 mb-6">
         {/* Assuming ForecastChart handles its own sizing */}
        <ForecastChart forecast={forecast} />
      </div>

      {/* Table Display */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm rounded-lg overflow-hidden">
          <thead>
            <tr className="bg-sky-50 text-sky-700 font-semibold border-b-2 border-sky-200">
              <th className="p-3 text-left w-1/3">Month</th>
              <th className="p-3 text-right">Predicted Yield (t/ha)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {forecast.map((f, i) => (
              <tr key={i} className="hover:bg-sky-50 transition-colors">
                <td className="p-3 text-left">{f.Month}</td>
                <td className="p-3 text-right font-medium text-gray-800">
                   {/* Format number for professional look */}
                  {Number(f.Predicted_Yield).toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  </div>
);


export default function ForecastPage() {
  const [form, setForm] = useState({
    Crop_Type: "Cotton",
    Months: "6",
    Soil_pH: "6.5",
    Temperature: "24",
    Humidity: "80",
    Wind_Speed: "17",
    N: "46",
    P: "51",
    K: "62",
  });

  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);

  function update(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));

    if (submitted) {
      const newErrors = validateForm({ ...form, [name]: value });
      setErrors(newErrors);
    }
  }

  async function submit(e) {
    e.preventDefault();

    setSubmitted(true);
    const newErrors = validateForm(form);
    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    // Convert numeric fields before sending to API (Crucial step)
    const payload = { ...form };
    const numericFields = ["Months", "Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K"];
    numericFields.forEach(field => {
        payload[field] = Number(payload[field]);
    });

    setLoading(true);
    setForecast(null); // Clear previous forecast
    try {
        const res = await forecastYield(payload);
        setForecast(res);
    } catch (error) {
        console.error("Forecast failed:", error);
    } finally {
        setLoading(false);
    }
  }
  
  // Array of fields for clean rendering
  const fields = [
    { label: "Crop Type", name: "Crop_Type", type: "select", options: CROP_OPTIONS, icon: TrendingUp },
    { label: "Forecast Period", name: "Months", type: "input", unit: "Months", icon: Calendar },
    
    { label: "Soil pH", name: "Soil_pH", type: "input", unit: "", icon: FlaskConical },
    { label: "Temperature", name: "Temperature", type: "input", unit: "°C", icon: CloudRain },
    { label: "Humidity", name: "Humidity", type: "input", unit: "%", icon: CloudRain },
    { label: "Wind Speed", name: "Wind_Speed", type: "input", unit: "km/h", icon: CloudRain },
    
    { label: "Nitrogen (N)", name: "N", type: "input", unit: "ppm", icon: FlaskConical },
    { label: "Phosphorus (P)", name: "P", type: "input", unit: "ppm", icon: FlaskConical },
    { label: "Potassium (K)", name: "K", type: "input", unit: "ppm", icon: FlaskConical },
  ];

  return (
    // Outer container for background design (assuming you've updated index.css)
    <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8"> 
      
      <div className="bg-white p-6 sm:p-10 rounded-2xl shadow-2xl border border-gray-100">
        
        <PageHeader />

        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-8">
            
            {/* --- Left Column: Input Form (1/3 or 2/5 width) --- */}
            <div className="md:col-span-1 lg:col-span-2">
                <form onSubmit={submit} className="grid grid-cols-2 gap-x-4 gap-y-4 animate-fadeUp p-4 bg-gray-50 rounded-lg border border-gray-100">
                    
                    <h3 className="col-span-2 text-lg font-semibold text-gray-700 border-b pb-2 mb-2">Input Parameters</h3>
                    
                    {fields.map((field) => (
                        <div key={field.name} className={`col-span-2 ${field.name === "Crop_Type" || field.name === "Months" ? 'md:col-span-2' : 'sm:col-span-1'}`}>
                            {field.type === "select" ? (
                                <SelectField
                                label={field.label} name={field.name}
                                options={field.options} value={form[field.name]}
                                onChange={update} error={errors[field.name]}
                                />
                            ) : (
                                <InputField 
                                label={field.label} name={field.name} 
                                value={form[field.name]} onChange={update} 
                                unit={field.unit} error={errors[field.name]} 
                                />
                            )}
                        </div>
                    ))}

                    {/* Submission Button */}
                    <div className="col-span-2 pt-4">
                        <button 
                            className="w-full py-2.5 px-6 bg-sky-600 text-white rounded-lg text-base font-semibold hover:bg-sky-700 transition duration-200 shadow-md shadow-sky-200"
                            disabled={loading}
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <Loader2 className="animate-spin h-5 w-5 text-white" />
                                    Forecasting...
                                </span>
                            ) : (
                                "Generate Forecast"
                            )}
                        </button>
                    </div>
                </form>
            </div>
            
            {/* --- Right Column: Results (2/3 or 3/5 width) --- */}
            {forecast ? (
                <ForecastResults forecast={forecast.Forecast} />
            ) : (
                <div className="md:col-span-2 lg:col-span-3 flex items-center justify-center p-8 bg-gray-50 border border-dashed border-gray-300 rounded-xl animate-fadeIn">
                    <p className="text-gray-500 italic text-lg">
                        Enter parameters and click "Generate Forecast" to view the multi-month yield trend here.
                    </p>
                </div>
            )}
        </div>
      </div>
    </div>
  );
}