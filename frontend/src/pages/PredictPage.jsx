import { useState } from "react";
import InputField from "../components/InputField";
import SelectField from "../components/SelectField";
import { predictYield } from "../services/api";
import { validateForm } from "../utils/validation";
// Assuming you have lucide-react or similar installed
import { Leaf, Droplets, Gauge, Thermometer, Wind, FlaskConical, CloudRain, Factory } from 'lucide-react'; 

const CROP_OPTIONS = ["Barley","Corn","Cotton","Potato","Rice","Soybean","Sugarcane","Sunflower","Tomato","Wheat"];
const SOIL_OPTIONS = ["Loamy", "Clay", "Sandy", "Saline", "Peaty"];

// Helper component for the form header
const Header = () => (
  <header className="mb-6 pb-3 animate-slideDown border-b border-gray-100">
    <h1 className="text-3xl font-bold text-emerald-700 flex items-center gap-2">
      <Factory size={26} className="text-emerald-500" />
      Yield Prediction
    </h1>
    <p className="text-gray-500 mt-1 text-base">
      Enter your farm parameters for an instant yield prediction.
    </p>
  </header>
);

// Helper component for the predicted result
const ResultDisplay = ({ result }) => (
  <div className="mt-6 p-4 bg-emerald-50 rounded-lg shadow-inner border-l-4 border-emerald-600 animate-fadeIn">
    <h3 className="font-bold text-base text-emerald-700 flex items-center gap-2">
      <Gauge size={18} />
      Predicted Crop Yield:
    </h3>
    <p className="text-3xl font-extrabold text-emerald-900 mt-1">
      {/* Use toFixed(2) for a professional look */}
      {Number(result.Predicted_Yield).toFixed(2)} <span className="text-lg font-medium text-emerald-600">tons/hectare</span>
    </p>
  </div>
);


export default function PredictPage() {
  const [form, setForm] = useState({
    Crop_Type: "",
    Soil_Type: "",
    Soil_pH: "6.5",
    Temperature: "24",
    Humidity: "80",
    Wind_Speed: "15",
    N: "40",
    P: "55",
    K: "59",
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [submitted, setSubmitted] = useState(false);
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

    const payload = { ...form };
    const numericFields = ["Soil_pH","Temperature","Humidity","Wind_Speed","N","P","K"];
    numericFields.forEach(field => {
      payload[field] = Number(payload[field]);
    });

    setLoading(true);
    setResult(null); 
    try {
      const res = await predictYield(payload);
      setResult(res);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  }

  // Group fields for compact 4-column display
  const fields = [
    { label: "Crop Type", name: "Crop_Type", type: "select", options: CROP_OPTIONS, icon: Leaf },
    { label: "Soil Type", name: "Soil_Type", type: "select", options: SOIL_OPTIONS, icon: CloudRain },
    { label: "Soil pH", name: "Soil_pH", type: "input", unit: "", icon: FlaskConical },
    { label: "Temperature", name: "Temperature", type: "input", unit: "°C", icon: Thermometer },
    { label: "Humidity", name: "Humidity", type: "input", unit: "%", icon: Droplets },
    { label: "Wind Speed", name: "Wind_Speed", type: "input", unit: "km/h", icon: Wind },
    { label: "Nitrogen (N)", name: "N", type: "input", unit: "ppm", icon: FlaskConical },
    { label: "Phosphorus (P)", name: "P", type: "input", unit: "ppm", icon: FlaskConical },
    { label: "Potassium (K)", name: "K", type: "input", unit: "ppm", icon: FlaskConical },
  ];

  return (
    // Reduced padding and max-width for compact view
    <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6"> 
      <div className="bg-white p-6 sm:p-8 rounded-2xl shadow-xl border border-gray-100 animate-fadeIn">
        
        <Header />

        <form onSubmit={submit} className="grid grid-cols-1 gap-x-4 gap-y-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 animate-fadeUp">

          {fields.map((field) => (
            <div key={field.name} className="col-span-1">
              {field.type === "select" ? (
                <SelectField
                  label={field.label}
                  name={field.name}
                  options={field.options}
                  value={form[field.name]}
                  onChange={update}
                  error={errors[field.name]}
                />
              ) : (
                <InputField 
                  label={field.label} 
                  name={field.name} 
                  value={form[field.name]} 
                  onChange={update} 
                  unit={field.unit} 
                  error={errors[field.name]}
                />
              )}
            </div>
          ))}

          {/* Submission Button: Centered and full-width on its own row */}
          <div className="col-span-full pt-2">
            <button 
              className="w-full py-2.5 px-6 bg-emerald-600 text-white rounded-lg text-base font-semibold hover:bg-emerald-700 transition duration-200 shadow-md shadow-emerald-200" 
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  Predicting...
                </span>
              ) : (
                "Predict Crop Yield"
              )}
            </button>
          </div>
        </form>

        {result && <ResultDisplay result={result} />}

      </div>
    </div>
  );
}