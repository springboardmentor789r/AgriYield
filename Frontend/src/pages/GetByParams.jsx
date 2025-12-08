import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../Components/Header";
import Footer from "../Components/Footer";
import Model from "../Components/Model";
import image1 from "../assets/image.png";

const GetDataByParams = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    soil_type: "",
    crop_type: "",
    soil_pH: "",
    temperature: "",
    humidity: "",
    wind_speed: "",
    n: "",
    p: "",
    k: "",
    soil_quality: "",
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [modalMessage, setModalMessage] = useState("");
  const [showModal, setShowModal] = useState(false);

  const fieldRanges = {
    soil_pH: { min: 5.5, max: 8.0, label: "Soil pH" },
    temperature: { min: 5, max: 45, label: "Temperature (°C)" },
    humidity: { min: 30, max: 95, label: "Humidity (%)" },
    wind_speed: { min: 0, max: 120, label: "Wind Speed (km/h)" },
    n: { min: 0, max: 200, label: "Nitrogen (kg/ha)" },
    p: { min: 0, max: 150, label: "Phosphorus (kg/ha)" },
    k: { min: 0, max: 200, label: "Potassium (kg/ha)" },
    soil_quality: { min: 13, max: 70, label: "Soil Quality Index" },
  };

  const validateField = (name, value) => {
    if (fieldRanges[name]) {
      const { min, max, label } = fieldRanges[name];
      if (value === "") return `${label} is required`;
      if (value < min || value > max)
        return `${label} must be between ${min} and ${max}`;
    }
    return "";
  };

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({ ...formData, [name]: value });

    const errorMessage = validateField(name, value);
    setErrors((prev) => ({ ...prev, [name]: errorMessage }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    let newErrors = {};
    Object.keys(formData).forEach((key) => {
      const msg = validateField(key, formData[key]);
      if (msg) newErrors[key] = msg;
    });

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) return;

    const payload = {
      soil_type: formData.soil_type,
      crop_type: formData.crop_type,
      soil_pH: parseFloat(formData.soil_pH),
      temperature: parseFloat(formData.temperature),
      humidity: parseFloat(formData.humidity),
      wind_speed: parseFloat(formData.wind_speed),
      n: parseFloat(formData.n),
      p: parseFloat(formData.p),
      k: parseFloat(formData.k),
      soil_quality: parseFloat(formData.soil_quality),
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/predict_regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      setResult(data);

      if (data.predicted_crop_yield !== undefined) {
        setModalMessage("Your Predicted Crop Yield is:");
      }

      setShowModal(true);
    } catch (err) {
      console.error(err);
      setModalMessage("Error occurred while fetching prediction!");
      setShowModal(true);
    }
  };

  return (
    <>
      <Header />

      <div
        className="min-h-screen"
        style={{ backgroundImage: `url(${image1})`, backgroundSize: "cover", backgroundPosition: "center" }}
      >
        <div className="p-10 max-w-xl mx-auto">
          <h2 className="text-2xl font-bold mb-6">Enter Parameters</h2>

          <form className="bg-white p-6 rounded-xl shadow-lg space-y-5" onSubmit={handleSubmit}>
  
  {/* Soil Type */}
  <div className="flex flex-col">
    <label className="font-semibold mb-1">Soil Type</label>
    <select
      name="soil_type"
      value={formData.soil_type}
      onChange={handleChange}
      className="w-full p-3 border rounded-lg"
      required
    >
      <option value="">Select Soil Type</option>
      <option value="Saline">Saline</option>
      <option value="Clay">Clay</option>
      <option value="Loamy">Loamy</option>
      <option value="Peaty">Peaty</option>
      <option value="Sandy">Sandy</option>
    </select>
  </div>

  {/* Crop Type */}
  <div className="flex flex-col">
    <label className="font-semibold mb-1">Crop Type</label>
    <select
      name="crop_type"
      value={formData.crop_type}
      onChange={handleChange}
      className="w-full p-3 border rounded-lg"
      required
    >
      <option value="">Select Crop Type</option>
      <option value="Corn">Corn</option>
      <option value="Barley">Barley</option>
      <option value="Soybean">Soybean</option>
      <option value="Cotton">Cotton</option>
      <option value="Tomato">Tomato</option>
      <option value="Potato">Potato</option>
      <option value="Sunflower">Sunflower</option>
      <option value="Wheat">Wheat</option>
      <option value="Sugarcane">Sugarcane</option>
      <option value="Rice">Rice</option>
    </select>
  </div>


            {Object.keys(fieldRanges).map((key) => (
              <div key={key} className="flex flex-col">
                <label className="font-semibold mb-1">{fieldRanges[key].label}</label>
                <input
                  type="number"
                  name={key}
                  placeholder={`Enter ${fieldRanges[key].label}`}
                  value={formData[key]}
                  onChange={handleChange}
                  className={`w-full p-3 border rounded-lg ${
                    errors[key] ? "border-red-500" : "border-gray-300"
                  }`}
                />
                {errors[key] && <p className="text-red-600 text-sm mt-1">{errors[key]}</p>}
              </div>
            ))}

            <div className="flex justify-between mt-4">
              <button
                type="button"
                onClick={() => navigate("/")}
                className="w-1/2 mr-2 bg-gray-400 text-white py-3 rounded-lg"
              >
                Back
              </button>
              <button type="submit" className="w-1/2 ml-2 bg-green-600 text-white py-3 rounded-lg">
                Submit
              </button>
            </div>
          </form>
        </div>
      </div>

      <Model show={showModal} onClose={() => setShowModal(false)} message={modalMessage} result={result} />
      <Footer />
    </>
  );
};

export default GetDataByParams;
