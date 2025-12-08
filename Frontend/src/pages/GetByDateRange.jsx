import { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Header from "../Components/Header";
import Footer from "../Components/Footer";
import Model from "../Components/Model";
import image2 from "../assets/img2.png";

const GetDataByDate = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    start_date: "",
    end_date: "",
    soil_ph: "",
    temperature: "",
    humidity: "",
    wind_speed: "",
    N: "",
    P: "",
    K: "",
    soil_quality: ""
  });

  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [modalMessage, setModalMessage] = useState("");
  const [showModal, setShowModal] = useState(false);

  const fieldRanges = {
    soil_ph: { min: 5.5, max: 8.0, label: "Soil pH" },
    temperature: { min: 5, max: 45, label: "Temperature (°C)" },
    humidity: { min: 30, max: 95, label: "Humidity (%)" },
    wind_speed: { min: 0, max: 120, label: "Wind Speed (km/h)" },
    N: { min: 0, max: 200, label: "Nitrogen (kg/ha)" },
    P: { min: 0, max: 150, label: "Phosphorus (kg/ha)" },
    K: { min: 0, max: 200, label: "Potassium (kg/ha)" },
    soil_quality: { min: 13, max: 70, label: "Soil Quality Index" }
  };

  const validateField = (name, value) => {
    if (fieldRanges[name]) {
      const { min, max, label } = fieldRanges[name];
      if (value === "") return `${label} is required`;
      if (value < min || value > max) return `${label} must be between ${min} and ${max}`;
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

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict_timeseries", {
        start_date: formData.start_date,
        end_date: formData.end_date,
        Soil_pH: parseFloat(formData.soil_ph),
        Temperature: parseFloat(formData.temperature),
        Humidity: parseFloat(formData.humidity),
        Wind_Speed: parseFloat(formData.wind_speed),
        N: parseFloat(formData.N),
        P: parseFloat(formData.P),
        K: parseFloat(formData.K),
        Soil_Quality: parseFloat(formData.soil_quality)
      });

      const avgYield =
        (response.data.min_predicted + response.data.max_predicted) / 2;

      const yieldValue = avgYield.toFixed(2);


      setResult(response.data);
      setModalMessage(`🌾 Your Predicted Crop Yield is: ${yieldValue} tons/acre`);
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
      <div style={{ backgroundImage: `url(${image2})`, backgroundSize: "cover", backgroundPosition: "center" }}>
        <div className="p-10 max-w-xl mx-auto">
          <h2 className="text-2xl font-bold mb-6">Time Series Prediction Parameters</h2>

          <form className="bg-white p-6 rounded-xl shadow-lg space-y-5" onSubmit={handleSubmit}>

            <label className="font-semibold">Start Date</label>
            <input type="date" name="start_date" value={formData.start_date} onChange={handleChange}
              className="w-full p-3 border rounded-lg" required />

            <label className="font-semibold">End Date</label>
            <input type="date" name="end_date" value={formData.end_date} onChange={handleChange}
              className="w-full p-3 border rounded-lg" required />

            {Object.keys(fieldRanges).map((key) => (
              <div key={key} className="flex flex-col">
                <label className="font-semibold mb-1">{fieldRanges[key].label}</label>
                <input
                  type="number"
                  name={key}
                  placeholder={`Enter ${fieldRanges[key].label}`}
                  value={formData[key]}
                  onChange={handleChange}
                  className={`w-full p-3 border rounded-lg ${errors[key] ? "border-red-500" : "border-gray-300"}`}
                />
                {errors[key] && <p className="text-red-600 text-sm mt-1">{errors[key]}</p>}
              </div>
            ))}

            <div className="flex justify-between mt-4">
              <button type="button" onClick={() => navigate("/")}
                className="w-1/2 mr-2 bg-gray-400 text-white py-3 rounded-lg">Back</button>

              <button type="submit"
                className="w-1/2 ml-2 bg-green-600 text-white py-3 rounded-lg">Submit</button>
            </div>
          </form>
        </div>
      </div>

      <Model show={showModal} onClose={() => setShowModal(false)} message={modalMessage} result={result} />
      <Footer />
    </>
  );
};

export default GetDataByDate;
