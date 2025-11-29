import { useState } from "react";
import api from "../services/api";

function Regression() {
  const [formData, setFormData] = useState({
    Crop_Type: "",
    Soil_Type: "",
    Soil_pH: "",
    Temperature: "",
    Humidity: "",
    Wind_Speed: "",
    N: "",
    P: "",
    K: "",
    Soil_Quality: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // HANDLE INPUT CHANGE
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // PREDICT
  const handlePredict = async () => {
    setLoading(true);
    setResult(null);

    try {
      const res = await api.post("/predict/regression", {
        ...formData,
        Soil_pH: Number(formData.Soil_pH),
        Temperature: Number(formData.Temperature),
        Humidity: Number(formData.Humidity),
        Wind_Speed: Number(formData.Wind_Speed),
        N: Number(formData.N),
        P: Number(formData.P),
        K: Number(formData.K),
        Soil_Quality: Number(formData.Soil_Quality),
      });

      setResult(res.data.Predicted_Yield.toFixed(2));
    } catch (err) {
      alert("Prediction failed. Check inputs / backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ paddingLeft: "260px", padding: "40px" }}>
      
      {/* PAGE CARD */}
      <div
        style={{
          background: "linear-gradient(135deg,#0b1f1a,#1f4037)",
          borderRadius: "28px",
          padding: "36px",
          color: "#e6e3db",
          boxShadow: "0 30px 60px rgba(0,0,0,0.35)",
        }}
      >
        <h2 style={{ color: "#e6e3db", marginBottom: "8px" }}>
          Crop Yield Prediction
        </h2>
        <p style={{ color: "#a8bfa0", marginBottom: "30px" }}>
          Enter crop, soil, and environmental parameters to predict yield.
        </p>

        {/* FORM GRID */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            columnGap: "32px",   // ✅ proper spacing between columns
            rowGap: "22px",      // ✅ vertical space
            background: "#dbe8e0",   // ✅ light sage
            padding: "28px",
            borderRadius: "22px",
          }}
        >

          <Select
            label="Crop Type"
            name="Crop_Type"
            value={formData.Crop_Type}
            onChange={handleChange}
            options={[
              "Wheat","Rice","Corn","Barley","Soybean",
              "Cotton","Sugarcane","Tomato","Potato","Sunflower",
            ]}
          />

          <Select
            label="Soil Type"
            name="Soil_Type"
            value={formData.Soil_Type}
            onChange={handleChange}
            options={["Peaty","Loamy","Sandy","Saline","Clay"]}
          />

          <Input label="Soil pH" name="Soil_pH" onChange={handleChange} />
          <Input label="Temperature (°C)" name="Temperature" onChange={handleChange} />
          <Input label="Humidity (%)" name="Humidity" onChange={handleChange} />
          <Input label="Wind Speed" name="Wind_Speed" onChange={handleChange} />
          <Input label="Nitrogen (N)" name="N" onChange={handleChange} />
          <Input label="Phosphorus (P)" name="P" onChange={handleChange} />
          <Input label="Potassium (K)" name="K" onChange={handleChange} />
          <Input label="Soil Quality" name="Soil_Quality" onChange={handleChange} />
        </div>

        {/* BUTTON */}
        <button
          onClick={handlePredict}
          style={{
            marginTop: "30px",
            width: "100%",
            padding: "16px",
            background: "#a8bfa0",
            border: "none",
            borderRadius: "14px",
            fontWeight: "600",
            cursor: "pointer",
            color: "#0b1f1a",
          }}
        >
          {loading ? "Predicting..." : "Predict Yield"}
        </button>

        {/* RESULT */}
        {result && (
          <div
            style={{
              marginTop: "20px",
              background: "#0b1f1a",
              borderRadius: "18px",
              padding: "4px",
              textAlign: "center",
              boxShadow: "0 0 20px rgba(168,191,160,0.5)",
            }}
          >
            <h3 style={{ color: "#a8bfa0", marginBottom: "6px" }}>
              Predicted Yield
            </h3>
            <p style={{ fontSize: "32px", fontWeight: "700", color: "#e6e3db" }}>
              {result} 
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ==== INPUT COMPONENTS ==== */

function Input({ label, name, onChange }) {
  return (
    <div>
      <label style={{ fontSize: "13px", color: "#355f54", fontWeight: 500 }}>{label}</label>
      <input
        name={name}
        onChange={onChange}
        style={inputStyle}
      />
    </div>
  );
}

function Select({ label, name, value, onChange, options }) {
  return (
    <div>
      <label style={{ fontSize: "13px", color: "#355f54", fontWeight: 500 }}>{label}</label>
      <select
        name={name}
        value={value}
        onChange={onChange}
        style={inputStyle}
      >
        <option value="">Select</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </div>
  );
}

const inputStyle = {
  width: "100%",
  height: "36px",             
  padding: "4px",        
  borderRadius: "10px",
  border: "1.5px solid #7ca99b",
  background: "#f7faf8",
  color: "#0b1f1a",
  fontSize: "14px",
};


export default Regression;
