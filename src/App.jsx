// App.jsx
import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [form, setForm] = useState({
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
  const [errMsg, setErrMsg] = useState("");

  function update(e) {
    setForm({ ...form, [e.target.name]: e.target.value });
  }

  function buildPayload() {
    return {
      Date: new Date().toISOString().split("T")[0],
      Crop_Type: form.Crop_Type?.toLowerCase() || null,
      Soil_Type: form.Soil_Type?.toLowerCase() || null,
      Soil_pH: form.Soil_pH ? Number(form.Soil_pH) : null,
      Temperature: form.Temperature ? Number(form.Temperature) : null,
      Humidity: form.Humidity ? Number(form.Humidity) : null,
      Wind_Speed: form.Wind_Speed ? Number(form.Wind_Speed) : null,
      N: form.N ? Number(form.N) : null,
      P: form.P ? Number(form.P) : null,
      K: form.K ? Number(form.K) : null,
      Soil_Quality: form.Soil_Quality ? Number(form.Soil_Quality) : null,
    };
  }

  async function predict(e) {
    e?.preventDefault();
    setErrMsg("");
    setResult(null);
    setLoading(true);

    try {
      const res = await axios.post(
        "http://127.0.0.1:5000/predict",
        buildPayload()
      );
      setResult(res.data);
    } catch (err) {
      setErrMsg("Prediction failed ❌ " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* 🌿 NAVBAR */}
      <header className="navbar">
        <div className="navbar-inner">
          <div className="logo">🌾 Crop Yield AI</div>

          <nav className="nav-links">
            <a className="active">Predict</a>
            <a>About</a>
            <a>Help</a>
          </nav>
        </div>
      </header>

      {/* 🌿 PAGE WRAPPER */}
      <main className="page">
        <div className="content-wrapper">

          {/* 🌿 HERO SECTION */}
         <section className="hero">
  <div className="hero-inner">
    <h1>🌱 Crop Yield Prediction</h1>
    <p>Enter soil & weather parameters to estimate expected yield</p>
  </div>
</section>


          {/* 🌿 FORM SECTION */}
          <section className="content">
            <div className="content-inner">
              <div className="form-box">

                <form className="form-grid" onSubmit={predict}>
                  <div>
                    <label>Crop Type</label>
                    <select
                      name="Crop_Type"
                      value={form.Crop_Type}
                      onChange={update}
                    >
                      <option value="">Select Crop</option>
                      <option value="rice">Rice</option>
                      <option value="wheat">Wheat</option>
                      <option value="maize">Maize</option>
                    </select>
                  </div>

                  <div>
                    <label>Soil Type</label>
                    <select
                      name="Soil_Type"
                      value={form.Soil_Type}
                      onChange={update}
                    >
                      <option value="">Select Soil</option>
                      <option value="clay">Clay</option>
                      <option value="black">Black</option>
                      <option value="loamy">Loamy</option>
                    </select>
                  </div>

                  <div>
                    <label>Soil pH</label>
                    <input
                      type="number"
                      step="0.1"
                      name="Soil_pH"
                      value={form.Soil_pH}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Temperature (°C)</label>
                    <input
                      type="number"
                      name="Temperature"
                      value={form.Temperature}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Humidity (%)</label>
                    <input
                      type="number"
                      name="Humidity"
                      value={form.Humidity}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Wind Speed (km/h)</label>
                    <input
                      type="number"
                      name="Wind_Speed"
                      value={form.Wind_Speed}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Nitrogen (N)</label>
                    <input
                      type="number"
                      name="N"
                      value={form.N}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Phosphorus (P)</label>
                    <input
                      type="number"
                      name="P"
                      value={form.P}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Potassium (K)</label>
                    <input
                      type="number"
                      name="K"
                      value={form.K}
                      onChange={update}
                    />
                  </div>

                  <div>
                    <label>Soil Quality Index</label>
                    <input
                      type="number"
                      name="Soil_Quality"
                      value={form.Soil_Quality}
                      onChange={update}
                    />
                  </div>
                </form>

                {/* 🌿 BUTTON */}
                <div className="actions">
                  <button
                    className="predict-btn"
                    onClick={predict}
                    disabled={loading}
                  >
                    {loading ? "Predicting…" : "Predict Yield"}
                  </button>
                </div>

                {/* 🌿 ERROR BOX */}
                {errMsg && <div className="result-box error">{errMsg}</div>}

                {/* 🌿 RESULT */}
                {result && (
                  <div className="result-box">
                    <h3>Predicted Yield</h3>
                    <div className="result-values">
                      <div>
                        <strong>{result.predicted_q_per_ha}</strong> q/ha
                      </div>
                      <div>
                        <strong>{result.predicted_kg_per_ha}</strong> kg/ha
                      </div>
                    </div>

                    <pre className="raw-json">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          </section>

        </div>
      </main>

      {/* 🌿 FOOTER */}
      <footer className="site-footer">
        <div className="footer-inner">
          © {new Date().getFullYear()} Crop Yield AI • Built with 💚
        </div>
      </footer>
    </>
  );
}
