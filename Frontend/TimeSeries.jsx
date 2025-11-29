import { useState, useRef } from "react";
import api from "../services/api";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { jsPDF } from "jspdf";


ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale);

function TimeSeriesForecast() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [forecast, setForecast] = useState([]); // ALWAYS array
  const [loading, setLoading] = useState(false);
  const chartRef = useRef(null);
  
  
  const handleForecast = async () => {
    if (!startDate || !endDate) {
      alert("Please select both dates");
      return;
    }

    setLoading(true);
    setForecast([]); // reset

    try {
      // ✅ Handle single-date forecast
      const adjustedEndDate =
        startDate === endDate
          ? new Date(new Date(endDate).setDate(new Date(endDate).getDate() + 1))
              .toISOString()
              .split("T")[0]
          : endDate;

      const res = await api.get("/predict/timeseries", {
        params: {
          start_date: startDate,
          end_date: adjustedEndDate,
        },
      });


      const values = res.data?.Predicted_Values;

      // 👉 SAFETY CHECK
      if (!Array.isArray(values)) {
        alert("Backend did not return valid forecast values.");
        setForecast([]);
      } else {
        setForecast(values);
      }
    } catch (err) {
      console.error(err);
      alert("Time series prediction failed");
      setForecast([]);
    } finally {
      setLoading(false);
    }
  };

  // SAFETY: even if something goes wrong, this will be array
  const safeForecast = Array.isArray(forecast) ? forecast : [];

  const chartData = {
    labels: safeForecast.map((_, i) => `Day ${i + 1}`),
    datasets: [
  {
    label: "Yield Forecast",
    data: safeForecast,
    tension: 0.45,

    borderColor: "#1f4037",
    backgroundColor: "rgba(31,64,55,0.25)",
    borderWidth: 3,
    fill: true,

    /* ✅ POINT GLOW */
    pointBackgroundColor: "#1f4037",
    pointBorderColor: "#a8bfa0",
    pointRadius: 4,
    pointHoverRadius: 9,           // 🔥 grows on hover
    pointHoverBorderWidth: 3,
    pointHoverBackgroundColor: "#ffffff",

    /* ✅ GLOW EFFECT */
    pointHoverBorderColor: "#9cffc7",
  },
],

  };
  const downloadPNG = () => {
  const chart = chartRef.current;
  if (!chart) return;

  const url = chart.toBase64Image();
  const link = document.createElement("a");
  link.href = url;
  link.download = "timeseries_forecast.png";
  link.click();
};

const downloadPDF = () => {
  const chart = chartRef.current;
  if (!chart) return;

  const imgData = chart.toBase64Image();
  const pdf = new jsPDF("landscape");
  pdf.text("Time Series Yield Forecast", 14, 10);
  pdf.addImage(imgData, "PNG", 10, 20, 260, 120);
  pdf.save("timeseries_forecast.pdf");
};

  return (
    <div style={{ paddingLeft: "260px", padding: "40px" }}>
      {/* SECTION 1: INPUTS */}
      <div
        style={{
          background: "linear-gradient(135deg,#102e26,#1f4037)",
          borderRadius: "28px",
          padding: "36px",
          boxShadow: "0 30px 60px rgba(0,0,0,0.3)",
        }}
      >
        <h2 style={{ color: "#e6e3db" }}>Time Series Yield Forecast</h2>
        <p style={{ color: "#a8bfa0", marginBottom: "26px" }}>
          Select a date range to forecast crop yield trends.  
          (For a single date, choose the same start and end date.)
        </p>

        {/* INPUT CARD */}
        <div
          style={{
            background: "#dbe8e0",
            padding: "26px",
            borderRadius: "22px",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "22px",
            overflow: "visible",
          }}
        >
          <div>
            <label style={labelStyle}>Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={inputStyle}
            />
          </div>

          <div>
            <label style={labelStyle}>End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              style={inputStyle}
            />
          </div>
        </div>

        {/* BUTTON */}
        <button
          onClick={handleForecast}
          style={{
            marginTop: "28px",
            width: "100%",
            padding: "14px",
            background: "linear-gradient(90deg,#9ab89f,#7da88a)",
            border: "none",
            borderRadius: "14px",
            fontWeight: "600",
            fontSize: "15px",
            cursor: "pointer",
            color: "#0b1f1a",
          }}
        >
          {loading ? "Forecasting..." : "Generate Forecast"}
        </button>

        {/* SECTION 2: RESULT (CHART + VALUES) */}
        {safeForecast.length > 0 && (
          <>
            {/* CHART CARD */}
            <div
              style={{
                marginTop: "32px",
                background: "#f7faf8",
                padding: "24px",
                borderRadius: "22px",
                height: "340px",
              }}
            >
              <Line
                ref={chartRef}
                data={chartData}
                options={{
                  interaction: {
  intersect: false,
  mode: "nearest",
},

animations: {
  radius: {
    duration: 300,
    easing: "easeOutQuad",
  },
  borderWidth: {
    duration: 300,
    easing: "easeOutQuad",
  },
},

  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: "#355f54",
        font: { weight: "600" },
      },
    },

    /* ✅ BEAUTIFIED TOOLTIP */
    tooltip: {
      backgroundColor: "#1f4037",   // 🌿 dark green
      titleColor: "#e6efe9",
      bodyColor: "#ffffff",
      borderColor: "#a8bfa0",
      borderWidth: 1,
      cornerRadius: 10,
      padding: 12,

      titleFont: {
        size: 14,
        weight: "600",
      },
      bodyFont: {
        size: 14,
        weight: "500",
      },

      callbacks: {
        label: function (context) {
          return ` Yield: ${context.formattedValue}`;
        },
      },
    },
  },

  scales: {
    x: {
      ticks: { color: "#355f54" },
      grid: { color: "rgba(53,95,84,0.15)" },
    },
    y: {
      ticks: { color: "#355f54" },
      grid: { color: "rgba(53,95,84,0.15)" },
    },
  },
}}

              />
            </div>
            <div
  style={{
    marginTop: "16px",
    display: "flex",
    gap: "14px",
    justifyContent: "flex-end",
  }}
>
  <button onClick={downloadPNG} style={downloadBtn}>
    Download PNG
  </button>

  <button onClick={downloadPDF} style={downloadBtn}>
    Download PDF
  </button>
</div>

            {/* VALUES CARD */}
            <div
              style={{
                marginTop: "24px",
                background: "#eaf3ff",
                padding: "24px",
                borderRadius: "18px",
              }}
            >
              <h3
                style={{
                  marginBottom: "12px",
                  color: "#1f4037",
                  fontWeight: "600",
                }}
              >
                Total Days: {safeForecast.length}
              </h3>

              <pre
                style={{
                  background: "#f5faff",
                  padding: "16px",
                  borderRadius: "12px",
                  maxHeight: "200px",
                  overflowY: "auto",
                  color: "#0b1f1a",
                  fontSize: "14px",
                  fontFamily: "monospace",
                  lineHeight: "1.6",
                }}
              >
                {JSON.stringify(safeForecast, null, 2)}
              </pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ===== STYLES ===== */
const labelStyle = {
  fontSize: "13px",
  fontWeight: "500",
  color: "#355f54",
};

const inputStyle = {
  width: "100%",
  height: "38px",
  padding: "4px 10px",
  borderRadius: "10px",
  border: "1.5px solid #7ca99b",
  background: "#f7faf8",
  color: "#0b1f1a",
  boxSizing: "border-box",
  fontSize: "14px",
};

const downloadBtn = {
  padding: "10px 18px",
  borderRadius: "10px",
  border: "none",
  background: "linear-gradient(90deg,#9ab89f,#7da88a)",
  color: "#0b1f1a",
  fontWeight: "600",
  cursor: "pointer",
};

export default TimeSeriesForecast;
