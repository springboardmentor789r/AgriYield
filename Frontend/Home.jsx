import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sprout, BarChart3, Clock } from "lucide-react";

// SLIDESHOW IMAGES
import slide1 from "../assets/home/slide1.jpg";
import slide2 from "../assets/home/slide2.jpg";
import slide3 from "../assets/home/slide3.jpg";
import slide4 from "../assets/home/slide4.jpg";
import slide5 from "../assets/home/slide5.jpg";

import card1 from "../assets/home/in-card.jpg";
import card2 from "../assets/home/reg-card.jpg";
import card3 from "../assets/home/ts-card.jpg";


function Home() {
  const navigate = useNavigate();

  // ✅ SLIDESHOW STATE
  const slides = [slide1, slide2, slide3, slide4, slide5];
  const [current, setCurrent] = useState(0);

  // ✅ AUTO SLIDE EFFECT
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrent((prev) => (prev + 1) % slides.length);
    }, 3000); // 3 sec

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ paddingLeft: "260px", padding: "40px" }}>
      
      {/* ================= HERO SECTION ================= */}
      <div
        style={{
          display: "flex",
          gap: "40px",
          padding: "50px",
          borderRadius: "28px",
          background: "linear-gradient(120deg,#0b1f1a,#1f4037)",
          color: "#e6e3db",
          alignItems: "center",
        }}
      >
        {/* LEFT TEXT */}
        <div style={{ flex: 1 }}>
          <h1 style={{ fontSize: "42px", marginBottom: "20px" }}>
            Smart Crop Yield Intelligence
          </h1>

          <p style={{ color: "#a8bfa0", lineHeight: "1.6" }}>
            AI-powered crop yield prediction and agricultural forecasting
            platform to support smart farming decisions.
          </p>

          <button
            onClick={() => navigate("/regression")}
            style={{
              marginTop: "30px",
              padding: "14px 30px",
              background: "#e6e3db",
              color: "#0b1f1a",
              border: "none",
              borderRadius: "14px",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Start Prediction
          </button>
        </div>

        {/* RIGHT SLIDESHOW */}
        <div
          style={{
            flex: 1,
            height: "300px",
            borderRadius: "22px",
            overflow: "hidden",
            boxShadow: "0 20px 40px rgba(0,0,0,0.25)",
          }}
        >
          <img
            src={slides[current]}
            alt="crop"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transition: "0.6s ease",
            }}
          />
        </div>
      </div>

      {/* ================= FEATURES SECTION ================= */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3,1fr)",
          gap: "25px",
          marginTop: "50px",
        }}
      >
        <Feature
          title="Crop Insights"
          desc="Analyze soil, nutrients, and environmental factors affecting crops."
          image={card1}
          icon={<Sprout size={20} />}
          button="Explore"
          onClick={() => navigate("/insights")}
        />

        <Feature
          title="Yield Prediction"
          desc="Predict crop yield using machine learning regression models."
          image={card2}
          icon={<BarChart3 size={20} />}
          button="Predict"
          onClick={() => navigate("/regression")}
        />

        <Feature
          title="Time Series Forecast"
          desc="Forecast future crop yield trends using time-series analysis."
          image={card3}
          icon={<Clock size={20} />}
          button="Forecast"
          onClick={() => navigate("/timeseries")}
        />
      </div>
    </div>
  );
}

/* ================= FEATURE CARD ================= */
/* ================= FEATURE CARD ================= */
function Feature({ title, desc, image, icon, button, onClick }) {
  return (
    <div
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-8px)";
        e.currentTarget.style.boxShadow = "0 35px 70px rgba(0,0,0,0.35)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.boxShadow = "0 25px 50px rgba(0,0,0,0.25)";
      }}
      style={{
        background: "linear-gradient(135deg,#0b1f1a,#1f4037)",
        borderRadius: "24px",
        overflow: "hidden",
        boxShadow: "0 25px 50px rgba(0,0,0,0.25)",
        color: "#e6e3db",
        transition: "all 0.3s ease",
      }}
    >
      {/* IMAGE */}
      <img
        src={image}
        alt={title}
        style={{
          width: "100%",
          height: "180px",
          objectFit: "cover",
        }}
      />

      {/* CONTENT */}
      <div style={{ padding: "24px" }}>
        {/* ICON */}
        <div
          style={{
            width: "44px",
            height: "44px",
            borderRadius: "50%",
            background: "#a8bfa0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#0b1f1a",
            marginBottom: "14px",
            boxShadow: "0 8px 20px rgba(0,0,0,0.3)",
          }}
        >
          {icon}
        </div>

        <h3 style={{ marginBottom: "10px", color: "#e6e3db" }}>
          {title}
        </h3>

        <p style={{ fontSize: "14px", color: "#a8bfa0" }}>
          {desc}
        </p>

        <button
          onClick={onClick}
          style={{
            marginTop: "18px",
            padding: "10px 22px",
            background: "#a8bfa0",
            border: "none",
            borderRadius: "12px",
            fontWeight: "600",
            cursor: "pointer",
            color: "#0b1f1a",
            boxShadow: "0 8px 20px rgba(0,0,0,0.25)",
          }}
        >
          {button}
        </button>
      </div>
    </div>
  );
}


export default Home;
