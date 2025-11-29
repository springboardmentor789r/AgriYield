import { TrendingUp, Leaf } from "lucide-react";

import field1 from "../assets/insights/field1.jpg";
import field2 from "../assets/insights/field2.jpg";

import chart1 from "../assets/insights/chart1.png";
import chart2 from "../assets/insights/chart2.png";
import chart3 from "../assets/insights/chart3.png";
import chart4 from "../assets/insights/chart4.png";
import chart5 from "../assets/insights/chart5.png";
import chart6 from "../assets/insights/chart6.png";

function Insights() {
  return (
    <div style={{ paddingLeft: "260px", padding: "40px" }}>
      <div
        style={{
          background: "linear-gradient(135deg,#102e26,#1f4037)",
          borderRadius: "28px",
          padding: "40px",
          boxShadow: "0 30px 60px rgba(0,0,0,0.35)",
        }}
      >
        <h2 style={{ color: "#e6e3db", marginBottom: "10px" }}>
          Crop Insights
        </h2>

        <p style={{ color: "#a8bfa0", marginBottom: "30px" }}>
          Explore agricultural trends, crop analytics, and real-world farming insights.
        </p>

        {/* ================= SECTION 1 ================= */}
        <Section
          title="Yield & Environmental Analytics"
          icon={<TrendingUp size={20} />}
        >
          <ChartCard
            image={chart1}
            title="Crop Yield Distribution by Soil Type"
            desc="Displays the percentage contribution of each soil type to total crop yield."
          />

          <ChartCard
            image={chart2}
            title="Average Soil pH by Soil Type"
            desc="Compares average pH levels across different soil categories."
          />

          <ChartCard
            image={chart3}
            title="Temperature Distribution"
            desc="Represents temperature variations influencing crop growth."
          />

          <ChartCard
            image={chart4}
            title="Correlation Between Features"
            desc="Shows relationships between soil nutrients, climate factors, and yield."
          />

          <ChartCard
            image={chart5}
            title="Average Crop Yield per Crop Type"
            desc="Highlights yield performance across major crop varieties."
          />

          <ChartCard
            image={chart6}
            title="Humidity vs Crop Yield"
            desc="Analyzes how humidity levels impact crop productivity."
          />
        </Section>

        {/* ================= SECTION 2 ================= */}
        <Section
          title="Field & Soil Observations"
          icon={<Leaf size={20} />}
        >
          <ImageCard
            image={field1}
            title="Soil Quality"
            desc="Observations related to soil nutrients and their effect on crops."
          />

          <ImageCard
            image={field2}
            title="Crop Health"
            desc="Visual field-level insights into crop growth conditions."
          />
        </Section>

        {/* ================= SECTION 3 ================= */}
        <Section
          title="Agricultural Knowledge & News"
          icon={<TrendingUp size={20} />}
        >
          <Article
            title="AI in Agriculture"
            content="Artificial Intelligence enables precision farming by optimizing agricultural decisions and improving yield predictions."
          />

          <Article
            title="Climate Change & Crops"
            content="Changing climate patterns directly impact soil health, water availability, and crop productivity."
          />
        </Section>
      </div>
    </div>
  );
}

/* ================= COMPONENTS ================= */

function Section({ title, icon, children }) {
  return (
    <div style={{ marginBottom: "42px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "14px",
          marginBottom: "20px",
        }}
      >
        <div
          style={{
            width: "42px",
            height: "42px",
            borderRadius: "50%",
            background: "#a8bfa0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#0b1f1a",
            boxShadow: "0 8px 20px rgba(0,0,0,0.3)",
          }}
        >
          {icon}
        </div>

        <h3 style={{ color: "#e6e3db", margin: 0 }}>
          {title}
        </h3>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2,1fr)",
          gap: "22px",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function ChartCard({ image, title, desc }) {
  return (
    <div
      style={{
        background: "#dbe8e0",
        padding: "18px",
        borderRadius: "20px",
        boxShadow: "0 15px 35px rgba(0,0,0,0.2)",
      }}
    >
      <img
        src={image}
        alt={title}
        style={{
          width: "100%",
          height: "220px",
          objectFit: "contain",
          borderRadius: "14px",
          marginBottom: "12px",
          background: "#ffffff",
        }}
      />

      <h4 style={{ color: "#1f4037", marginBottom: "6px" }}>
        {title}
      </h4>

      <p style={{ fontSize: "14px", color: "#355f54" }}>
        {desc}
      </p>
    </div>
  );
}

function ImageCard({ image, title, desc }) {
  return (
    <div
      style={{
        background: "#dbe8e0",
        borderRadius: "20px",
        overflow: "hidden",
        boxShadow: "0 15px 35px rgba(0,0,0,0.2)",
      }}
    >
      <img
        src={image}
        alt={title}
        style={{
          width: "100%",
          height: "200px",
          objectFit: "cover",
        }}
      />

      <div style={{ padding: "16px" }}>
        <h4 style={{ color: "#1f4037", marginBottom: "6px" }}>
          {title}
        </h4>

        <p style={{ fontSize: "14px", color: "#355f54" }}>
          {desc}
        </p>
      </div>
    </div>
  );
}

function InsightCard({ title, desc }) {
  return (
    <div
      style={{
        background: "#dbe8e0",
        padding: "22px",
        borderRadius: "18px",
        boxShadow: "0 12px 25px rgba(0,0,0,0.15)",
      }}
    >
      <h4 style={{ color: "#1f4037", marginBottom: "8px" }}>
        {title}
      </h4>

      <p style={{ fontSize: "14px", color: "#355f54" }}>
        {desc}
      </p>
    </div>
  );
}

function Article({ title, content }) {
  return (
    <div
      style={{
        background: "#f7faf8",
        padding: "22px",
        borderRadius: "18px",
        borderLeft: "5px solid #7da88a",
      }}
    >
      <h4 style={{ color: "#1f4037", marginBottom: "10px" }}>
        {title}
      </h4>

      <p style={{ fontSize: "14px", color: "#355f54", lineHeight: "1.6" }}>
        {content}
      </p>
    </div>
  );
}

export default Insights;
