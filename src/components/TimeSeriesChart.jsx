import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  TimeScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  PointElement,
  Filler,
} from "chart.js";
import "chartjs-adapter-date-fns";

ChartJS.register(
  LineElement,
  PointElement,
  TimeScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function TimeSeriesChart({ forecast }) {
  if (!forecast || !forecast.length) return null;

  const labels = forecast.map((item) => item.date);
  const dataValues = forecast.map((item) => item.predicted_yield);

  const data = {
    labels,
    datasets: [
      {
        label: "Predicted Yield (t/ha)",
        data: dataValues,
        tension: 0.4,
        borderWidth: 2,
        fill: true,
        borderColor: "rgba(75, 192, 192, 1)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
        pointRadius: 3,
        pointHoverRadius: 5,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      title: { 
        display: true, 
        text: "Time Series Forecasted Crop Yield",
        font: { size: 18 },
      },
      tooltip: {
        callbacks: {
          label: (context) => `Yield: ${context.raw} t/ha`,
        },
      },
    },
    scales: {
      x: { 
        type: "time",
        time: { unit: "day" },
        title: { display: true, text: "Date" },
      },
      y: { 
        beginAtZero: false,
        title: { display: true, text: "Predicted Yield (t/ha)" },
      },
    },
  };

  return (
    <div style={{ width: "100%", height: "350px" }}>
      <Line data={data} options={options} />
    </div>
  );
}
