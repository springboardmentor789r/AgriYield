
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function ForecastChart({ forecast }) {
  const labels = forecast.map(r => r.Month);
  const dataPoints = forecast.map(r => r.Predicted_Yield);

  const data = {
    labels,
    datasets: [
      {
        label: "Predicted Yield",
        data: dataPoints,
        borderColor: "#0369a1",
        backgroundColor: "#508fafff",
        tension: 0.4
      }
    ]
  };

  return <Line data={data} />;
}
