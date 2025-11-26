import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

export async function predictCrop(input) {
  const res = await api.post("/predict", input);
  return res.data;
}

export async function predictTimeseries(input) {
  // matches: POST /timeseries/predict/timeseries
  const res = await api.post("/timeseries/predict/timeseries", input);
  return res.data;
}

export default api;
