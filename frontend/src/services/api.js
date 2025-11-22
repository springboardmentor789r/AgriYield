import axios from "axios";

const api = axios.create({
    baseURL: "http://localhost:8000",
    timeout: 5000,
});

export async function predictYield(data) {
    try {
        const res = await api.post("/predict", data);
        return res.data;
    } catch (error) {
        console.error("Predict error:", error);
        throw error;
    }
}

export async function forecastYield(data) {
    try {
        const res = await api.post("/forecast", data);
        return res.data;
    } catch (error) {
        console.error("Forecast error:", error);
        throw error;
    }
}

