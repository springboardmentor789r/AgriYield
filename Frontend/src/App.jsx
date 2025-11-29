import React, { useState } from "react";
import Sidebar from "./components/Sidebar";
import RegressionPage from "./pages/RegressionPage";
import TimeSeriesPage from "./pages/TimeSeriesPage";

export default function App() {
  const [route, setRoute] = useState("regression"); // default
  return (
    <div className="flex min-h-screen app-bg">
      <Sidebar route={route} setRoute={setRoute} />
      <main className="flex-1 p-8 max-w-6xl mx-auto">
        {route === "regression" && <RegressionPage />}
        {route === "timeseries" && <TimeSeriesPage />}
      </main>
    </div>
  );
}
