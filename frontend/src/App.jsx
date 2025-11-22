import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Home from "./pages/Home.jsx";
import Predict from "./pages/Predict.jsx";
import Forecast from "./pages/Forecast.jsx";

function App() {
  return (
    <div className="min-h-screen flex bg-agri-bg">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <main className="flex-1 min-h-screen overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-6 sm:py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/forecast" element={<Forecast />} />
            {/* Default redirect */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
