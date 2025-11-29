import { Routes, Route, Link } from "react-router-dom";
import CatBoostPage from "./CatBoostPage.jsx";
import ProphetPage from "./ProphetPage.jsx";

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur flex flex-wrap items-center justify-between gap-2 px-6 py-3">
        <h1 className="text-lg font-semibold tracking-wide">
          Crop Yield Dashboard
        </h1>
        <nav className="flex gap-2 text-sm">
          <Link
            to="/"
            className="px-3 py-1 rounded-full border border-transparent text-slate-300 hover:text-white hover:border-emerald-400/50 hover:bg-slate-900 transition"
          >
            CatBoost Prediction
          </Link>
          <Link
            to="/prophet"
            className="px-3 py-1 rounded-full border border-transparent text-slate-300 hover:text-white hover:border-emerald-400/50 hover:bg-slate-900 transition"
          >
            Prophet Forecast
          </Link>
        </nav>
      </header>

      <main className="flex-1 flex justify-center px-4 py-6">
        <div className="w-full max-w-5xl">
          <Routes>
            <Route path="/" element={<CatBoostPage />} />
            <Route path="/prophet" element={<ProphetPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default App;
