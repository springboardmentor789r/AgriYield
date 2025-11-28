import { NavLink } from "react-router-dom";
import logo from "../assets/logo.png"; // put your AgriYield logo here

const navLinkClass =
  "flex items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-medium transition hover:bg-emerald-100/70";

export default function Sidebar() {
  return (
    <aside className="hidden md:flex w-64 flex-col bg-emerald-900 text-emerald-50 shadow-lg">
      {/* Logo + title */}
      <div className="flex items-center gap-3 px-6 py-4 border-b border-emerald-800">
        <img
          src={logo}
          alt="AgriYield Predictor Logo"
          className="h-10 w-10 rounded-full object-cover border border-emerald-500 shadow"
        />
        <div>
          <h1 className="text-lg font-semibold leading-tight">AgriYield</h1>
          <p className="text-[11px] uppercase tracking-widest text-emerald-200">
            AI Crop Analytics
          </p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `${navLinkClass} ${
              isActive ? "bg-emerald-700/80" : "bg-transparent"
            }`
          }
        >
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-700">
            📊
          </span>
          <span>Dashboard</span>
        </NavLink>

        <NavLink
          to="/predict"
          className={({ isActive }) =>
            `${navLinkClass} ${
              isActive ? "bg-emerald-700/80" : "bg-transparent"
            }`
          }
        >
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-700">
            🌾
          </span>
          <span>Yield Prediction</span>
        </NavLink>

        <NavLink
          to="/forecast"
          className={({ isActive }) =>
            `${navLinkClass} ${
              isActive ? "bg-emerald-700/80" : "bg-transparent"
            }`
          }
        >
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-emerald-700">
            ⏱️
          </span>
          <span>Time Forecast</span>
        </NavLink>
      </nav>

      {/* Footer / credits */}
      <div className="mt-auto pt-6 border-t border-emerald-800/40 text-[11px] leading-relaxed text-emerald-100/80">
      <p>Credits</p>
      <p className="mt-1">
        Built by <span className="font-semibold"> AI Intern</span> @2025.
        </p>
        <p className="mt-1 text-emerald-100/70">
        Live project - Infosys Springboard Virtual Internship 6.0  {new Date().getFullYear()}
        </p>
      </div>

    </aside>
  );
}

 
