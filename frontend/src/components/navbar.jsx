// src/components/Navbar.jsx
import { NavLink } from 'react-router-dom';
import logo from '../assets/logo.png';

const Navbar = () => {
  return (
    <header className="sticky top-0 z-30 bg-emerald-950/90 backdrop-blur shadow-md">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        {/* Logo & brand */}
        <div className="flex items-center gap-2">
          <img
            src={logo}
            alt="AgriYield Predictor logo"
            className="h-9 w-9 rounded-full border border-emerald-400/70 bg-white object-cover"
          />
          <div className="leading-tight">
            <div className="text-sm font-semibold uppercase tracking-[0.25em] text-emerald-300">
              AgriYield
            </div>
            <div className="text-xs text-emerald-100/80">
              AI Crop Analytics
            </div>
          </div>
        </div>

        {/* Menu */}
        <div className="flex items-center gap-1 text-sm font-medium">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `rounded-full px-4 py-2 transition ${
                isActive
                  ? 'bg-emerald-400 text-emerald-950 shadow-md'
                  : 'text-emerald-100 hover:bg-emerald-800/70 hover:text-white'
              }`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/predict"
            className={({ isActive }) =>
              `rounded-full px-4 py-2 transition ${
                isActive
                  ? 'bg-emerald-400 text-emerald-950 shadow-md'
                  : 'text-emerald-100 hover:bg-emerald-800/70 hover:text-white'
              }`
            }
          >
            Yield Prediction
          </NavLink>
          <NavLink
            to="/forecast"
            className={({ isActive }) =>
              `rounded-full px-4 py-2 transition ${
                isActive
                  ? 'bg-emerald-400 text-emerald-950 shadow-md'
                  : 'text-emerald-100 hover:bg-emerald-800/70 hover:text-white'
              }`
            }
          >
            Time Forecast
          </NavLink>
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
