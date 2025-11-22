import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-50 via-emerald-100 to-emerald-200 
                    flex items-center justify-center relative overflow-hidden">

      {/* Floating animated circles (background accents) */}
      <div className="absolute w-72 h-72 bg-emerald-300/30 rounded-full blur-3xl top-10 left-10 animate-pulse"></div>
      <div className="absolute w-96 h-96 bg-green-200/40 rounded-full blur-3xl bottom-10 right-10 animate-ping"></div>

      <div className="max-w-4xl w-full mx-auto px-6 animate-fadeIn">
        
        {/* Main card */}
        <div className="bg-white/70 backdrop-blur-lg rounded-3xl p-12 shadow-2xl border border-white/40">
          
          {/* Title Section */}
          <h1 className="text-4xl md:text-5xl font-extrabold text-emerald-700 drop-shadow-sm text-center mb-4 animate-slideDown">
            🌾 AgriYield
          </h1>
          <p className="text-center text-slate-700 text-lg mb-10 animate-slideUp">
            Smart AI-powered crop yield prediction & seasonal forecasting.
          </p>

          {/* Card Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

            {/* Predict Card */}
            <Link to="/predict">
              <div className="group p-8 rounded-2xl border bg-white/90 shadow-md cursor-pointer 
                              hover:shadow-xl hover:-translate-y-1 hover:bg-emerald-50
                              transition-all duration-300
                              animate-fadeUp">
                
                <h2 className="text-2xl font-semibold text-emerald-700">Predict Yield</h2>
                <p className="text-slate-600 mt-2">
                  Real-time prediction using soil and weather parameters.
                </p>

                <button className="mt-5 px-5 py-2 bg-emerald-600 text-white rounded-lg shadow 
                                   group-hover:bg-emerald-700 transition">
                  Open
                </button>

                <div className="mt-4 text-emerald-400 text-sm opacity-0 group-hover:opacity-100 transition">
                  → Instantly analyze your field
                </div>
              </div>
            </Link>

            {/* Forecast Card */}
            <Link to="/forecast">
              <div className="group p-8 rounded-2xl border bg-white/90 shadow-md cursor-pointer 
                              hover:shadow-xl hover:-translate-y-1 hover:bg-sky-50
                              transition-all duration-300
                              animate-fadeUp delay-200">
                
                <h2 className="text-2xl font-semibold text-sky-700">Forecast Yield</h2>
                <p className="text-slate-600 mt-2">
                  Predict monthly yield using advanced Prophet forecasting.
                </p>

                <button className="mt-5 px-5 py-2 bg-sky-600 text-white rounded-lg shadow
                                   group-hover:bg-sky-700 transition">
                  Open
                </button>

                <div className="mt-4 text-sky-500 text-sm opacity-0 group-hover:opacity-100 transition">
                  → Plan your farming season better
                </div>
              </div>
            </Link>

          </div>

        </div>
      </div>
    </div>
  );
}
