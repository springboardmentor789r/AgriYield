import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="w-full h-screen bg-gradient-to-br from-slate-900 via-emerald-900 to-slate-900 
                    relative overflow-hidden flex flex-col">

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-80 h-80 bg-emerald-500/20 rounded-full blur-3xl -top-10 -left-10 animate-pulse"></div>
        <div className="absolute w-80 h-80 bg-blue-500/15 rounded-full blur-3xl top-1/4 -right-10 animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute w-72 h-72 bg-emerald-400/10 rounded-full blur-3xl bottom-10 left-1/4 animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 w-full h-full flex flex-col">
        
        {/* Navigation Bar */}
        <nav className="backdrop-blur-md bg-white/5 border-b border-white/10 px-8 py-4 flex-shrink-0">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🌾</span>
              <span className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                AgriYield
              </span>
            </div>
            <div className="text-xs text-gray-300 font-medium">
              AI-Powered Agriculture Solutions
            </div>
          </div>
        </nav>

        {/* Hero Section - Centered */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-5xl">
            
            {/* Hero Content */}
            <div className="text-center mb-10 animate-fadeIn">
              <h1 className="text-5xl md:text-6xl font-black mb-4 leading-tight">
                <span className="bg-gradient-to-r from-emerald-300 via-cyan-300 to-emerald-300 bg-clip-text text-transparent">
                  Revolutionize
                </span>
                <br />
                <span className="text-white">Your Farming</span>
              </h1>
              
              <p className="text-base md:text-lg text-gray-300 mb-2 font-light max-w-2xl mx-auto">
                AI-powered crop yield prediction & seasonal forecasting
              </p>
              
              <p className="text-xs md:text-sm text-gray-400 max-w-xl mx-auto">
                Real-time predictions • Data-driven insights • 97% accuracy
              </p>
            </div>

            {/* Feature Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">

              {/* Predict Yield Card */}
              <Link to="/predict" className="group h-full">
                <div className="h-full relative backdrop-blur-xl bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 
                                border border-emerald-400/30 rounded-xl p-6
                                hover:border-emerald-400/60 hover:from-emerald-500/20 hover:to-emerald-500/10
                                hover:shadow-xl hover:shadow-emerald-500/20
                                transition-all duration-500 cursor-pointer group
                                overflow-hidden animate-fadeUp">
                  
                  <div className="relative z-10">
                    <div className="text-4xl mb-3">📊</div>
                    
                    <h2 className="text-xl md:text-2xl font-bold text-emerald-200 mb-2 group-hover:text-emerald-100 transition">
                      Predict Yield
                    </h2>
                    
                    <p className="text-gray-300 text-sm leading-relaxed mb-4">
                      Real-time predictions using soil, weather & nutrients
                    </p>

                    <button className="w-full px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 
                                       text-white text-sm font-semibold rounded-lg shadow
                                       hover:from-emerald-600 hover:to-emerald-700 hover:shadow-lg
                                       transition-all duration-300">
                      Start →
                    </button>
                  </div>
                </div>
              </Link>

              {/* Forecast Yield Card */}
              <Link to="/forecast" className="group h-full">
                <div className="h-full relative backdrop-blur-xl bg-gradient-to-br from-cyan-500/10 to-blue-500/5 
                                border border-cyan-400/30 rounded-xl p-6
                                hover:border-cyan-400/60 hover:from-cyan-500/20 hover:to-blue-500/10
                                hover:shadow-xl hover:shadow-cyan-500/20
                                transition-all duration-500 cursor-pointer group
                                overflow-hidden animate-fadeUp" style={{animationDelay: '0.2s'}}>
                  
                  <div className="relative z-10">
                    <div className="text-4xl mb-3">📈</div>
                    
                    <h2 className="text-xl md:text-2xl font-bold text-cyan-200 mb-2 group-hover:text-cyan-100 transition">
                      Forecast Yield
                    </h2>
                    
                    <p className="text-gray-300 text-sm leading-relaxed mb-4">
                      Monthly forecasts for entire season planning
                    </p>

                    <button className="w-full px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 
                                       text-white text-sm font-semibold rounded-lg shadow
                                       hover:from-cyan-600 hover:to-blue-700 hover:shadow-lg
                                       transition-all duration-300">
                      Start →
                    </button>
                  </div>
                </div>
              </Link>

            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-3 text-center text-xs md:text-sm">
              <div>
                <div className="font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  97%
                </div>
                <p className="text-gray-400 text-xs">Accuracy</p>
              </div>
              <div>
                <div className="font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  30K+
                </div>
                <p className="text-gray-400 text-xs">Data Points</p>
              </div>
              <div>
                <div className="font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  24/7
                </div>
                <p className="text-gray-400 text-xs">Analysis</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="backdrop-blur-md bg-white/5 border-t border-white/10 px-8 py-3 flex-shrink-0 text-center text-xs text-gray-400">
          <p>© 2025 AgriYield • Powered by CatBoost & Prophet AI</p>
        </footer>
      </div>
    </div>
  );
}