import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sprout } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const current = location.pathname;

  const Button = ({ to, children }) => (
    <Link
      to={to}
      className={`px-4 py-2 rounded-lg font-medium transition ${
        current === to ? 'bg-green-600 text-white' : 'text-gray-700 hover:bg-gray-100'
      }`}
    >
      {children}
    </Link>
  );

  return (
    <nav className="bg-white shadow sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Sprout className="text-green-600 w-8 h-8" />
            <span className="text-xl font-bold text-gray-800">CropPred</span>
          </Link>

          <div className="flex items-center space-x-2">
            <Button to="/">Home</Button>
            <Button to="/catboost">CatBoost Model</Button>
            <Button to="/prophet">Prophet Forecast</Button>
          </div>
        </div>
      </div>
    </nav>
  );
}
