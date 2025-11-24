import React from 'react';

export default function FactorCard({ icon, title, description }) {
  return (
    <div className="bg-gradient-to-br from-green-50 to-emerald-100 p-6 rounded-lg text-center hover:shadow-lg transition">
      <div className="text-green-600 flex justify-center mb-3">{icon}</div>
      <h4 className="font-semibold text-gray-800 mb-1">{title}</h4>
      <p className="text-sm text-gray-600">{description}</p>
    </div>
  );
}
