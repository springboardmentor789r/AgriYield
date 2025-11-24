import React from 'react';

export default function StatCard({ number, label }) {
  return (
    <div className="bg-white p-8 rounded-xl shadow-lg">
      <div className="text-5xl font-bold text-green-600 mb-2">{number}</div>
      <div className="text-gray-600 font-medium">{label}</div>
    </div>
  );
}
