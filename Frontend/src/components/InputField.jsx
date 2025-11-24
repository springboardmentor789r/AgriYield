import React from 'react';

export default function InputField({ label, name, type = 'text', value, onChange, placeholder, step, icon }) {
  return (
    <div>
      <label className="block mb-2 font-medium text-gray-700">{label}</label>
      <div className="relative">
        <input
          name={name}
          type={type}
          step={step}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-400 outline-none transition"
        />
        {icon && <div className="absolute right-3 top-3">{icon}</div>}
      </div>
    </div>
  );
}
