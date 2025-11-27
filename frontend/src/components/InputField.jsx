import React from "react";

export default function InputField({label,name,type = "text",value,onChange,placeholder,unit,error}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        {label}
      </label>

      <div className="flex items-center gap-2">
        <input
          name={name}
          type={type}
          value={value ?? ""}
          placeholder={placeholder ?? ""}
          onChange={onChange}
          autoComplete="on"
          className={`w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2
            ${error ? "border-red-500 focus:ring-red-300" : "border-gray-300 focus:ring-emerald-300"}
          `}
        />
        {unit && (
          <span className="text-sm text-gray-500 whitespace-nowrap">{unit}</span>
        )}
      </div>

      {error && <p className="text-red-600 text-sm mt-1">{error}</p>}
    </div>
  );
}
