export default function SelectField({label,name,value,options,onChange,error,}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        {label}
      </label>

      <select
        name={name}
        value={value}
        onChange={onChange}
        className={`w-full px-3 py-2 border rounded-lg bg-white focus:outline-none
          ${error ? "border-red-500 focus:ring-red-300" : "focus:ring-emerald-300"}
        `}
      >
        <option value="">Select {label}</option>

        {options.map((opt) => (
          <option key={opt} value={opt.toLowerCase()}>
            {opt}
          </option>
        ))}
      </select>

      {error && <p className="text-red-600 text-sm mt-1">{error}</p>}
    </div>
  );
}
