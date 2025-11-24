
export default function InputField({label, name, type="text", value, onChange, placeholder, unit }) {
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
          className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-300"
        />

        {unit && (
          <span className="text-sm text-slate-600 whitespace-nowrap">
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}
