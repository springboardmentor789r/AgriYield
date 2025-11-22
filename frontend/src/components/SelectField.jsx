

export default function SelectField({ label, name, value, onChange, options }) {
  return(
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">{label}</label>
      <select
        name={name}
        value={value ?? ""}
        onChange={onChange}
        className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-300"
      >
        <option value="">Select</option>
        {options.map(opt => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </div>
  );
}
