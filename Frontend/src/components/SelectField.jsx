export default function SelectField({ label, name, options, onChange }) {
    return (
        <div className="space-y-1">
            <label className="font-semibold">{label}</label>
            <select
                name={name}
                onChange={onChange}
                className="w-full p-2 border rounded"
            >
                <option value="">Select {label}</option>
                {options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                ))}
            </select>
        </div>
    );
}