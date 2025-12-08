const Model = ({ show, onClose, message, result }) => {
  if (!show) return null;

  // Safely convert values to numbers
  const minVal = Number(result?.min_predicted);
  const maxVal = Number(result?.max_predicted);

  // Compute average ONLY if valid numbers
  const avgYield =
    !isNaN(minVal) && !isNaN(maxVal)
      ? ((minVal + maxVal) / 2).toFixed(2)
      : null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div className="bg-white p-6 rounded-xl shadow-lg w-80 text-center">
        <h2 className="text-xl font-bold mb-4">Response</h2>

        {/* Message */}
        <p className="text-gray-700 mb-4">{message}</p>

        {/* ---- CATBOOST REGRESSION SINGLE VALUE ---- */}
        {result?.predicted_crop_yield !== undefined && (
          <p className="text-green-600 font-bold text-lg">
            {Number(result.predicted_crop_yield).toFixed(2)}
          </p>
        )}

        {/* ---- AVERAGE YIELD (TIME SERIES) ---- */}
        {avgYield !== null && (
          <p className="text-green-600 font-bold text-lg">
            Average Yield: {avgYield}
          </p>
        )}

        {/* ---- FULL TIME SERIES PREDICTIONS ---- */}
        {Array.isArray(result?.predicted_values) && (
          <div className="text-left mt-3">
            <p className="font-semibold">Predictions:</p>

            <ul className="list-disc ml-5 max-h-40 overflow-y-auto">
              {result.predicted_values.map((val, idx) => (
                <li key={idx}>{Number(val).toFixed(2)}</li>
              ))}
            </ul>

            {/* Min/Max */}
            {!isNaN(minVal) && <p>Min: {minVal.toFixed(2)}</p>}
            {!isNaN(maxVal) && <p>Max: {maxVal.toFixed(2)}</p>}
          </div>
        )}

        {/* Close Button */}
        <button
          onClick={onClose}
          className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default Model;
