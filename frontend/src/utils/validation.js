// ...existing code...
export function validateForm(form) {
  const errors = {};

  if (!form.Crop_Type) errors.Crop_Type = "Please select a crop.";
  if ('Soil_Type' in form && !form.Soil_Type)
    errors.Soil_Type = "Please select a soil type.";

  const numeric = ["Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K"];
  if ('Months' in form) numeric.push("Months");

  numeric.forEach(field => {
    if (!(field in form)) return;
    if (form[field] === "") {
      errors[field] = "This field is required.";
    } else if (isNaN(form[field])) {
      errors[field] = "Enter a valid number.";
    }
  });

  return errors;
}
// ...existing code...