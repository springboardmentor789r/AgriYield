export function validateForm(form) {
  const errors = {};

  if (!form.Crop_Type) errors.Crop_Type = "Please select a crop.";
  if (form.Soil_Type !== undefined && !form.Soil_Type)
    errors.Soil_Type = "Please select a soil type.";

  const numeric = ["Soil_pH", "Temperature", "Humidity", "Wind_Speed", "N", "P", "K", "Months"];

  numeric.forEach(field => {
    if (form[field] === "") {
      errors[field] = "This field is required.";
    } else if (isNaN(form[field])) {
      errors[field] = "Enter a valid number.";
    }
  });

  return errors;
}
