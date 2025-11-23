document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show loading
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    const submitBtn = document.getElementById('submitBtn');
    btnText.textContent = 'Predicting...';
    spinner.style.display = 'block';
    submitBtn.disabled = true;

    // Collect data
    const data = {
        crop_type: document.getElementById('cropType').value,
        soil_type: document.getElementById('soilType').value,
        soil_pH: parseFloat(document.getElementById('soilPH').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        wind_speed: parseFloat(document.getElementById('windSpeed').value),
        N: parseFloat(document.getElementById('N').value),
        P: parseFloat(document.getElementById('P').value),
        K: parseFloat(document.getElementById('K').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('result').innerText = `Predicted Yield: ${result.predicted_yield} tons/ha`;
        } else {
            const error = await response.json();
            document.getElementById('result').innerText = `Error: ${error.detail}`;
        }
    } catch (error) {
        document.getElementById('result').innerText = `Network Error: ${error.message}`;
    } finally {
        // Hide loading
        btnText.textContent = 'Predict Yield';
        spinner.style.display = 'none';
        submitBtn.disabled = false;
    }
});