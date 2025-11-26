# Agri Predictor - Project Guide

Welcome to **Agri Predictor**! This guide will explain what this project is, how it works, and how to run it on your machine.

## 1. Project Overview

**Agri Predictor** is a full-stack web application designed to predict crop yields based on environmental and soil conditions. It uses a Machine Learning model to forecast results and displays them on an interactive dashboard.

### Architecture
The project is split into two main parts:
1.  **Backend (`crop-backend`)**: A Python-based API that handles the logic and predictions.
2.  **Frontend (`crop-frontend`)**: A React-based user interface that allows users to input data and view results.

---

## 2. Technology Stack

### Backend
-   **Language**: Python
-   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (A modern, fast web framework for building APIs).
-   **Machine Learning**:
    -   **Prophet**: A forecasting procedure implemented in Python, used here for predicting yield.
    -   **Joblib**: Used to load the pre-trained model (`Prophet.pkl`).
    -   **Pandas & NumPy**: Used for data manipulation.

### Frontend
-   **Library**: [React](https://reactjs.org/) (Bootstrapped with Create React App).
-   **Charting**: [Chart.js](https://www.chartjs.org/) & `react-chartjs-2` (For visualizing the predictions).
-   **Styling**: CSS.

---

## 3. How It Works

1.  The **Frontend** sends a request to the Backend with data like Soil pH, Temperature, Humidity, etc.
2.  The **Backend** receives this data via the `/predict` endpoint.
3.  It formats the data and feeds it into the loaded **Prophet model**.
4.  The model returns a predicted yield (`yhat`) along with lower and upper bounds (`yhat_lower`, `yhat_upper`).
5.  The Backend sends this response back to the Frontend.
6.  The Frontend displays the prediction to the user, likely using a chart.

---

## 4. How to Run the Project

You need to run both the Backend and the Frontend simultaneously in two separate terminals.

### Prerequisites
-   **Python** (3.8 or higher recommended)
-   **Node.js** and **npm** (for the frontend)

### Step 1: Run the Backend

1.  Open a terminal (Command Prompt or PowerShell).
2.  Navigate to the backend directory:
    ```bash
    cd "Agri Predictor/crop-backend"
    ```
3.  (Optional but Recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```
4.  Install the required Python packages:
    ```bash
    python -m pip install -r requirements.txt
    ```
5.  Start the FastAPI server:
    ```bash
    python -m uvicorn main:app --reload
    ```
    You should see output indicating the server is running, typically at `http://127.0.0.1:8000`.

### Step 2: Run the Frontend

1.  Open a **new** terminal window.
2.  Navigate to the frontend directory:
    ```bash
    cd "Agri Predictor/crop-frontend"
    ```
3.  Install the Node.js dependencies:
    ```bash
    npm install
    ```
4.  Start the React application:
    ```bash
    npm start
    ```
5.  Your browser should automatically open to `http://localhost:3000`. If not, open that URL manually.

---

## 5. Troubleshooting

-   **Backend Errors**: If you see errors about missing modules, make sure you ran `pip install -r requirements.txt` inside the `crop-backend` folder.
-   **Frontend Errors**: If `npm start` fails, try deleting the `node_modules` folder and running `npm install` again.
-   **Connection Issues**: Ensure the backend is running on port 8000. The frontend expects the backend to be available to fetch predictions.

Enjoy exploring Agri Predictor!
