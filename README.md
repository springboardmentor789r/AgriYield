# 🌱 AgriYield Predictor — AI Powered Agriculture Insights 🚜📈

A complete **End-to-End Machine Learning + Time-Series Forecasting** web application designed to help farmers, researchers, and agri-business professionals accurately **predict crop yield** and **forecast future production** using environmental & soil factors.

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| 🎯 Crop Yield Prediction | Random Forest ML model trained on multi-year agricultural dataset |
| ⏳ Time-Series Forecasting | Prophet model to forecast upcoming yield trends |
| 🌦️ Environmental Analysis | Uses Temperature, Humidity, Wind Speed, Soil pH, NPK, Soil Quality |
| 📊 Interactive UI Charts | Real-time insights for prediction & forecast |
| 🌐 Full-Stack Application | FastAPI backend + ReactJS + Tailwind CSS frontend |
| 🔁 Pipeline Automation | Encoders, preprocessing, scaling — fully automated |
| 📱 Responsive Design | Works smoothly on desktop and mobile |

---

## 📚 Data Source (Dataset Reference)

Dataset from Kaggle:  
📌 Crop Yield and Environmental Factors (2014-2023)  
🔗 https://www.kaggle.com/datasets/madhankumar789/crop-yield-and-environmental-factors-2014-2023/data

The dataset includes:
- **10 major crop types** (Wheat, Corn, Rice, etc.)
- **5 soil types** (Sandy, Clay, Loamy, Peaty, Saline)
- Daily environmental measurements + crop yield per hectare

---

## 🧩 Technology Stack

### 🔙 Backend (AI Model API)
- Python (FastAPI)
- Joblib (Model Serialization)
- Prophet (Time Series Forecasting)
- Random Forest Regressor (ML Prediction)
- Pandas, NumPy

### 🎨 Frontend (User Interface)
- React.js
- React Router
- Tailwind CSS (Custom theming)
- Axios (API Integration)
- Chart.js (Visual Analytics)

---

## 🚀 Architecture Overview

```mermaid
graph TD;
    User-->Frontend_UI(React + Tailwind);
    Frontend_UI-- API Calls -->Backend(FastAPI);
    Backend-- ML Prediction -->RF_Model(Random Forest);
    Backend-- Time Forecast -->Prophet_Model;
    RF_Model-->Response_Prediction;
    Prophet_Model-->Forecast_Response;


# Machine Learning Models
| Model                   | Purpose                            | Performance               |
| ----------------------- | ---------------------------------- | ------------------------- |
| Random Forest Regressor | Prediction based on soil & weather | High accuracy (R² > 0.99) |
| Prophet (Meta/Facebook) | Short-term yield forecasting       | Stable forecasting trends |


🔮 Future Enhancements

📌 Add deployment (Netlify + Render / Docker)

📌 Real weather & soil sensor integration (IoT)

📌 User accounts & crop planning dashboard

📌 Multi-region farm analytics support

# 🏆 Acknowledgements

This project is developed as part of Infosys Springboard Virtual Internship 6.0
Under the theme of AI in Agriculture

# Authors

| Developer       | Contact                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| **Farhan Abid** | 📧 [f.abid.work@gmail.com](mailto:f.abid.work@gmail.com)                                                   |
|                 | 🔗 LinkedIn: [www.linkedin.com/in/farhan-abid-8001a9253](http://www.linkedin.com/in/farhan-abid-8001a9253) |
|                 | 🐙 GitHub: [http://github.com/farhanabid786](http://github.com/farhanabid786)                              |


# License
This project is licensed — “Built by Farhan Abid with License”
All rights reserved © 2025
