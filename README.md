# 🌾 Crop Yield Prediction using Machine Learning

## 📘 Project Overview
This project aims to **predict crop yield** based on environmental and soil parameters using multiple **Machine Learning regression models**.  
It demonstrates a complete **end-to-end ML workflow** — from data preprocessing and feature engineering to model training, comparison, and deployment via a reusable pipeline.

---

## 🧠 Objective
Accurately estimate the crop yield for different crop and soil types given factors such as temperature, humidity, nutrients, and other environmental attributes.

---

## ⚙️ Workflow Summary

### 1️⃣ Data Preprocessing (`crop_yield_dataset.csv`)
- **Loaded the dataset** using `pandas`.
- **Removed invalid records** (`Crop_Yield == 0`).
- **Checked for missing values**, duplicates, and outliers.
- **Renamed and standardized column names** (e.g., `Temperature → temp`, `Wind_Speed → wind_speed`).
- **Explored the dataset** using:
  - `.info()`, `.describe()`, `.value_counts()`
  - Distribution plots and bar charts (`matplotlib`).

---

### 2️⃣ Exploratory Data Analysis (EDA)
- **Visualized key patterns**:
  - Temperature variation over time.
  - Average crop yield by crop type.
  - Average soil quality by soil type.
  - Distribution of soil pH.
- Helped understand feature importance and possible scaling needs.

---

### 3️⃣ Encoding Categorical Features
Various encoding techniques were applied to handle categorical data efficiently:

| Encoding Type | Library | Description |
|----------------|----------|-------------|
| Label Encoding | `sklearn.preprocessing.LabelEncoder` | Encodes categories as integers. |
| One-Hot Encoding | `OneHotEncoder` | Expands categories into binary columns. |
| Binary Encoding | `category_encoders` | Converts categories into binary bits. |
| Target Encoding | `category_encoders` | Replaces categories with target mean. |
| Leave-One-Out Encoding | `category_encoders` | Similar to target encoding but avoids data leakage. |

Each encoded dataset was stored for further use:
- `Crop_Mod_OneH_MinMax.csv`
- `Crop_Mod_OneH_Stand.csv`

---

### 4️⃣ Feature Scaling
Two normalization techniques were tested:

| Scaling Method | Library | Description |
|----------------|----------|-------------|
| **MinMaxScaler** | `sklearn.preprocessing.MinMaxScaler` | Scales features to a fixed [0,1] range. |
| **StandardScaler** | `sklearn.preprocessing.StandardScaler` | Standardizes data (zero mean, unit variance). |

Both scaling methods were compared to observe impact on model performance.

---

### 5️⃣ Model Training and Evaluation
A wide range of **regression models** were trained and compared on both scaled datasets.

#### 🧩 Models Used
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **XGBoost Regressor**
- **LightGBM Regressor**
- **CatBoost Regressor**
- **AdaBoost Regressor**

#### 🧮 Evaluation Metrics
Each model was evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score (Coefficient of Determination)**

Results were stored as DataFrames for both scaling approaches:
- `res` → MinMax scaled results  
- `res2` → Standard scaled results

---

### 6️⃣ End-to-End ML Pipeline
A fully automated **ML pipeline** was built using `scikit-learn` pipelines for robust preprocessing and prediction.

#### 🔧 Pipeline Components
1. **Custom Date Feature Extractor**
   - Extracts `year`, `month`, `day` from the `Date` column using a custom `TransformerMixin`.

2. **ColumnTransformer**
   - Scales numeric features using `StandardScaler`.
   - Encodes categorical features (`Crop_Type`, `Soil_Type`) using `OneHotEncoder`.

3. **Regressor**
   - A `CatBoostRegressor` model with tuned hyperparameters.

#### ⚡ Pipeline Steps
```python
Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", CatBoostRegressor(...))
])
```

---

### 7️⃣ Model Persistence
Trained pipeline saved using Joblib:
```python
joblib.dump(model, "Catboost.pkl")
```
Reloaded seamlessly for predictions:
```python
loaded_model = joblib.load("Catboost.pkl")
```

---

## 🧾 Approach Summary
- Data Cleaning → Removed zeros, handled missing values, standardized columns.

- Exploratory Analysis → Identified key variable relationships.

- Feature Engineering → Extracted date components, scaled numeric features, encoded categorical ones.

- Model Comparison → Benchmarked multiple regression models using standard metrics.

- Best Model Selection → CatBoostRegressor delivered the most stable and accurate results.

- Pipeline Deployment → Implemented automated preprocessing + prediction pipeline.

- Serialization → Saved and reloaded model for reuse or API integration.


---

# Time Series Models

## 📘 Project Overview
This project focuses on **forecasting crop yield** over time using a combination of **classical statistical models (SARIMAX, ARIMA)** and **machine learning-based time series forecasting (Prophet)**.  
The objective is to predict future crop yields based on historical data and environmental factors such as **Temperature, Humidity, Wind Speed, N, P, K, Soil Quality**, and **Soil pH**.

---

## 🧩 Dataset
The dataset (`crop_yield_dataset.csv`) contains daily records of crop yields and related features.

### Key Columns:
- `Date` — Timestamp of observation  
- `Crop_Yield` — Yield value (target variable)  
- `Temperature`, `Humidity`, `Wind_Speed` — Weather parameters  
- `N`, `P`, `K` — Soil nutrient levels  
- `Soil_pH`, `Soil_Quality` — Soil characteristics  

Rows with `Crop_Yield = 0` were removed to ensure data quality.

---

## ⚙️ Preprocessing Steps
1. **Data Cleaning**
   - Dropped categorical columns: `Crop_Type`, `Soil_Type`.
   - Removed zero-yield records.
2. **Datetime Indexing**
   - Converted `Date` column to datetime.
   - Set it as index for time series operations.
3. **Resampling**
   - Resampled the data to a **daily frequency** to ensure regular intervals.
4. **Handling Missing Values**
   - Used **mean interpolation** for smooth continuity in Prophet.

---

## 🔍 Exploratory Data Analysis (EDA)
- Visualized overall trends in crop yield.
- Conducted **stationarity checks** using:
  - **ADF (Augmented Dickey–Fuller)** test.
  - **KPSS (Kwiatkowski–Phillips–Schmidt–Shin)** test.
- Plotted **ACF** and **PACF** graphs to understand autocorrelation and partial autocorrelation.

---

## 🧠 Model Building

### 1️⃣ SARIMAX Model
- Model: `SARIMAX(order=(1,0,1))`
- Trained on 90% of data, forecasted next 20 days.
- Captured short-term temporal dependencies.

### 2️⃣ ARIMA Model
- Model: `ARIMA(order=(1,1,1))`
- Used differencing to ensure stationarity.
- Diagnostic plots generated to validate residual normality and homoscedasticity.

### 3️⃣ Prophet Model (with Regressors)
- Model: `Prophet()` from Facebook’s Prophet library.
- Added multiple environmental regressors:
  ```python
  ['Soil_pH', 'Temperature', 'Humidity', 'Wind_Speed', 'N', 'P', 'K', 'Soil_Quality']

---

# 🌾 AgriPredict Backend (FastAPI)

This is the **FastAPI backend** for the AgriPredict system.  
It exposes ML prediction endpoints for:

- 🤖 CatBoost regression model  
- 📈 Prophet time series forecasting  
- 🔄 Preprocessing & validation using Pydantic  

---

# 🚀 Features

### ✔ CatBoost Prediction API (`/predict-catboost`)
- Accepts soil, nutrient, and weather inputs  
- Validates them using Pydantic  
- Returns predicted crop yield

### ✔ Prophet Forecasting API (`/forecast`)
- Accepts start date + future periods  
- Takes time-series regressors (Temperature, N, P, K, etc.)  
- Produces future yield forecasts  

### ✔ Model Loading
- `catboost_model.pkl` (or `Catboost.pkl`)
- `prophet_model.pkl`

---

# 📦 Tech Stack

- **FastAPI**
- **Uvicorn**
- **CatBoost**
- **Prophet**
- **Pandas**
- **Joblib**
- **Pydantic**

---


# 🧩 Pydantic Models

### Crop Yield Input
```python
class CropYieldInput(BaseModel):
    Crop_Type: str
    Soil_Type: str
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float
```

---

### Forecast Input
```python
class ForecastInput(BaseModel):
    start_date: date
    periods: int
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float
```
---

# 🌾 Frontend – Crop Yield Prediction (React + JavaScript)

This folder contains the **React-based frontend** for the Crop Yield Prediction System.  
It provides an intuitive interface that collects agricultural parameters and sends them to the FastAPI backend powered by CatBoost.

---

## 🚀 Features

- Developed using **React (JavaScript)**
- **Axios** used for API requests
- Clean reusable components:
  - `InputField`
  - `SelectField`
  - `ResultCard`
- Dropdown options for **Crop Type** and **Soil Type**
- Styled using **Tailwind CSS**
- Fetches predictions from backend and displays neatly

---

## UI

### Home Page
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/83f825e1-b657-48ad-bf13-194a1d0d8913" />

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bb1d6cee-3270-4d7e-b5db-1bec6176f83d" />

---

### CatBoost Page
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d51c263f-781e-412d-9438-1afb6fdea7b9" />

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d05cc511-cf2d-41d0-81de-70e576c7c3d4" />

---

### Prophet Page
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/da902205-0a53-470c-964b-8a97881cada5" />

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4ec204dd-8914-4494-acc1-9d414ddec1e6" />

---



