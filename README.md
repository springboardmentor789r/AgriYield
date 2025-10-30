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
joblib.dump(model, "RandomReg.pkl")
```
Reloaded seamlessly for predictions:
```python
loaded_model = joblib.load("RandomReg.pkl")
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
