# 🚗 Car Price Prediction Using Machine Learning

This project focuses on predicting used car prices through data cleaning, outlier detection, and training multiple regression models. The workflow is implemented using Python and popular machine learning libraries such as scikit-learn.

---

## 📌 Step-by-Step Summary

1. **Library Import**  
   - Utilized `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, and `sklearn`.

2. **Data Loading**  
   - Loaded `used_cars.csv`. Focused on the top 15 most common car brands to ensure sample balance.

3. **Feature Engineering & Cleaning**  
   - Extracted and converted relevant fields: mileage, model year, price, and engine power (horsepower).  
   - Missing horsepower values were imputed with the median.

4. **Feature Selection**  
   - Dropped columns like brand, transmission, fuel type, and accident history to focus on numerical predictors.

5. **Data Visualization**  
   - Boxplots were used to visualize price distribution. Correlation heatmaps were used to inspect linear relationships.

6. **Outlier Detection with OneClassSVM**  
   - Anomaly detection on scaled mileage vs. price using RBF-kernel `OneClassSVM`. Best `gamma` and `nu` parameters were selected based on lowest deviation from target outlier fraction.

7. **Feature Scaling & Splitting**  
   - Applied `MinMaxScaler` to normalize all features. Split data into 80/20 train-test subsets.

8. **Model Training**  
   - Trained and evaluated the following models:

---

## 🧠 Models Used

| Model                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Support Vector Regression (SVR)** | Used RBF kernel with C=100. Good for capturing non-linear trends.         |
| **Polynomial Regression**           | Applied polynomial transformation and trained using Linear Regression.   |
| **Random Forest Regressor**         | Ensemble method to handle feature interactions and reduce overfitting.   |

Each model was evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

## 📊 Features Used
- Car Age (`2024 - model_year`)
- Mileage (numeric miles)
- Horsepower (extracted and imputed)
- Price (target variable)

---

## 📈 Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

---

## 🌐 Web App

A lightweight **web interface** is included to make predictions interactively.

### 🔧 Features:
- Enter inputs: mileage, model year, horsepower
- Get predicted price instantly
