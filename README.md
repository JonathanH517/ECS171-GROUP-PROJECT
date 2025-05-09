# 🚗 Car Price Prediction Using Machine Learning

This project predicts used car prices using regression techniques and includes a lightweight user interface where users can input car attributes and receive price estimates. It combines data preprocessing, anomaly filtering, and regression modeling with an interactive command-line experience.

---

## 📌 Step-by-Step Summary

1. **Library Import**  
   - Includes: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `sklearn`.

2. **Data Loading**  
   - Loads `used_cars.csv` and filters to the top 15 most frequent car brands.

3. **Feature Engineering & Cleaning**  
   - Extracts numeric data from mileage, engine, model year, and price columns.
   - Imputes missing horsepower values using the dataset’s median horsepower.

4. **Feature Selection**  
   - Drops redundant or unused columns such as `brand`, `model`, `fuel_type`, etc.

5. **Data Visualization**  
   - Visual insights via boxplots and heatmaps to identify outliers and correlations.

6. **Outlier Detection with OneClassSVM**  
   - Applies `OneClassSVM` on scaled features to remove pricing anomalies.

7. **Feature Scaling & Splitting**  
   - Normalizes features with `MinMaxScaler` and splits data into training/testing subsets.

8. **Model Training**  
   - Trains and evaluates Support Vector Regression, Polynomial Regression, and Random Forest.

---

## 🧠 Models Used

| Model                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Support Vector Regression (SVR)** | RBF kernel (C=100), captures smooth nonlinear trends.                   |
| **Polynomial Regression**           | Expands features using polynomial terms before applying Linear Regression. |
| **Random Forest Regressor**         | Ensemble method optimized using `GridSearchCV` with hyperparameter tuning. |

---

## 🔁 Prediction Interface

A simple command-line interface is included via `model.py`, where users can input:

- Car model year  
- Mileage  
- Horsepower  

And receive an **instant price prediction**.

### 🖥️ Run the interface:
```bash
python model.py
