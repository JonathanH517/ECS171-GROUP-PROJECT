# 🚗 Car Price Prediction Using Machine Learning

This project predicts the prices of used cars using regression techniques and outlier filtering based on a curated dataset. The model pipeline includes data preprocessing, outlier removal, scaling, and training on SVR and polynomial regression models.

---

## 📌 Step-by-Step Summary

1. **Library Import**  
   - Imported libraries for data handling (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`, `plotly`), and machine learning (`sklearn`).

2. **Data Loading**  
   - Loaded `used_cars.csv`, filtered for the 15 most popular car brands.

3. **Feature Engineering & Cleaning**  
   - Cleaned and extracted numeric values from `milage`, `model_year`, `price`, and `engine` fields.  
   - Replaced missing `horse_power` with median values.

4. **Feature Selection**  
   - Removed non-predictive or redundant columns such as `color`, `transmission`, `accident`, etc.

5. **Data Visualization**  
   - Used box plots and heatmaps to inspect distribution and correlations.

6. **Outlier Detection with OneClassSVM**  
   - Applied `OneClassSVM` on scaled data (`milage` vs. `price`) to detect and remove outliers.

7. **Feature Scaling & Splitting**  
   - Applied `MinMaxScaler` to numerical features and split the dataset into training and testing subsets.

8. **Modeling**  
   - Trained:
     - **Support Vector Regression (SVR)** – RBF kernel, C=100
     - **Polynomial Regression** – Feature expansion using `PolynomialFeatures` + Linear Regression  
   - Evaluated using MAE, RMSE, and R².

---

## 📊 Features Used
- Model Age (2024 - model year)
- Mileage
- Horsepower
- Price (target variable)

---

## 🧠 Models Trained
- **Support Vector Regression (SVR)**
  - Kernel: RBF
  - C = 100
- **Polynomial Regression**
  - Degree configurable
  - Linear model trained on polynomially transformed features

---

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📂 File Structure
