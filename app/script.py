import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import re
from sklearn.model_selection import GridSearchCV
import joblib


def extract_hp(s):
    match = re.search(r'(\d+(\.\d+)?)HP', s)
    return float(match.group(1)) if match else float("0")


def extract_hp_with_avg(s, average_hp):
    match = re.search(r'(\d+(\.\d+)?)HP', s)
    return float(match.group(1)) if match else average_hp


def extract_milage(s):
    numeric_string = ''.join(re.findall(r'\d+', s))
    return int(numeric_string) if numeric_string else float("0")


def extract_monetary_value(s):
    numeric_string = ''.join(re.findall(r'\d+', s))
    return int(numeric_string) if numeric_string else float("0")


def extract_model_year(s):
    try:
        year = int(s)
    except ValueError:
        year = 2024
    return 2024 - year

def RF_training_metrics_after_tuning(X_train, y_train, RF):
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train.ravel(), y_pred))
    r2 = r2_score(y_train.ravel(), y_pred)
    mae = mean_absolute_error(y_train, y_pred)

    return RF

def load_and_preprocess_data(filepath):
    # Load in data
    dataset = pd.read_csv(filepath)

    # Pre-process dataset to only use the most popular 15 types of car
    top_15 = dataset['brand'].value_counts().head(15).index
    dataset_top_brands = dataset[dataset['brand'].isin(top_15)]

    # Clean and format dataset features
    dataset_top_brands['milage'] = dataset_top_brands['milage'].astype(str).apply(extract_milage)
    dataset_top_brands['model_year'] = dataset_top_brands['model_year'].astype(str).apply(extract_model_year)
    dataset_top_brands['price'] = dataset_top_brands['price'].astype(str).apply(extract_monetary_value)
    average_hp = dataset_top_brands['engine'].apply(lambda s: extract_hp(s)).median()
    dataset_top_brands['horse_power'] = dataset_top_brands['engine'].apply(
        lambda s: extract_hp_with_avg(s, average_hp)).astype('float64').round(2)

    # Drop unnecessary columns
    dataset_top_brands = dataset_top_brands.drop(
        ['brand', 'clean_title', 'fuel_type', 'transmission', 'engine', 'ext_col', 'int_col', 'model', 'accident'],
        axis=1)

    # Scale the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = dataset_top_brands.drop(['price'], axis=1)
    y = dataset_top_brands['price']
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, y_train

# Assuming the filepath to your dataset is correctly provided


X_train, y_train = load_and_preprocess_data('used_cars.csv')
# Now you define the parameter grid and perform the grid search
param_grid = {'n_estimators': [100, 200, 500], 'max_features': [2, 3, 5, 8], "max_depth": [3, 5, 10, 12]}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, verbose=10, n_jobs=-1)

grid_search.fit(X_train, y_train.ravel())

print(grid_search.best_params_)
# Use the best parameters found by grid_search
tuned_RF = RandomForestRegressor(max_depth=10, max_features=2, n_estimators=200)
RF_training_metrics_after_tuning(X_train, y_train, tuned_RF)

# After training, save the model and scaler to disk
joblib.dump(tuned_RF, 'tuned_RF.joblib')  # Save the tuned RandomForestRegressor\
