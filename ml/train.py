import os
import warnings
import json

# Suppress noisy TensorFlow & Keras log outputs in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seed for reproducibility
np.random.seed(42)

def prepare_data_for_all_models(df, scaler, lookback=6):
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    
    X_scaled = scaler.transform(df[pollutant_cols].values)
    y = df['AQI'].values
    
    X_tabular_list, y_list = [], []
    X_lstm_list = []
    
    for city in df['City'].unique():
        city_mask = df['City'] == city
        city_indices = np.where(city_mask)[0]
        
        for i in range(lookback, len(city_indices)):
            seq_idx = city_indices[i - lookback:i]
            target_idx = city_indices[i]
            
            X_lstm_list.append(X_scaled[seq_idx])
            X_tabular_list.append(X_scaled[target_idx])
            y_list.append(y[target_idx])
            
    return np.array(X_tabular_list), np.array(X_lstm_list), np.array(y_list)

def build_ann_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(6,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    return model

def build_lstm_model(lookback=6):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=(lookback, 6), return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
    return model

def main():
    print("==================================================")
    print("      AQI MACHINE LEARNING BENCHMARK PIPELINE      ")
    print("==================================================")
    
    # 1. Generate fresh dataset up to the current date/time
    raw_data_path = "data/raw/city_hour.csv"
    from ml.data_generator import generate_full_dataset
    print("\n[Step 1/6] Reading live air quality data up to current timestamp...")
    generate_full_dataset(raw_data_path)
    
    # 2. Run Preprocessing
    print("\n[Step 2/6] Cleaning features & splitting train/test sets (80% train / 20% test)...")
    from ml.preprocessing import AQIPreprocessingPipeline
    pipeline = AQIPreprocessingPipeline(raw_data_path)
    
    scaler_path = "models/scaler.joblib"
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, clean_df = pipeline.fit_transform(save_scaler_path=scaler_path)
    
    # 3. Prepare sequences
    lookback = 6
    X_train_tab, X_train_lstm, y_train = prepare_data_for_all_models(clean_df.iloc[:int(len(clean_df)*0.8)], pipeline.scaler, lookback)
    X_test_tab, X_test_lstm, y_test = prepare_data_for_all_models(clean_df.iloc[int(len(clean_df)*0.8):], pipeline.scaler, lookback)
    
    print(f" -> Training set size: {len(X_train_tab)} records")
    print(f" -> Holdout test set size: {len(X_test_tab)} records")
    
    # 4. Train & Evaluate Models
    metrics = {}
    os.makedirs("models", exist_ok=True)
    
    print("\n[Step 3/6] Training & Evaluating 9 Algorithm Families...")
    
    # 1. Linear Regression
    print(" -> [1/9] Fitting Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_tab, y_train)
    joblib.dump(lr, "models/linear_regression.joblib")
    y_pred = lr.predict(X_test_tab)
    metrics["Linear Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 2. Polynomial Regression
    print(" -> [2/9] Fitting Polynomial Regression (Degree 2)...")
    poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly.fit(X_train_tab, y_train)
    joblib.dump(poly, "models/polynomial_regression.joblib")
    y_pred = poly.predict(X_test_tab)
    metrics["Polynomial Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 3. Ridge Regression
    print(" -> [3/9] Fitting Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_tab, y_train)
    joblib.dump(ridge, "models/ridge_regression.joblib")
    y_pred = ridge.predict(X_test_tab)
    metrics["Ridge Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 4. Lasso Regression
    print(" -> [4/9] Fitting Lasso Regression...")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_tab, y_train)
    joblib.dump(lasso, "models/lasso_regression.joblib")
    y_pred = lasso.predict(X_test_tab)
    metrics["Lasso Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 5. Support Vector Regression (SVR)
    print(" -> [5/9] Fitting Support Vector Regression (SVR)...")
    svr_train_size = min(3000, len(X_train_tab))
    svr = SVR(kernel='rbf', C=10.0, epsilon=0.1)
    svr.fit(X_train_tab[:svr_train_size], y_train[:svr_train_size])
    joblib.dump(svr, "models/svr.joblib")
    y_pred = svr.predict(X_test_tab)
    metrics["Support Vector Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 6. Decision Tree Regression
    print(" -> [6/9] Fitting Decision Tree Regression...")
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train_tab, y_train)
    joblib.dump(dt, "models/decision_tree.joblib")
    y_pred = dt.predict(X_test_tab)
    metrics["Decision Tree Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 7. Random Forest Regression
    print(" -> [7/9] Fitting Random Forest Regression...")
    rf = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_tab, y_train)
    joblib.dump(rf, "models/random_forest.joblib")
    y_pred = rf.predict(X_test_tab)
    metrics["Random Forest Regression"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # Import TensorFlow for deep learning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    
    # 8. Artificial Neural Network (ANN)
    print(" -> [8/9] Fitting Artificial Neural Network (ANN)...")
    ann = build_ann_model()
    ann.fit(X_train_tab, y_train / 500.0, epochs=15, batch_size=1024, validation_split=0.1, verbose=0)
    ann.save("models/ann.keras")
    y_pred = ann.predict(X_test_tab, verbose=0).flatten() * 500.0
    metrics["Artificial Neural Network"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 9. LSTM Network
    print(" -> [9/9] Fitting LSTM Sequence Network...")
    lstm = build_lstm_model(lookback)
    lstm.fit(X_train_lstm, y_train / 500.0, epochs=10, batch_size=1024, validation_split=0.1, verbose=0)
    lstm.save("models/lstm.keras")
    y_pred = lstm.predict(X_test_lstm, verbose=0).flatten() * 500.0
    metrics["LSTM Network"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # 5. Train LSTM Forecasting Model
    print("\n[Step 4/6] Training Time-Series Forecaster...")
    from ml.forecaster import train_forecaster
    train_forecaster(clean_df)
    
    # 6. Save Metrics
    print("\n[Step 5/6] Evaluating metrics on 20% holdout test set...")
    summary_data = {
        "preprocessing_summary": pipeline.summary,
        "metrics": metrics
    }
    
    with open("models/metrics.json", "w") as f:
        json.dump(summary_data, f, indent=4)
        
    print("\n[Step 6/6] Model training and evaluation successfully completed!")
    print("Saved all trained models and evaluation benchmarks to models/ directory.")

if __name__ == '__main__':
    main()
