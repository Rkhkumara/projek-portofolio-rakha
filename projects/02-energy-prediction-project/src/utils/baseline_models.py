import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

from src.utils.ablation import feature_engineering, _evaluate, ALL_FEATURE_COLS, TARGET

def load_saved_model():
    """Load the existing XGBoost model without retraining."""
    model = XGBRegressor()
    model_path = os.path.join("models", "model_tetouan_xgb.json")
    model.load_model(model_path)
    return model

def train_and_evaluate_baselines(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Train baselines and evaluate against the saved XGBoost model.
    """
    results = []
    
    # --- 1. Baseline 1: Naive (Persistence) ---
    # Prediksi = nilai pada timestep sebelumnya (Zone1_lag1)
    y_pred_naive = X_test['Zone1_lag1'].ffill().bfill().values
    metrics_naive = _evaluate(y_test, y_pred_naive)
    results.append({
        'Model': 'Naive (Persistence)',
        **metrics_naive
    })
    
    # --- 2. Baseline 2: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    metrics_lr = _evaluate(y_test, y_pred_lr)
    results.append({
        'Model': 'Linear Regression',
        **metrics_lr
    })
    
    # --- 3. XGBoost (Proposed) ---
    model_xgb = load_saved_model()
    y_pred_xgb = model_xgb.predict(X_test)
    metrics_xgb = _evaluate(y_test, y_pred_xgb)
    results.append({
        'Model': 'XGBoost (Proposed)',
        **metrics_xgb
    })
    
    return pd.DataFrame(results)
