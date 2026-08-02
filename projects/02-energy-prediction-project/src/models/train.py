import pandas as pd
import numpy as np
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_model(X_fit, y_fit, X_val, y_val, config):
    """Train XGBoost Regressor dengan validation set internal untuk early stopping."""

    params = config["model_params"]["xgboost"]

    model = XGBRegressor(
        n_estimators        = params.get("n_estimators", 500),
        max_depth           = params.get("max_depth", 6),
        learning_rate       = params.get("learning_rate", 0.05),
        subsample           = params.get("subsample", 0.8),
        colsample_bytree    = params.get("colsample_bytree", 0.8),
        min_child_weight    = params.get("min_child_weight", 5),
        gamma               = params.get("gamma", 0.1),
        reg_alpha           = params.get("reg_alpha", 0.1),
        reg_lambda          = params.get("reg_lambda", 1.0),
        random_state        = params.get("random_state", 42),
        n_jobs              = -1,
        early_stopping_rounds = 30,
        eval_metric         = 'rmse',
    )

    logging.info("Training XGBoost...")
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    logging.info(f"Training selesai! Best iteration: {model.best_iteration}")

    return model


def main():
    config = load_config()
    logging.info("Memulai Training Pipeline (Tetouan Power Consumption)...")

    processed_path = config["data"]["processed_data_path"]
    if not os.path.exists(processed_path):
        logging.error(
            f"Data tidak ditemukan di '{processed_path}'. "
            "Jalankan 'python src/data/preprocessing.py' terlebih dahulu."
        )
        return

    df = pd.read_csv(processed_path)
    logging.info(f"Dataset dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")

    target_col   = config["data"]["target_col"]
    feature_cols = config["data"]["feature_cols"]
    test_size    = config["training"]["test_size"]

    # Pastikan semua fitur tersedia
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning(f"Fitur tidak ditemukan (diabaikan): {missing}")

    X = df[available]
    y = df[target_col]

    # Holdout kronologis: test set tetap tidak disentuh saat training/early stopping.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=config["training"].get("shuffle", False),
    )
    logging.info(f"Split: Train={X_train.shape[0]:,} | Test={X_test.shape[0]:,}")

    validation_size = config["training"].get("validation_size", 0.1)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_size,
        shuffle=False,
    )
    logging.info(
        "Early stopping split: Fit=%s | Validation=%s | Test unseen=%s",
        f"{X_fit.shape[0]:,}",
        f"{X_val.shape[0]:,}",
        f"{X_test.shape[0]:,}",
    )

    # Train model
    model = train_and_save_model(X_fit, y_fit, X_val, y_val, config)

    # Simpan model
    model_path = config["training"]["model_save_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    logging.info(f"Model tersimpan di: {model_path}")

    # Simpan daftar fitur
    feature_path = config["training"]["feature_cols_save_path"]
    with open(feature_path, 'w') as f:
        json.dump(available, f, indent=2)
    logging.info(f"Feature list tersimpan di: {feature_path}")


if __name__ == "__main__":
    main()
