import pandas as pd
import numpy as np
import json
import logging
import os
import sys

# Pastikan root proyek ada di sys.path agar import 'src' bisa berjalan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from src.utils.config_loader import load_config
from src.utils.drift_monitoring import build_feature_baseline
from src.utils.time_series_validation import run_time_series_cv, summarize_time_series_cv
from src.data.preprocessing import clean_data, feature_engineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _resolve_project_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, path_value)


def _load_model_feature_cols(config: dict) -> list:
    feature_path = _resolve_project_path(config["training"]["feature_cols_save_path"])
    if os.path.exists(feature_path):
        try:
            with open(feature_path, "r", encoding="utf-8") as f:
                cols = json.load(f)
            if isinstance(cols, list) and cols:
                return cols
        except Exception:
            logging.warning("Feature cols model tidak bisa dibaca, fallback ke config.data.feature_cols.")
    return config["data"]["feature_cols"]


def _load_or_rebuild_processed_data(config: dict, selected_feature_cols: list) -> tuple[pd.DataFrame, str]:
    """Load processed data. If schema invalid, rebuild from raw Tetouan source."""
    processed_path = _resolve_project_path(config["data"]["processed_data_path"])
    raw_path = _resolve_project_path(config["data"]["raw_data_path"])
    target_col = config["data"]["target_col"]
    expected_cols = [target_col] + selected_feature_cols

    if os.path.exists(processed_path):
        df_processed = pd.read_csv(processed_path)
        if set(expected_cols).issubset(df_processed.columns):
            return df_processed, processed_path
        logging.warning(
            "Processed data schema tidak cocok untuk Tetouan. Rebuild dari raw source akan dilakukan."
        )

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data tidak ditemukan di '{raw_path}'. "
            "Periksa config.yaml pada data.raw_data_path."
        )

    df_raw = pd.read_csv(raw_path)
    if "Datetime" not in df_raw.columns:
        raise ValueError("Raw data harus memiliki kolom 'Datetime'.")
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    df_clean = clean_data(df_raw)
    df_feat = feature_engineering(df_clean, target_col=target_col)
    missing = [c for c in expected_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Kolom hasil feature engineering belum lengkap: {missing}")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_feat.to_csv(processed_path, index=False)
    logging.info(f"Processed data Tetouan direbuild dan disimpan ke: {processed_path}")
    return df_feat, processed_path


def calculate_metrics(y_true, y_pred) -> dict:
    """Menghitung R², RMSE, MAE, dan MAPE."""
    return {
        "r2":   r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def plot_feature_importance(model: XGBRegressor, feature_names: list, save_dir: str = "notebooks"):
    """Memvisualisasikan dan menyimpan plot feature importance."""
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax,
                palette='viridis')
    ax.set_title('XGBoost Feature Importance — Prediksi Konsumsi Energi', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score (Gain)')
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'feature_importance.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Plot feature importance tersimpan di '{save_path}'!")


def plot_actual_vs_predicted(y_test, y_pred, save_dir: str = "notebooks"):
    """Plot scatter actual vs predicted dan simpan ke file."""
    fig, ax = plt.subplots(figsize=(8, 8))

    sample_n = min(3000, len(y_test))
    idx = np.random.choice(len(y_test), sample_n, replace=False)
    ax.scatter(y_test[idx], y_pred[idx], alpha=0.3, color='#3b82f6', s=8)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

    r2 = r2_score(y_test, y_pred)
    ax.set_title(f'Actual vs Predicted (R²={r2:.4f})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Nilai Aktual (Watt)')
    ax.set_ylabel('Nilai Prediksi (Watt)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'actual_vs_predicted.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Plot actual vs predicted tersimpan di '{save_path}'!")


def main():
    config = load_config()
    selected_feature_cols = _load_model_feature_cols(config)

    model_path = _resolve_project_path(config["training"]["model_save_path"])

    if not os.path.exists(model_path):
        logging.error(
            f"Model tidak ditemukan di '{model_path}'. "
            "Jalankan 'python src/models/train.py' terlebih dahulu."
        )
        return

    try:
        df, processed_path = _load_or_rebuild_processed_data(config, selected_feature_cols)
    except Exception as e:
        logging.error(f"Gagal menyiapkan data evaluasi: {e}")
        return
    logging.info(f"Dataset evaluasi dimuat: {df.shape[0]:,} baris dari '{processed_path}'")

    target_col   = config["data"]["target_col"]
    feature_cols = selected_feature_cols
    test_size    = config["training"]["test_size"]

    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df[target_col]

    # Recreate split yang sama seperti saat training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=config["training"].get("shuffle", False),
    )

    # Load model
    model = XGBRegressor()
    model.load_model(model_path)
    logging.info("Model berhasil dimuat.")

    # Prediksi & evaluasi
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.values, y_pred)

    logging.info("=" * 50)
    logging.info("📊 HASIL EVALUASI TEST SET")
    logging.info("=" * 50)
    logging.info(f"  R²   : {metrics['r2']:.4f}")
    logging.info(f"  RMSE : {metrics['rmse']:,.2f} Watt")
    logging.info(f"  MAE  : {metrics['mae']:,.2f} Watt")
    logging.info(f"  MAPE : {metrics['mape']:.2f}%")
    logging.info("=" * 50)

    # Simpan metrik ke JSON
    import datetime as _dt
    metrics_path = _resolve_project_path(config["training"].get("metrics_save_path", "models/tetouan_metrics.json"))
    metrics_save = {
        "r2":   round(metrics["r2"], 4),
        "rmse": round(metrics["rmse"], 2),
        "mae":  round(metrics["mae"], 2),
        "mape": round(metrics["mape"], 2),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(available),
        "split_strategy": "chronological_holdout",
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d"),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    logging.info(f"Metrik tersimpan di: {metrics_path}")

    # Validasi tambahan: TimeSeriesSplit menjaga urutan temporal antar fold.
    cv_folds = int(config["training"].get("cv_folds", 5))
    cv_path = _resolve_project_path(
        config["training"].get("timeseries_cv_path", "models/tetouan_timeseries_cv.json")
    )
    logging.info("Menjalankan TimeSeriesSplit CV (%s fold)...", cv_folds)
    cv_results = run_time_series_cv(
        X=X,
        y=y,
        model_params=config["model_params"]["xgboost"],
        n_splits=cv_folds,
    )
    cv_payload = {
        "method": "TimeSeriesSplit",
        "n_splits": cv_folds,
        "description": "Expanding-window validation; validation fold selalu setelah training fold.",
        "folds": cv_results.round(6).to_dict(orient="records"),
        "summary": summarize_time_series_cv(cv_results),
    }
    os.makedirs(os.path.dirname(cv_path), exist_ok=True)
    with open(cv_path, "w") as f:
        json.dump(cv_payload, f, indent=2)
    logging.info(f"Hasil TimeSeriesSplit tersimpan di: {cv_path}")

    # Simpan baseline distribusi training untuk monitoring drift realtime
    drift_path = _resolve_project_path(
        config["training"].get("drift_baseline_path", "models/tetouan_drift_baseline.json")
    )
    os.makedirs(os.path.dirname(drift_path), exist_ok=True)
    drift_baseline = build_feature_baseline(X_train=X_train, feature_cols=available, n_bins=10)
    with open(drift_path, "w") as f:
        json.dump(drift_baseline, f, indent=2)
    logging.info(f"Baseline drift tersimpan di: {drift_path}")

    # Visualisasi
    plot_feature_importance(model, available)
    plot_actual_vs_predicted(y_test.values, y_pred)


if __name__ == "__main__":
    main()
