"""
Ablation Study — Validasi bahwa performa model bukan dari data leakage.

Modul ini melatih 3 varian model XGBoost dengan subset fitur berbeda
menggunakan hyperparameter dan chronological split yang identik,
lalu membandingkan metrik evaluasi.

Variant A: Full model (21 fitur)
Variant B: Drop 2 fitur SHAP teratas (Zone1_roll3, Zone1_lag1)
Variant C: Drop SEMUA lag & rolling features (fitur non-temporal saja)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error,
    mean_absolute_error, mean_absolute_percentage_error
)
from src.utils.time_series_validation import _normalise_params


# --- Hyperparameter akan di-load dari config ---

ALL_FEATURE_COLS = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'Month_sin', 'Month_cos',
    'Zone1_lag1', 'Zone1_lag6', 'Zone1_lag24',
    'Zone1_roll3', 'Zone1_roll6', 'Zone1_roll24',
]

TARGET = 'PowerConsumption_Zone1'


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur temporal, cyclic, lag, dan rolling — identik dengan notebook."""
    df = df.copy()

    # Fitur Temporal
    df['Hour']       = df['Datetime'].dt.hour
    df['DayOfWeek']  = df['Datetime'].dt.dayofweek
    df['DayOfMonth'] = df['Datetime'].dt.day
    df['Month']      = df['Datetime'].dt.month
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

    # Cyclic Encoding
    df['Hour_sin']      = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos']      = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)

    # Lag Features (observasi masa lalu saja)
    df['Zone1_lag1']  = df[TARGET].shift(1)
    df['Zone1_lag6']  = df[TARGET].shift(6)     # 1 jam
    df['Zone1_lag24'] = df[TARGET].shift(144)    # 1 hari (144 × 10 menit)

    # Rolling Statistics
    df['Zone1_roll3']  = df[TARGET].rolling(3).mean()
    df['Zone1_roll6']  = df[TARGET].rolling(6).mean()
    df['Zone1_roll24'] = df[TARGET].rolling(24).mean()

    # Drop NaN dari shift/rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _evaluate(y_true, y_pred):
    """Hitung 4 metrik evaluasi standar."""
    return {
        'R²': round(r2_score(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'MAE': round(mean_absolute_error(y_true, y_pred), 2),
        'MAPE (%)': round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def _train_and_evaluate(X_train, X_test, y_train, y_test, feature_subset, model_params=None):
    """Latih XGBoost pada subset fitur dan kembalikan metrik."""
    params = _normalise_params(model_params)
    mdl = XGBRegressor(**params)
    mdl.fit(X_train[feature_subset], y_train, verbose=0)
    y_pred = mdl.predict(X_test[feature_subset])
    return _evaluate(y_test, y_pred)


def run_ablation_study(df_raw: pd.DataFrame, model_params=None, test_size=0.2) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Jalankan ablation study dengan 3 varian model.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Dataset mentah dari powerconsumption.csv (sudah ada kolom Datetime).
    model_params : dict
        Hyperparameter untuk model XGBoost (diambil dari config.yaml).
    test_size : float
        Proporsi test set (diambil dari config.yaml).

    Returns
    -------
    pd.DataFrame
        Tabel perbandingan dengan kolom:
        [Skenario, Fitur Dihapus, Jumlah Fitur, R², RMSE, MAE, MAPE (%)]
    """
    # Feature engineering identik dengan pipeline training
    df_feat = feature_engineering(df_raw)

    X = df_feat[ALL_FEATURE_COLS]
    y = df_feat[TARGET]

    # Chronological split tanpa shuffle — identik dengan notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    results = []

    # --- Variant A: Full model (21 fitur) ---
    feats_a = ALL_FEATURE_COLS.copy()
    metrics_a = _train_and_evaluate(X_train, X_test, y_train, y_test, feats_a, model_params)
    results.append({
        'Skenario': 'A — Full Model',
        'Fitur Dihapus': '—',
        'Jumlah Fitur': len(feats_a),
        **metrics_a,
    })

    # --- Variant B: Drop 2 fitur SHAP teratas ---
    drop_b = ['Zone1_roll3', 'Zone1_lag1']
    feats_b = [f for f in ALL_FEATURE_COLS if f not in drop_b]
    metrics_b = _train_and_evaluate(X_train, X_test, y_train, y_test, feats_b, model_params)
    results.append({
        'Skenario': 'B — Tanpa Top-2 SHAP',
        'Fitur Dihapus': ', '.join(drop_b),
        'Jumlah Fitur': len(feats_b),
        **metrics_b,
    })

    # --- Variant C: Drop SEMUA lag & rolling ---
    drop_c = [f for f in ALL_FEATURE_COLS if 'lag' in f or 'roll' in f]
    feats_c = [f for f in ALL_FEATURE_COLS if f not in drop_c]
    metrics_c = _train_and_evaluate(X_train, X_test, y_train, y_test, feats_c, model_params)
    results.append({
        'Skenario': 'C — Tanpa Lag & Rolling',
        'Fitur Dihapus': ', '.join(drop_c),
        'Jumlah Fitur': len(feats_c),
        **metrics_c,
    })

    metadata = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return pd.DataFrame(results), metadata
