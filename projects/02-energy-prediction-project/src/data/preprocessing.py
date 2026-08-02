import pandas as pd
import numpy as np
import logging
import os
import sys

# Pastikan root proyek ada di sys.path agar import 'src' bisa berjalan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplikat, tipe numerik, dan missing values dengan interpolasi linier."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # Kolom numerik dari CSV kadang terbaca sebagai object karena nilai rusak.
    for col in df.columns:
        if col == "Datetime" or pd.api.types.is_numeric_dtype(df[col]):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted

    # Handle missing values: interpolasi linier untuk kolom numerik jika ada
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    df = df.dropna()
    return df


def feature_engineering(df: pd.DataFrame, target_col: str = 'PowerConsumption_Zone1') -> pd.DataFrame:
    """Tambahkan fitur temporal, cyclic encoding, lag, dan rolling statistics.

    Proses ini HARUS identik dengan yang ada di notebook training
    (notebooks/02_modeling_tetouan.ipynb) agar fitur yang masuk ke model konsisten.
    """
    df = df.copy()
    required_cols = {"Datetime", target_col}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia untuk feature engineering: {missing}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", target_col]).sort_values("Datetime").reset_index(drop=True)
    if df.empty:
        return df

    # --- Buang kolom zona yang tidak digunakan dalam penelitian ini ---
    # Penelitian ini hanya berfokus pada Zone 1 sebagai target prediksi.
    # Zone 2 dan Zone 3 dikeluarkan secara eksplisit agar tidak masuk sebagai fitur model.
    cols_to_drop = [c for c in ['PowerConsumption_Zone2', 'PowerConsumption_Zone3'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logging.info(f"Kolom zona tidak dipakai dibuang: {cols_to_drop}")


    # --- Fitur Temporal ---
    df['Hour']       = df['Datetime'].dt.hour
    df['DayOfWeek']  = df['Datetime'].dt.dayofweek   # 0=Senin, 6=Minggu
    # df['DayOfMonth'] = df['Datetime'].dt.day  # Dieliminasi karena SHAP value rendah
    df['Month']      = df['Datetime'].dt.month
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

    # --- Cyclic Encoding (agar model menangkap siklus harian/mingguan/bulanan) ---
    df['Hour_sin']      = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos']      = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin']     = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos']     = np.cos(2 * np.pi * df['Month'] / 12)

    # --- Lag Features ---
    df['Zone1_lag1']  = df[target_col].shift(1)
    df['Zone1_lag6']  = df[target_col].shift(6)    # 1 jam lalu (10 menit × 6)
    df['Zone1_lag24'] = df[target_col].shift(144)   # 1 hari lalu (10 menit × 144)

    # --- Rolling Statistics ---
    df['Zone1_roll3']  = df[target_col].rolling(3).mean()
    df['Zone1_roll6']  = df[target_col].rolling(6).mean()
    df['Zone1_roll24'] = df[target_col].rolling(24).mean()

    # Drop NaN yang dihasilkan oleh shift/rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main():
    config = load_config()
    logging.info("Memulai proses Data Preprocessing (Tetouan Power Consumption)...")

    raw_data_path  = config["data"]["raw_data_path"]
    processed_path = config["data"]["processed_data_path"]
    target_col     = config["data"]["target_col"]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    if not os.path.exists(raw_data_path):
        logging.error(f"File '{raw_data_path}' tidak ditemukan.")
        return

    df = pd.read_csv(raw_data_path)
    logging.info(f"Dataset dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")

    # Parse datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=False)
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Bersihkan data
    df_cleaned = clean_data(df)
    logging.info(f"Setelah cleaning: {df_cleaned.shape[0]:,} baris")

    # Feature engineering
    df_feat = feature_engineering(df_cleaned, target_col=target_col)
    logging.info(f"Setelah feature engineering: {df_feat.shape[0]:,} baris × {df_feat.shape[1]} kolom")

    df_feat.to_csv(processed_path, index=False)
    logging.info(f"Data terproses berhasil disimpan di: {processed_path}")


if __name__ == "__main__":
    main()
