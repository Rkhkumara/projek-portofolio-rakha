import os
import sys
import pandas as pd
import numpy as np
import pytest

# Pastikan root proyek ada di sys.path agar import 'src' dapat ditemukan oleh pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import clean_data, feature_engineering


# ============================
# Tests untuk clean_data()
# ============================

def test_clean_data_removes_nulls():
    """Unit test: clean_data menginterpolasi dan menghapus nilai null."""
    df = pd.DataFrame({
        'feature_1': [25.0, np.nan, 30.0],
        'PowerConsumption_Zone1': [10000.0, 11000.0, 12000.0]
    })

    result_df = clean_data(df)

    # Memastikan tidak ada sisa null value
    assert result_df.isnull().sum().sum() == 0

    # Nilai interpolasi linier di indeks 1 harusnya persis di tengah 25 dan 30
    assert result_df['feature_1'].iloc[1] == 27.5


def test_clean_data_no_mutation():
    """Unit test: fungsi tidak melakukan in-place mutation pada DataFrame asli."""
    df_original = pd.DataFrame({
        'feature_1': [25.0],
        'PowerConsumption_Zone1': [10000.0]
    })
    df_copy = df_original.copy()

    clean_data(df_copy)

    assert df_original.equals(pd.DataFrame({
        'feature_1': [25.0],
        'PowerConsumption_Zone1': [10000.0]
    }))


def test_clean_data_handles_empty_dataset():
    """Dataset kosong tidak boleh membuat cleaning crash."""
    df = pd.DataFrame(columns=[
        'Datetime', 'Temperature', 'Humidity', 'PowerConsumption_Zone1'
    ])

    result = clean_data(df)

    assert result.empty
    assert list(result.columns) == list(df.columns)


def test_clean_data_extreme_missing_values_interpolated():
    """Missing value ekstrem tetap diisi selama ada anchor numerik yang valid."""
    df = pd.DataFrame({
        'Datetime': pd.date_range('2017-01-01', periods=5, freq='10min'),
        'Temperature': [np.nan, np.nan, 20.0, np.nan, np.nan],
        'Humidity': [50.0, np.nan, np.nan, np.nan, 70.0],
        'PowerConsumption_Zone1': [10000.0, np.nan, 12000.0, np.nan, 14000.0],
    })

    result = clean_data(df)

    assert result.isnull().sum().sum() == 0
    assert result['Temperature'].tolist() == [20.0] * 5
    assert result['Humidity'].iloc[0] == 50.0
    assert result['Humidity'].iloc[-1] == 70.0


def test_clean_data_drops_exact_duplicate_rows():
    """Baris duplikat exact dibuang agar split time-series tidak menghitung sampel ganda."""
    df = pd.DataFrame({
        'Datetime': [pd.Timestamp('2017-01-01 00:00')] * 2 + [pd.Timestamp('2017-01-01 00:10')],
        'Temperature': [20.0, 20.0, 21.0],
        'PowerConsumption_Zone1': [10000.0, 10000.0, 10100.0],
    })

    result = clean_data(df)

    assert len(result) == 2


def test_clean_data_coerces_numeric_strings_and_drops_invalid_types():
    """Kolom numerik yang terbaca sebagai string dikonversi sebelum interpolasi."""
    df = pd.DataFrame({
        'Datetime': pd.date_range('2017-01-01', periods=3, freq='10min'),
        'Temperature': ['20.0', 'bad-value', '22.0'],
        'PowerConsumption_Zone1': ['10000', 'bad-value', '12000'],
    })

    result = clean_data(df)

    assert result.isnull().sum().sum() == 0
    assert result['Temperature'].iloc[1] == 21.0
    assert result['PowerConsumption_Zone1'].iloc[1] == 11000.0


# ============================
# Tests untuk feature_engineering()
# ============================

def _make_sample_df(n=200):
    """Buat sample DataFrame yang menyerupai dataset Tetouan."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2017-01-01', periods=n, freq='10min')
    return pd.DataFrame({
        'Datetime':               dates,
        'Temperature':            rng.uniform(5, 35, n),
        'Humidity':               rng.uniform(20, 90, n),
        'WindSpeed':              rng.uniform(0, 10, n),
        'GeneralDiffuseFlows':    rng.uniform(0, 500, n),
        'DiffuseFlows':           rng.uniform(0, 200, n),
        'PowerConsumption_Zone1': rng.uniform(14000, 50000, n),
        'PowerConsumption_Zone2': rng.uniform(15000, 45000, n),
        'PowerConsumption_Zone3': rng.uniform(10000, 48000, n),
    })


def test_feature_engineering_adds_columns():
    """Setelah feature engineering, kolom temporal dan lag harus ada."""
    df = _make_sample_df(200)
    result = feature_engineering(df)

    expected_cols = [
        'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'IsWeekend',
        'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
        'Month_sin', 'Month_cos',
        'Zone1_lag1', 'Zone1_lag6', 'Zone1_lag24',
        'Zone1_roll3', 'Zone1_roll6', 'Zone1_roll24',
    ]
    for col in expected_cols:
        assert col in result.columns, f"Kolom '{col}' tidak ditemukan setelah feature engineering"


def test_feature_engineering_empty_dataset_returns_empty():
    """Feature engineering pada dataset kosong harus aman."""
    df = pd.DataFrame(columns=['Datetime', 'PowerConsumption_Zone1'])

    result = feature_engineering(df)

    assert result.empty


def test_feature_engineering_incomplete_columns_raises_clear_error():
    """Kolom wajib yang hilang harus menghasilkan error eksplisit."""
    missing_target = pd.DataFrame({
        'Datetime': pd.date_range('2017-01-01', periods=3, freq='10min')
    })
    missing_datetime = pd.DataFrame({
        'PowerConsumption_Zone1': [10000.0, 11000.0, 12000.0]
    })

    with pytest.raises(ValueError, match='Kolom wajib'):
        feature_engineering(missing_target)

    with pytest.raises(ValueError, match='Kolom wajib'):
        feature_engineering(missing_datetime)


def test_feature_engineering_invalid_datetime_type_is_dropped_safely():
    """Datetime tidak valid tidak boleh lolos menjadi fitur temporal rusak."""
    df = _make_sample_df(150)
    df['Datetime'] = df['Datetime'].astype(object)
    df.loc[:4, 'Datetime'] = 'not-a-date'

    result = feature_engineering(df)

    assert not result.empty
    assert result['Datetime'].notna().all()


def test_feature_engineering_no_nulls():
    """Setelah feature engineering, tidak boleh ada null."""
    df = _make_sample_df(200)
    result = feature_engineering(df)

    assert result.isnull().sum().sum() == 0


def test_feature_engineering_rows_reduced():
    """Jumlah baris harus berkurang karena shift/rolling menghasilkan NaN yang di-drop."""
    df = _make_sample_df(200)
    result = feature_engineering(df)

    assert len(result) < len(df)


def test_feature_engineering_cyclic_range():
    """Nilai cyclic encoding harus berada di rentang [-1, 1]."""
    df = _make_sample_df(200)
    result = feature_engineering(df)

    for col in ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos']:
        assert result[col].min() >= -1.0, f"{col} memiliki nilai < -1"
        assert result[col].max() <= 1.0,  f"{col} memiliki nilai > 1"
