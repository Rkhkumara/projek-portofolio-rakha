import json
import os
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Pastikan root proyek ada di sys.path agar import 'src' dapat ditemukan oleh pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.realtime_features import compute_zone1_history_features


def _load_realtime_assets():
    df = pd.read_csv("dataset/powerconsumption.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)

    with open("models/tetouan_feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    model = XGBRegressor()
    model.load_model("models/model_tetouan_xgb.json")
    return df, feature_cols, model


def _build_single_realtime_row(df, feature_cols):
    ref = df.iloc[-1]
    history_features, meta = compute_zone1_history_features(
        df_raw=df,
        month_input=ref["Datetime"].month,
        day_input=ref["Datetime"].day,
        hour_input=ref["Datetime"].hour,
        fallback_load=float(df["PowerConsumption_Zone1"].mean()),
    )

    hour = int(ref["Datetime"].hour)
    dow = int(ref["Datetime"].dayofweek)
    month = int(ref["Datetime"].month)
    is_weekend = 1 if dow >= 5 else 0

    feature_map = {
        "Temperature": float(ref["Temperature"]),
        "Humidity": float(ref["Humidity"]),
        "WindSpeed": float(ref["WindSpeed"]),
        "GeneralDiffuseFlows": float(ref["GeneralDiffuseFlows"]),
        "DiffuseFlows": float(ref["DiffuseFlows"]),
        "Hour": hour,
        "DayOfWeek": dow,
        "DayOfMonth": int(ref["Datetime"].day),
        "Month": month,
        "IsWeekend": is_weekend,
        "Hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "Hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "DayOfWeek_sin": float(np.sin(2 * np.pi * dow / 7)),
        "DayOfWeek_cos": float(np.cos(2 * np.pi * dow / 7)),
        "Month_sin": float(np.sin(2 * np.pi * month / 12)),
        "Month_cos": float(np.cos(2 * np.pi * month / 12)),
    }
    feature_map.update(history_features)

    X = pd.DataFrame([[feature_map[c] for c in feature_cols]], columns=feature_cols)
    return X, meta


def test_realtime_payload_uses_observed_history():
    df, feature_cols, _ = _load_realtime_assets()
    X, meta = _build_single_realtime_row(df, feature_cols)

    assert meta["source_datetime"] is not None
    assert str(meta["mode"]).startswith("observed")
    assert list(X.columns) == feature_cols
    assert X.shape == (1, len(feature_cols))
    assert not X.isna().any().any()

    source_idx = int(df.index[df["Datetime"] == pd.to_datetime(meta["source_datetime"])][0])
    expected_lag24 = float(df["PowerConsumption_Zone1"].iloc[source_idx - 144])
    assert float(X.iloc[0]["Zone1_lag24"]) == expected_lag24


def test_realtime_inference_latency_and_finite_output():
    df, feature_cols, model = _load_realtime_assets()
    X, _ = _build_single_realtime_row(df, feature_cols)

    y = float(model.predict(X)[0])
    assert np.isfinite(y)
    assert y > 0

    t0 = time.perf_counter()
    for _ in range(50):
        _ = model.predict(X)
    avg_ms = ((time.perf_counter() - t0) * 1000.0) / 50.0

    # Ambang longgar agar stabil lintas mesin CI/lokal
    assert avg_ms < 50.0
