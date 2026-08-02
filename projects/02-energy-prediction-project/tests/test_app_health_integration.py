import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Pastikan root proyek ada di sys.path agar import 'src' dapat ditemukan oleh pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import feature_engineering
from src.utils.drift_monitoring import build_feature_baseline, build_model_health_snapshot


def test_app_health_payload_with_real_model_pipeline():
    df = pd.read_csv("dataset/powerconsumption.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    feat_df = feature_engineering(df)

    with open("models/tetouan_feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    X = feat_df[feature_cols]
    y = feat_df["PowerConsumption_Zone1"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    baseline = build_feature_baseline(X_train=X_train, feature_cols=feature_cols, n_bins=10)

    model = XGBRegressor()
    model.load_model("models/model_tetouan_xgb.json")

    recent_features = X_test.tail(72)
    recent_actuals = y_test.tail(72).values
    recent_preds = model.predict(recent_features)

    snapshot = build_model_health_snapshot(
        baseline_stats=baseline,
        recent_features_df=recent_features,
        recent_actuals=recent_actuals,
        recent_predictions=recent_preds,
        min_input_samples=24,
        min_labeled_samples=12,
        timestamp=feat_df["Datetime"].iloc[-1],
    )

    assert snapshot["global_status"] in ("stable", "warning", "critical")
    assert snapshot["input_drift"]["status"] in ("stable", "warning", "critical", "insufficient_data")
    assert snapshot["error_drift"]["status"] in ("stable", "warning", "critical", "pending_label")
    assert isinstance(snapshot["recommendation"], str)
