import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.inference import build_realtime_input_frame


def _valid_payload():
    feature_cols = ["Temperature", "Humidity", "Hour", "Zone1_lag1"]
    feature_map = {
        "Temperature": 20.0,
        "Humidity": "55.0",
        "Hour": 12,
        "Zone1_lag1": 32344.97,
    }
    return feature_map, feature_cols


def test_build_realtime_input_frame_orders_and_coerces_numeric_values():
    feature_map, feature_cols = _valid_payload()

    frame = build_realtime_input_frame(feature_map, feature_cols)

    assert list(frame.columns) == feature_cols
    assert frame.shape == (1, len(feature_cols))
    assert frame["Humidity"].iloc[0] == 55.0


def test_build_realtime_input_frame_rejects_missing_feature():
    feature_map, feature_cols = _valid_payload()
    feature_map.pop("Hour")

    with pytest.raises(ValueError, match="tidak lengkap"):
        build_realtime_input_frame(feature_map, feature_cols)


def test_build_realtime_input_frame_rejects_non_numeric_value():
    feature_map, feature_cols = _valid_payload()
    feature_map["Temperature"] = "panas"

    with pytest.raises(ValueError, match="harus numerik"):
        build_realtime_input_frame(feature_map, feature_cols)


def test_build_realtime_input_frame_rejects_nan_and_infinite_values():
    feature_map, feature_cols = _valid_payload()
    feature_map["Temperature"] = np.nan

    with pytest.raises(ValueError, match="tidak boleh NaN"):
        build_realtime_input_frame(feature_map, feature_cols)

    feature_map, feature_cols = _valid_payload()
    feature_map["Temperature"] = np.inf

    with pytest.raises(ValueError, match="tidak boleh infinite"):
        build_realtime_input_frame(feature_map, feature_cols)
