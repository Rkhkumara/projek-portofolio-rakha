"""Validation helpers for model inference payloads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_realtime_input_frame(feature_map: dict, feature_cols: list[str]) -> pd.DataFrame:
    """
    Build a one-row inference DataFrame with strict feature order and numeric checks.

    XGBoost accepts pandas DataFrames, but it will fail late or behave unclearly if
    required columns are missing, non-numeric, NaN, or infinite. This helper fails
    early with a readable validation error.
    """
    if not feature_cols:
        raise ValueError("feature_cols tidak boleh kosong.")

    missing = [col for col in feature_cols if col not in feature_map]
    if missing:
        raise ValueError(f"Fitur inference tidak lengkap: {missing}")

    frame = pd.DataFrame([{col: feature_map[col] for col in feature_cols}], columns=feature_cols)
    frame = frame.apply(pd.to_numeric, errors="coerce")

    invalid_cols = [col for col in feature_cols if frame[col].isna().any()]
    if invalid_cols:
        raise ValueError(f"Fitur inference harus numerik dan tidak boleh NaN: {invalid_cols}")

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad_cols = [
            feature_cols[idx]
            for idx in np.where(~np.isfinite(values))[1].tolist()
        ]
        raise ValueError(f"Fitur inference tidak boleh infinite: {sorted(set(bad_cols))}")

    return frame
