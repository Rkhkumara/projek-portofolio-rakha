"""Time-series cross validation helpers for regression models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


def _normalise_params(model_params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(model_params or {})
    params.pop("early_stopping_rounds", None)
    params.pop("eval_metric", None)
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    return params


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return standard regression metrics used across the project."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def run_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict[str, Any] | None,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Evaluate XGBoost with expanding-window TimeSeriesSplit.

    Each validation fold occurs after its corresponding training fold, so this
    checks robustness without random KFold leakage.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(X) <= n_splits:
        raise ValueError("Dataset is too small for the requested TimeSeriesSplit.")

    params = _normalise_params(model_params)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        model = XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=0)
        y_pred = model.predict(X.iloc[val_idx])
        metrics = calculate_regression_metrics(y.iloc[val_idx], y_pred)
        rows.append(
            {
                "fold": fold,
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "validation_start": int(val_idx[0]),
                "validation_end": int(val_idx[-1]),
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_time_series_cv(cv_results: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarise fold metrics as mean/std pairs."""
    metric_cols = ["r2", "rmse", "mae", "mape"]
    if cv_results.empty:
        return {}

    summary: dict[str, dict[str, float]] = {}
    for col in metric_cols:
        values = cv_results[col].astype(float)
        summary[col] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return summary
