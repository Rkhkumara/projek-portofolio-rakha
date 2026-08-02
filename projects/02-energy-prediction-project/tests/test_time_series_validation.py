import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.time_series_validation import run_time_series_cv, summarize_time_series_cv


def test_time_series_cv_returns_ordered_folds_and_summary():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "lag1": np.linspace(10, 20, 80),
            "temperature": rng.normal(25, 2, 80),
        }
    )
    y = X["lag1"] * 100 + X["temperature"] * 5

    results = run_time_series_cv(
        X,
        y,
        model_params={"n_estimators": 5, "max_depth": 2, "random_state": 42, "n_jobs": 1},
        n_splits=3,
    )
    summary = summarize_time_series_cv(results)

    assert len(results) == 3
    assert all(results["train_end"] < results["validation_start"])
    assert {"r2", "rmse", "mae", "mape"}.issubset(results.columns)
    assert "mean" in summary["mape"]
    assert "std" in summary["r2"]


def test_time_series_cv_rejects_invalid_shape():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        run_time_series_cv(X, y, model_params={"n_estimators": 2}, n_splits=2)
