import os
import sys
import time

import numpy as np
import pandas as pd

# Pastikan root proyek ada di sys.path agar import 'src' dapat ditemukan oleh pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.drift_monitoring import (
    build_feature_baseline,
    build_model_health_snapshot,
    evaluate_trigger_action,
    evaluate_error_drift,
    evaluate_input_drift,
)


def _make_feature_df(n=200, seed=42, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Temperature": rng.normal(25 + shift, 2.0, n),
            "Humidity": rng.normal(55 + shift, 4.0, n),
            "WindSpeed": rng.normal(2.5, 0.4, n),
            "Hour_sin": rng.uniform(-1, 1, n),
        }
    )


def test_input_drift_stable_for_similar_distribution():
    train_df = _make_feature_df(300, seed=1, shift=0.0)
    recent_df = train_df.sample(n=120, random_state=7).reset_index(drop=True)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    result = evaluate_input_drift(baseline, recent_df, min_samples=24)
    assert result["status"] == "stable"
    assert result["global_psi"] is not None
    assert len(result["per_feature"]) > 0


def test_input_drift_warning_or_critical_for_shifted_distribution():
    train_df = _make_feature_df(300, seed=1, shift=0.0)
    recent_df = _make_feature_df(120, seed=2, shift=8.0)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    result = evaluate_input_drift(baseline, recent_df, min_samples=24)
    assert result["status"] in ("warning", "critical")
    assert result["global_psi"] is not None


def test_input_drift_insufficient_data_safe_fallback():
    train_df = _make_feature_df(300, seed=1, shift=0.0)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    result = evaluate_input_drift(baseline, recent_features_df=pd.DataFrame(), min_samples=24)
    assert result["status"] == "insufficient_data"
    assert result["global_psi"] is None


def test_error_drift_pending_label_and_window_variants():
    pending = evaluate_error_drift(None, None, min_samples=12)
    assert pending["status"] == "pending_label"

    y_true = np.array([100, 110, 120, 130, 140, 150], dtype=float)
    y_pred = np.array([101, 108, 119, 131, 142, 148], dtype=float)
    short = evaluate_error_drift(y_true, y_pred, min_samples=12)
    assert short["status"] == "pending_label"

    y_true_long = np.tile(np.array([100, 120, 140, 160], dtype=float), 10)
    y_pred_long = y_true_long * 0.99
    ok = evaluate_error_drift(y_true_long, y_pred_long, min_samples=12)
    assert ok["status"] == "stable"
    assert ok["mape"] is not None
    assert ok["mae"] is not None


def test_model_health_snapshot_contract_and_perf():
    train_df = _make_feature_df(500, seed=11, shift=0.0)
    recent_df = _make_feature_df(120, seed=12, shift=0.4)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    actuals = recent_df["Temperature"].values * 1000 + 5000
    preds = actuals * 0.98

    t0 = time.perf_counter()
    snapshot = build_model_health_snapshot(
        baseline_stats=baseline,
        recent_features_df=recent_df,
        recent_actuals=actuals,
        recent_predictions=preds,
        min_input_samples=24,
        min_labeled_samples=12,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert "global_status" in snapshot
    assert "input_drift" in snapshot
    assert "error_drift" in snapshot
    assert "recommendation" in snapshot
    assert elapsed_ms < 1000.0


def test_input_drift_can_be_scoped_to_monitored_features():
    train_df = _make_feature_df(300, seed=1, shift=0.0)
    recent_df = _make_feature_df(120, seed=2, shift=0.0)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    result = evaluate_input_drift(
        baseline,
        recent_df,
        min_samples=24,
        monitored_features=["Temperature", "Humidity"],
    )
    names = [r["feature"] for r in result["per_feature"]]
    assert set(names) == {"Temperature", "Humidity"}


def test_health_status_can_downgrade_when_error_is_stable():
    train_df = _make_feature_df(400, seed=1, shift=0.0)
    recent_df = _make_feature_df(120, seed=2, shift=6.0)
    baseline = build_feature_baseline(train_df, feature_cols=list(train_df.columns), n_bins=10)

    # Strong feature drift, but very low prediction error.
    actuals = np.linspace(20000.0, 22000.0, 120)
    preds = actuals * 0.995

    snapshot = build_model_health_snapshot(
        baseline_stats=baseline,
        recent_features_df=recent_df,
        recent_actuals=actuals,
        recent_predictions=preds,
        min_input_samples=24,
        min_labeled_samples=12,
        stable_mape_gate=2.0,
    )
    assert snapshot["input_drift"]["status"] == "critical"
    assert snapshot["error_drift"]["status"] == "stable"
    assert snapshot["global_status"] == "warning"
    assert snapshot["status_adjustment"] == "downgraded_by_error_gate"


def test_trigger_action_rules():
    base = {
        "global_status": "stable",
        "error_drift": {"mape": 0.9},
    }
    d1 = evaluate_trigger_action(base, warning_streak=0, critical_streak=0)
    assert d1["action"] == "monitor"
    assert d1["should_retrain"] is False

    warning_case = {
        "global_status": "warning",
        "error_drift": {"mape": 1.2},
    }
    d2 = evaluate_trigger_action(warning_case, warning_streak=2, critical_streak=0, retrain_warning_streak=6)
    assert d2["action"] == "alert"
    assert d2["should_retrain"] is False

    d3 = evaluate_trigger_action(warning_case, warning_streak=6, critical_streak=0, retrain_warning_streak=6)
    assert d3["action"] == "retrain"
    assert d3["should_retrain"] is True

    critical_case = {
        "global_status": "critical",
        "error_drift": {"mape": 1.0},
    }
    d4 = evaluate_trigger_action(critical_case, warning_streak=0, critical_streak=2, retrain_critical_streak=2)
    assert d4["action"] == "retrain"
    assert d4["reason"] == "critical_streak_threshold"
