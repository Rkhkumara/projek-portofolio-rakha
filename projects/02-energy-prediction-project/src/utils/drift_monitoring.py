from datetime import UTC, datetime

import numpy as np
import pandas as pd


DEFAULT_PSI_WARNING = 0.10
DEFAULT_PSI_CRITICAL = 0.25
DEFAULT_MAPE_WARNING = 5.0
DEFAULT_MAPE_CRITICAL = 10.0
DEFAULT_MAE_WARNING = 1500.0
DEFAULT_MAE_CRITICAL = 3000.0


def _safe_status_from_value(value, warning_threshold, critical_threshold):
    if value is None or np.isnan(value):
        return "stable"
    if value >= critical_threshold:
        return "critical"
    if value >= warning_threshold:
        return "warning"
    return "stable"


def _status_rank(status):
    return {"stable": 0, "warning": 1, "critical": 2}.get(status, 0)


def _max_status(*statuses):
    return max(statuses, key=_status_rank) if statuses else "stable"


def _to_datetime_str(ts):
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)


def build_feature_baseline(
    X_train,
    feature_cols,
    n_bins=10,
):
    baseline = {
        "version": 1,
        "created_at": _to_datetime_str(datetime.now(UTC)),
        "n_train": int(len(X_train)),
        "n_bins": int(n_bins),
        "features": {},
    }

    for col in feature_cols:
        if col not in X_train.columns:
            continue
        s = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if s.empty:
            continue

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(s.values, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            min_v = float(s.min())
            max_v = float(s.max())
            if np.isclose(min_v, max_v):
                max_v = min_v + 1e-9
            bins = np.linspace(min_v, max_v, n_bins + 1)
        expected_counts, _ = np.histogram(s.values, bins=bins)
        expected_props = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6 * len(expected_counts))

        baseline["features"][col] = {
            "bins": bins.tolist(),
            "expected_props": expected_props.tolist(),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    return baseline


def _compute_psi_from_hist(expected_props, actual_props):
    exp = np.asarray(expected_props, dtype=float)
    act = np.asarray(actual_props, dtype=float)
    exp = np.where(exp <= 0, 1e-6, exp)
    act = np.where(act <= 0, 1e-6, act)
    return float(np.sum((act - exp) * np.log(act / exp)))


def evaluate_input_drift(
    baseline_stats,
    recent_features_df,
    min_samples=24,
    psi_warning=DEFAULT_PSI_WARNING,
    psi_critical=DEFAULT_PSI_CRITICAL,
    monitored_features=None,
):
    if baseline_stats is None or "features" not in baseline_stats:
        return {
            "status": "insufficient_data",
            "reason": "baseline_missing",
            "n_samples": 0 if recent_features_df is None else int(len(recent_features_df)),
            "global_psi": None,
            "per_feature": [],
        }

    if recent_features_df is None or len(recent_features_df) < min_samples:
        return {
            "status": "insufficient_data",
            "reason": "recent_window_too_small",
            "n_samples": 0 if recent_features_df is None else int(len(recent_features_df)),
            "global_psi": None,
            "per_feature": [],
        }

    if monitored_features is not None:
        monitored_set = set(monitored_features)
        feature_items = [(c, m) for c, m in baseline_stats["features"].items() if c in monitored_set]
    else:
        feature_items = list(baseline_stats["features"].items())

    rows = []
    for col, meta in feature_items:
        if col not in recent_features_df.columns:
            continue
        s = pd.to_numeric(recent_features_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        bins = np.asarray(meta["bins"], dtype=float)
        clipped = np.clip(s.values, bins[0], bins[-1])
        counts, _ = np.histogram(clipped, bins=bins)
        actual_props = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        psi = _compute_psi_from_hist(meta["expected_props"], actual_props.tolist())
        status = _safe_status_from_value(psi, psi_warning, psi_critical)
        rows.append(
            {
                "feature": col,
                "psi": float(psi),
                "status": status,
            }
        )

    if not rows:
        return {
            "status": "insufficient_data",
            "reason": "no_compatible_features",
            "n_samples": int(len(recent_features_df)),
            "global_psi": None,
            "per_feature": [],
        }

    rows = sorted(rows, key=lambda x: x["psi"], reverse=True)
    global_psi = float(np.median([r["psi"] for r in rows]))
    global_status = _safe_status_from_value(global_psi, psi_warning, psi_critical)
    critical_ratio = float(np.mean([1.0 if r["status"] == "critical" else 0.0 for r in rows]))
    warning_ratio = float(np.mean([1.0 if r["status"] in ("warning", "critical") else 0.0 for r in rows]))
    if critical_ratio >= 0.35:
        global_status = "critical"
    elif global_status == "stable" and warning_ratio >= 0.50:
        global_status = "warning"

    return {
        "status": global_status,
        "reason": "ok",
        "n_samples": int(len(recent_features_df)),
        "global_psi": global_psi,
        "critical_ratio": critical_ratio,
        "warning_ratio": warning_ratio,
        "per_feature": rows,
    }


def evaluate_error_drift(
    recent_actuals=None,
    recent_predictions=None,
    min_samples=12,
    mape_warning=DEFAULT_MAPE_WARNING,
    mape_critical=DEFAULT_MAPE_CRITICAL,
    mae_warning=DEFAULT_MAE_WARNING,
    mae_critical=DEFAULT_MAE_CRITICAL,
):
    if recent_actuals is None or recent_predictions is None:
        return {
            "status": "pending_label",
            "reason": "labels_not_available",
            "n_samples": 0,
            "mape": None,
            "mae": None,
        }

    a = np.asarray(recent_actuals, dtype=float)
    p = np.asarray(recent_predictions, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a = a[mask]
    p = p[mask]
    if len(a) < min_samples:
        return {
            "status": "pending_label",
            "reason": "insufficient_labeled_samples",
            "n_samples": int(len(a)),
            "mape": None,
            "mae": None,
        }

    abs_err = np.abs(a - p)
    mae = float(np.mean(abs_err))
    denom = np.where(np.abs(a) < 1e-9, np.nan, np.abs(a))
    mape = float(np.nanmean(abs_err / denom) * 100.0)

    mape_status = _safe_status_from_value(mape, mape_warning, mape_critical)
    mae_status = _safe_status_from_value(mae, mae_warning, mae_critical)
    return {
        "status": _max_status(mape_status, mae_status),
        "reason": "ok",
        "n_samples": int(len(a)),
        "mape": mape,
        "mae": mae,
    }


def build_recommendation(global_status, input_status, error_status):
    if global_status == "critical":
        return "Drift tinggi. Prioritaskan investigasi fitur dominan drift dan lakukan retraining."
    if global_status == "warning":
        return "Mulai waspada. Pantau trend 24 jam dan siapkan trigger retraining jika memburuk."
    if input_status == "insufficient_data" or error_status == "pending_label":
        return "Data belum cukup untuk evaluasi penuh. Lanjutkan pengumpulan window realtime."
    return "Model stabil pada window saat ini."


def build_model_health_snapshot(
    baseline_stats,
    recent_features_df,
    recent_actuals=None,
    recent_predictions=None,
    min_input_samples=24,
    min_labeled_samples=12,
    timestamp=None,
    psi_warning=DEFAULT_PSI_WARNING,
    psi_critical=DEFAULT_PSI_CRITICAL,
    monitored_features=None,
    stable_mape_gate=2.0,
):
    input_drift = evaluate_input_drift(
        baseline_stats=baseline_stats,
        recent_features_df=recent_features_df,
        min_samples=min_input_samples,
        psi_warning=psi_warning,
        psi_critical=psi_critical,
        monitored_features=monitored_features,
    )
    error_drift = evaluate_error_drift(
        recent_actuals=recent_actuals,
        recent_predictions=recent_predictions,
        min_samples=min_labeled_samples,
    )

    statuses = [input_drift["status"]]
    if error_drift["status"] != "pending_label":
        statuses.append(error_drift["status"])
    global_status = _max_status(*statuses) if statuses else "stable"
    adjustment = "none"
    # Prevent over-alerting when covariate shift exists but prediction error remains healthy.
    if (
        input_drift.get("status") == "critical"
        and error_drift.get("status") == "stable"
        and error_drift.get("mape") is not None
        and float(error_drift["mape"]) <= float(stable_mape_gate)
    ):
        global_status = "warning"
        adjustment = "downgraded_by_error_gate"

    recommendation = build_recommendation(global_status, input_drift["status"], error_drift["status"])

    return {
        "timestamp": _to_datetime_str(timestamp or datetime.now(UTC)),
        "global_status": global_status,
        "status_adjustment": adjustment,
        "input_drift": input_drift,
        "error_drift": error_drift,
        "recommendation": recommendation,
    }


def evaluate_trigger_action(
    health_snapshot,
    warning_streak=0,
    critical_streak=0,
    retrain_mape_threshold=2.5,
    retrain_warning_streak=6,
    retrain_critical_streak=2,
):
    """
    Keputusan aksi otomatis berbasis health snapshot.
    - monitor  : status stabil / belum cukup data
    - alert    : warning, perlu pemantauan ketat
    - retrain  : trigger retraining otomatis
    """
    status = health_snapshot.get("global_status", "stable")
    err = health_snapshot.get("error_drift", {})
    mape = err.get("mape")

    should_retrain = False
    reason = "stable_state"
    action = "monitor"

    if status == "critical" and critical_streak >= retrain_critical_streak:
        should_retrain = True
        reason = "critical_streak_threshold"
    elif status == "warning" and warning_streak >= retrain_warning_streak:
        should_retrain = True
        reason = "warning_streak_threshold"
    elif mape is not None and float(mape) >= float(retrain_mape_threshold):
        should_retrain = True
        reason = "mape_threshold"

    if should_retrain:
        action = "retrain"
    elif status == "warning":
        action = "alert"
        reason = "warning_status"
    elif status == "critical":
        action = "alert"
        reason = "critical_status_waiting_streak"
    elif status == "insufficient_data":
        action = "monitor"
        reason = "insufficient_data"

    return {
        "action": action,
        "should_retrain": should_retrain,
        "reason": reason,
        "warning_streak": int(warning_streak),
        "critical_streak": int(critical_streak),
        "retrain_mape_threshold": float(retrain_mape_threshold),
        "retrain_warning_streak": int(retrain_warning_streak),
        "retrain_critical_streak": int(retrain_critical_streak),
    }
