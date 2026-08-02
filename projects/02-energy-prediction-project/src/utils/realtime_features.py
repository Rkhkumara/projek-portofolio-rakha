import pandas as pd


def compute_zone1_history_features(df_raw, month_input, day_input, hour_input, minute_input, fallback_load, year=2017):
    """
    Bangun 6 fitur historis (lag/rolling) untuk inference realtime Zone 1.

    Jika histori tidak memadai/invalid, nilai akan fallback ke fallback_load.
    """
    fallback = {
        "Zone1_lag1": float(fallback_load),
        "Zone1_lag6": float(fallback_load),
        "Zone1_lag24": float(fallback_load),
        "Zone1_roll3": float(fallback_load),
        "Zone1_roll6": float(fallback_load),
        "Zone1_roll24": float(fallback_load),
    }

    if df_raw is None or df_raw.empty:
        return fallback, {"mode": "fallback_empty_df", "source_datetime": None}

    required_cols = {"Datetime", "PowerConsumption_Zone1"}
    if not required_cols.issubset(df_raw.columns):
        return fallback, {"mode": "fallback_missing_columns", "source_datetime": None}

    work = df_raw[["Datetime", "PowerConsumption_Zone1"]].copy()
    work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
    work = work.dropna(subset=["Datetime", "PowerConsumption_Zone1"]).sort_values("Datetime").reset_index(drop=True)

    if len(work) < 145:
        return fallback, {"mode": "fallback_insufficient_history", "source_datetime": None}

    try:
        ref_ts = pd.Timestamp(year=int(year), month=int(month_input), day=int(day_input), hour=int(hour_input), minute=int(minute_input))
    except Exception:
        ref_ts = None

    if ref_ts is None:
        idx = len(work) - 1
        mode = "observed_latest"
    else:
        idx = int(work["Datetime"].searchsorted(ref_ts, side="right") - 1)
        mode = "observed_reference"

    idx = min(max(idx, 144), len(work) - 1)
    zone = work["PowerConsumption_Zone1"]

    features = {
        "Zone1_lag1": float(zone.iloc[idx - 1]),
        "Zone1_lag6": float(zone.iloc[idx - 6]),
        "Zone1_lag24": float(zone.iloc[idx - 144]),
        "Zone1_roll3": float(zone.iloc[idx - 3:idx].mean()),
        "Zone1_roll6": float(zone.iloc[idx - 6:idx].mean()),
        "Zone1_roll24": float(zone.iloc[idx - 24:idx].mean()),
    }
    meta = {
        "mode": mode,
        "source_datetime": work["Datetime"].iloc[idx],
    }
    return features, meta
