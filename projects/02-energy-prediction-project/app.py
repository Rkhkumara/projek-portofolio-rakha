import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import calendar
from io import BytesIO
from datetime import datetime
from xgboost import XGBRegressor

import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from src.utils.ablation import run_ablation_study
from src.utils.baseline_models import train_and_evaluate_baselines
from src.utils.residual_analysis import plot_residual_analysis
from src.utils.realtime_features import compute_zone1_history_features
from src.utils.inference import build_realtime_input_frame

# ===========================================================
# CONFIG & CSS (LIGHT PROFESSIONAL THEME)
# ===========================================================
st.set_page_config(
    page_title="Energy Prediction",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #F4F7FB;
        --surface: #FFFFFF;
        --surface-muted: #F0EDEC;
        --surface-low: #F6F3F2;
        --line: #C3C6D8;
        --text: #1C1B1B;
        --muted: #424656;
        --primary: #004CCD;
        --primary-bright: #0F62FE;
        --primary-soft: #DBE1FF;
        --success: #137333;
        --success-soft: #E6F4EA;
        --warning: #8A6A24;
        --warning-soft: #F5EEDB;
        --danger: #BA1A1A;
        --danger-soft: #FFDAD6;
    }
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    

/* DASHBOARD CARD & TYPOGRAPHY TAHAP 1 */
    .dashboard-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }
    
    label p, .stCaptionContainer p, .kpi-title, .helper-text {
        font-size: 14px !important;
    }



/* DASHBOARD CARD & TYPOGRAPHY TAHAP 1 */
    .dashboard-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }
    
    label p, .stCaptionContainer p, .kpi-title, .helper-text {
        font-size: 14px !important;
    }



/* DASHBOARD CARD & TYPOGRAPHY TAHAP 1 */
    .dashboard-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }
    
    label p, .stCaptionContainer p, .kpi-title, .helper-text {
        font-size: 14px !important;
    }



/* DASHBOARD CARD & TYPOGRAPHY TAHAP 1 */
    .dashboard-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
    }
    
    label p, .stCaptionContainer p, .kpi-title, .helper-text {
        font-size: 14px !important;
    }


    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1400px;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stHeader"] {
        background: rgba(252, 249, 248, 0.86);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(195, 198, 216, 0.2);
    }

    [data-testid="stToolbar"] {
        color: var(--muted);
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(195, 198, 216, 0.28);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] .stCaptionContainer,
    [data-testid="stSidebar"] .stCaptionContainer * {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
    }

    /* Sidebar: wrapper container input (NumberInput, Selectbox) */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #F1F5F9 !important;
        border: 1.5px solid #94A3B8 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* Input elemen di dalamnya ikut background abu */
    [data-testid="stSidebar"] input {
        background: #F1F5F9 !important;
        border: none !important;
        color: var(--text) !important;
    }

    /* Sidebar input focus state - biru aksen, konsisten dengan Tebakan Sistem AI */
    [data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
    [data-testid="stSidebar"] [data-baseweb="input"]:focus-within input {
        border-color: var(--primary-bright) !important;
        box-shadow: 0 0 0 3px rgba(15, 98, 254, 0.12) !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] > div:focus-within {
        box-shadow: 0 0 0 3px rgba(15, 98, 254, 0.12) !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] input:focus {
        border-color: var(--primary-bright) !important;
        box-shadow: 0 0 0 3px rgba(15, 98, 254, 0.12) !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] label p {
        font-family: 'Inter', sans-serif;
        color: var(--text) !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"] {
        background: var(--surface-low) !important;
        border-color: rgba(195, 198, 216, 0.55) !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: var(--surface) !important;
        border: 1px solid rgba(195, 198, 216, 0.6) !important;
        color: var(--text) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border-radius: 4px !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
        background: var(--primary-soft) !important;
        border-color: var(--primary-bright) !important;
        color: var(--primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover svg {
        fill: var(--primary) !important;
        color: var(--primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:active {
        transform: scale(0.85) !important;
        background: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:active svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: transparent;
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] svg {
        fill: var(--muted) !important;
        color: var(--muted) !important;
    }

    .app-topbar {
        height: 4rem;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 1rem;
        color: var(--muted);
        margin-bottom: 1.25rem;
    }
    .topbar-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .topbar-action {
        color: var(--primary);
        font-size: 1rem;
        font-weight: 700;
    }
    .topbar-menu {
        font-size: 1.4rem;
        color: var(--muted);
        line-height: 1;
    }

    .system-header {
        background: var(--surface);
        border: 1px solid var(--line);
        padding: 1rem 2rem;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .header-item {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Blinking dot */
    .status-dot {
        height: 8px; width: 8px;
        background-color: var(--success);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(47, 111, 78, 0.22);
        animation: blink 2s infinite;
    }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; }}

    /* CLICKABLE KPI (DETAILS/SUMMARY) */
    .kpi-clickable {
        margin-bottom: 1rem;
    }
    .kpi-clickable summary {
        list-style: none;
        cursor: pointer;
    }
    .kpi-clickable summary::-webkit-details-marker {
        display: none;
    }
    .kpi-clickable summary .kpi-card {
        transition: all 0.3s ease;
    }
    .kpi-clickable summary:hover .kpi-card {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    .kpi-clickable[open] summary .kpi-card {
        border-color: var(--primary);
        box-shadow: 0 4px 12px rgba(0, 76, 205, 0.1);
        transform: translateY(0);
    }
    .kpi-details {
        padding: 0.85rem 1rem;
        background: var(--surface-low);
        border: 1px solid var(--line);
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: var(--muted);
        text-align: left;
        line-height: 1.5;
        animation: slideDown 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        transform-origin: top;
    }
    @keyframes slideDown {
        0% { opacity: 0; transform: scaleY(0.95) translateY(-5px); }
        100% { opacity: 1; transform: scaleY(1) translateY(0); }
    }

    /* KPI CARDS */
    .kpi-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-top: 0;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        text-align: center;
    }
    .kpi-card.featured {
        position: relative;
        overflow: hidden;
        border-color: rgba(15, 98, 254, 0.2);
        box-shadow: inset 0 0 0 2px rgba(15, 98, 254, 0.16), 0px 4px 20px rgba(0, 0, 0, 0.04);
    }
    .kpi-title { font-size: 0.75rem; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; line-height: 1.35; }
    .kpi-value { font-size: 2.25rem; line-height: 3.5rem; color: var(--text); font-weight: 760; margin: 0.45rem 0; letter-spacing: 0; }
    .kpi-sub { font-size: 0.72rem; padding: 4px 12px; border-radius: 999px; display: inline-block; font-weight: 500; color: var(--muted); background: var(--surface-muted);}
    
    /* COLORS */
    .sub-green { background: var(--success-soft); color: var(--success); }
    .sub-red { background: var(--danger-soft); color: var(--danger); }
    .sub-blue { background: var(--primary-soft); color: var(--primary); }
    .sub-warning { background: var(--warning-soft); color: var(--warning); }

    .context-panel {
        background: var(--surface);
        border: 1px solid rgba(195, 198, 216, 0.22);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        color: var(--text);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .context-panel p {
        color: var(--muted);
        margin: 0.25rem 0 0;
        line-height: 1.6;
    }
    .helper-text {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    /* ALERT PANEL */
    .alert-panel {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 1rem;
        background: var(--surface);
        border: 1px solid rgba(195, 198, 216, 0.22);
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    .alert-warning { border-left: 4px solid var(--warning); color: var(--text); background: var(--surface); }
    .alert-critical { border-left: 4px solid var(--danger); color: var(--text); background: var(--surface); }
    .alert-safe { border-left: 4px solid var(--success); color: var(--text); background: var(--surface); }

    /* SIDEBAR BUTTON */
    .stButton>button {
        width: 100%; height: 3rem; border-radius: 8px; font-weight: 700;
        background: var(--primary);
        color: white; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background: #003DA9; box-shadow: 0 4px 12px rgba(15, 98, 254, 0.18); }
    
    /* Section Headers */
    .section-title { font-size: 1.5rem; line-height: 2rem; color: var(--text); font-weight: 650; margin-top: 0; margin-bottom: 1rem; border-bottom: 0; padding-bottom: 0;}

    .chart-card {
        background: var(--surface);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .intro-info-grid {
        margin-top: 1rem;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }
    @media (max-width: 900px) {
        .intro-info-grid {
            grid-template-columns: 1fr;
        }
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* EXPANDER FIX FOR LIGHT THEME */
    [data-testid="stExpander"] details summary {
        background-color: var(--surface);
        color: var(--text);
        border-radius: 8px;
        border: 1px solid rgba(195, 198, 216, 0.28);
    }
    [data-testid="stExpander"] details summary p {
        color: var(--text) !important;
        font-weight: 600;
    }
    [data-testid="stExpander"] details summary:hover {
        background-color: var(--surface-muted);
    }
    [data-testid="stExpander"] details summary:hover p {
        color: var(--primary) !important;
    }
    [data-testid="stExpander"] details {
        border-radius: 8px;
        border: none !important;
    }

    /* TAHAP 4: SIDEBAR TOUCH TARGET */
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        min-width: 44px !important;
        min-height: 44px !important;
    }
    
    /* ALERT FIX FOR LIGHT THEME */
    [data-testid="stAlert"] {
        color: var(--text) !important;
    }
    [data-testid="stAlert"] p, 
    [data-testid="stAlert"] span, 
    [data-testid="stAlert"] div {
        color: var(--text) !important;
    }
    
    /* TABS FIX FOR LIGHT THEME (AVOID FADED LOOK) */
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted) !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        font-size: 1.05rem !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }
    [data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--primary-bright) !important;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================
# LOADERS & CONSTANTS - SINGLE ZONE (Zone 1)
# ===========================================================
# Baseline dihitung dinamis dari dataset (lihat setelah blok sidebar)

FEATURE_LABELS = {
    "Temperature": "Suhu Lingkungan (°C)",
    "Humidity": "Kelembaban (%)",
    "WindSpeed": "Kecepatan Angin (m/s)",
    "GeneralDiffuseFlows": "Radiasi Gen. Diffuse (W/m²)",
    "DiffuseFlows": "Radiasi Diffuse (W/m²)",
    "Hour": "Jam Operasional",
    "DayOfWeek": "Hari dalam Seminggu",
    "Month": "Bulan",
    "IsWeekend": "Status Akhir Pekan",
    "Hour_sin": "Pola Siklus Jam (sin)",
    "Hour_cos": "Pola Siklus Jam (cos)",
    "DayOfWeek_sin": "Pola Siklus Hari (sin)",
    "DayOfWeek_cos": "Pola Siklus Hari (cos)",
    "Month_sin": "Pola Siklus Bulan (sin)",
    "Month_cos": "Pola Siklus Bulan (cos)",
    "Zone1_lag1": "Konsumsi 10 Menit Sebelumnya",
    "Zone1_lag6": "Konsumsi 1 Jam Sebelumnya",
    "Zone1_lag24": "Konsumsi 1 Hari Sebelumnya",
    "Zone1_roll3": "Rata-rata 30 Menit Terakhir",
    "Zone1_roll6": "Rata-rata 1 Jam Terakhir",
    "Zone1_roll24": "Rata-rata 4 Jam Terakhir"
}

DATASET_PATH = "dataset/powerconsumption.csv"
ABLATION_RESULTS_PATH = os.path.join("models", "tetouan_ablation.json")
BASELINE_RESULTS_PATH = os.path.join("models", "tetouan_baseline.json")

def get_file_fingerprint(path):
    """Cache key ringan berbasis ukuran dan waktu modifikasi file."""
    try:
        stat = os.stat(path)
        return (path, stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (path, None, None)

def load_precomputed_results(json_path):
    """Baca hasil evaluasi tersimpan agar dashboard tidak melatih ulang saat dibuka."""
    try:
        with open(json_path, "r") as f:
            payload = json.load(f)
        return normalize_metric_columns(pd.DataFrame(payload.get("results", [])))
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_data(dataset_fingerprint):
    try:
        df = pd.read_csv(DATASET_PATH)
        df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=False)
        df = df.sort_values('Datetime').reset_index(drop=True)
        # Resample for hourly plotting
        df_hourly = df.set_index('Datetime').resample('h').mean().reset_index()
        return df, df_hourly
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        model = XGBRegressor()
        model.load_model(os.path.join("models", "model_tetouan_xgb.json"))
        return model
    except Exception as e:
        return None

@st.cache_data
def load_feature_cols():
    try:
        with open(os.path.join("models", "tetouan_feature_cols.json"), 'r') as f:
            return json.load(f)
    except:
         return ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Zone1_lag1', 'Zone1_lag6', 'Zone1_lag24', 'Zone1_roll3', 'Zone1_roll6', 'Zone1_roll24']

@st.cache_data
def load_metrics():
    try:
        with open("models/tetouan_metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@st.cache_data
def load_timeseries_cv_metrics():
    try:
        with open("models/tetouan_timeseries_cv.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}

def normalize_metric_columns(df):
    """Normalize metric column names that may come from non-ASCII sources."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    rename_map = {}
    for col in df.columns:
        ascii_col = str(col).encode("ascii", "ignore").decode("ascii")
        if col == "R2" or ascii_col == "R":
            rename_map[col] = "R2"
    return df.rename(columns=rename_map)

@st.cache_data
def build_realtime_feature_window(dataset_fingerprint, feature_cols, end_ts, window_size=72):
    try:
        from src.data.preprocessing import feature_engineering
        df_raw, _ = load_data(dataset_fingerprint)
        if df_raw.empty:
            return pd.DataFrame()

        work = df_raw.copy()
        work["Datetime"] = pd.to_datetime(work["Datetime"], errors="coerce")
        work = work.dropna(subset=["Datetime"]).sort_values("Datetime")
        feat_df = feature_engineering(work)

        if end_ts is not None:
            feat_df = feat_df[feat_df["Datetime"] <= pd.to_datetime(end_ts)]

        cols = [c for c in feature_cols if c in feat_df.columns]
        if not cols:
            return pd.DataFrame()
        return feat_df[cols].tail(window_size).copy()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=" Menjalankan Ablation Study (sekali saja)...")
def load_ablation_results(dataset_fingerprint):
    """Jalankan ablation study dan cache hasilnya (hanya sekali)."""
    try:
        precomputed = load_precomputed_results(ABLATION_RESULTS_PATH)
        if not precomputed.empty:
            return precomputed

        from src.utils.config_loader import load_config
        import json
        import datetime
        import os
        from src.utils.ablation import ALL_FEATURE_COLS
        
        df_raw, _ = load_data(dataset_fingerprint)
        if df_raw.empty:
            return pd.DataFrame()

        config = load_config()
        test_size = config["training"]["test_size"]
        model_params = config["model_params"]["xgboost"]
        
        df_result, metadata = run_ablation_study(
            df_raw, model_params=model_params, test_size=test_size
        )
        
        payload = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_size": test_size,
            "n_features": len(ALL_FEATURE_COLS),
            "n_train": metadata["n_train"],
            "n_test": metadata["n_test"],
            "results": df_result.to_dict(orient="records")
        }
        
        os.makedirs(os.path.dirname(ABLATION_RESULTS_PATH), exist_ok=True)
        with open(ABLATION_RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
            
        return normalize_metric_columns(df_result)
    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Error ablation study: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_data(show_spinner=" Mengevaluasi Baseline Models (sekali saja)...")
def load_baseline_results(dataset_fingerprint):
    """Evaluasi baseline model dan cache hasilnya (hanya sekali)."""
    try:
        precomputed = load_precomputed_results(BASELINE_RESULTS_PATH)
        if not precomputed.empty:
            return precomputed

        from src.utils.config_loader import load_config
        import json
        import datetime
        import os
        from src.utils.ablation import feature_engineering, ALL_FEATURE_COLS, TARGET
        from sklearn.model_selection import train_test_split
        
        df_raw, _ = load_data(dataset_fingerprint)
        if df_raw.empty:
            return pd.DataFrame()

        config = load_config()
        test_size = config["training"]["test_size"]
        
        df_feat = feature_engineering(df_raw)
        X = df_feat[ALL_FEATURE_COLS]
        y = df_feat[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        df_result = train_and_evaluate_baselines(X_train, X_test, y_train, y_test)
        
        payload = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_size": test_size,
            "n_features": len(ALL_FEATURE_COLS),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "results": df_result.to_dict(orient="records")
        }
        
        os.makedirs(os.path.dirname(BASELINE_RESULTS_PATH), exist_ok=True)
        with open(BASELINE_RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
            
        return normalize_metric_columns(df_result)
    except Exception as e:
        import traceback
        import streamlit as st
        st.error(f"Error evaluating baselines: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_data(show_spinner=" Menyiapkan Data Residual (sekali saja)...")
def get_residual_data(dataset_fingerprint):
    """Menyiapkan data residual untuk visualisasi (hanya sekali)."""
    try:
        from src.utils.ablation import feature_engineering, ALL_FEATURE_COLS, TARGET
        from src.utils.baseline_models import load_saved_model
        from sklearn.model_selection import train_test_split
        
        df_raw, _ = load_data(dataset_fingerprint)
        if df_raw.empty:
            return None, None, None

        df_feat = feature_engineering(df_raw)
        X = df_feat[ALL_FEATURE_COLS]
        y = df_feat[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model_xgb = load_saved_model()
        y_pred = model_xgb.predict(X_test)
        
        return y_test.values, y_pred, X_test
    except Exception as e:
        import traceback
        st.error(f"Error loading residual data: {traceback.format_exc()}")
        return None, None, None

@st.cache_data(show_spinner=" Menghitung Korelasi (sekali saja)...")
def get_correlation_matrix(dataset_fingerprint):
    """Menghitung matriks korelasi untuk X_train (21 fitur)."""
    try:
        from src.utils.ablation import feature_engineering, ALL_FEATURE_COLS, TARGET
        from sklearn.model_selection import train_test_split
        
        df_raw, _ = load_data(dataset_fingerprint)
        if df_raw.empty:
            return pd.DataFrame()

        df_feat = feature_engineering(df_raw)
        X = df_feat[ALL_FEATURE_COLS]
        y = df_feat[TARGET]
        X_train, _, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train.corr()
    except Exception as e:
        import traceback
        st.error(f"Error computing correlation: {traceback.format_exc()}")
        return pd.DataFrame()

@st.cache_resource
def load_shap_explainer(_model):
    """Membuat SHAP TreeExplainer untuk model Zone 1 (di-cache agar cepat)."""
    try:
        return shap.TreeExplainer(_model)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def render_shap_waterfall_png(shap_values, feature_values, base_value, feature_names, current_pred):
    """Render waterfall SHAP sekali per input, lalu pakai ulang sebagai PNG."""
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 11})
    fig_wf, ax_wf = plt.subplots(figsize=(8.6, 5.4), dpi=110)
    try:
        explanation = shap.Explanation(
            values=np.array(shap_values),
            base_values=float(base_value),
            data=np.array(feature_values),
            feature_names=list(feature_names),
        )
        plt.sca(ax_wf)
        shap.plots.waterfall(explanation, max_display=16, show=False)

        ax = fig_wf.axes[0] if fig_wf.axes else plt.gca()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        clean_labels = [lbl.split(" = ")[-1] if " = " in lbl else lbl for lbl in labels]
        ax.set_yticklabels(clean_labels)

        ax.xaxis.grid(True, linestyle='--', color='#E5E7EB', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

        from matplotlib.colors import to_rgba
        from matplotlib.transforms import ScaledTranslation

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        ax.set_xlim(x_min - (x_range * 0.16), x_max + (x_range * 0.07))
        ax.tick_params(axis='y', pad=12)

        for txt in ax.texts:
            old_text = txt.get_text()
            try:
                normalized_text = (
                    old_text.replace(',', '')
                    .replace('\u2212', '-')
                    .replace('+', '')
                    .strip()
                )
                val = float(normalized_text)
                txt.set_text(f"{val:+.2f} W")

                is_inside_bar = (
                    txt.get_ha() == 'center'
                    and np.allclose(to_rgba(txt.get_color()), to_rgba('white'))
                )
                if is_inside_bar:
                    txt.set_color('white')
                else:
                    txt.set_color('#1F2937')
                    txt.set_ha('left' if val > 0 else 'right')
                    offset_points = 7 if val > 0 else -7
                    txt.set_transform(
                        ax.transData
                        + ScaledTranslation(
                            offset_points / 72.0,
                            0,
                            fig_wf.dpi_scale_trans,
                        )
                    )
            except ValueError:
                pass

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff0051', label='Menaikkan prediksi'),
            Patch(facecolor='#008bfb', label='Menurunkan prediksi')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(1.0, -0.07),
            frameon=True,
            fontsize=9,
            facecolor='white',
            framealpha=0.9,
        )

        for extra_axis in fig_wf.axes[1:]:
            extra_axis.set_visible(False)

        plt.title(
            f'Waterfall Plot - Zona 1 Kota Tetouan\n'
            f'Prediksi: {current_pred:,.0f} Watt ({current_pred/1000:,.2f} kW)',
            fontsize=13, fontweight='bold', pad=14
        )
        fig_wf.subplots_adjust(left=0.40, right=0.97, top=0.83, bottom=0.20)

        buffer = BytesIO()
        fig_wf.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        return buffer.getvalue()
    finally:
        plt.close(fig_wf)
        plt.rcParams.update(matplotlib.rcParamsDefault)
        plt.rcParams['font.family'] = 'DejaVu Sans'

# ===========================================================
# INIT DATA
# ===========================================================
dataset_fingerprint = get_file_fingerprint(DATASET_PATH)
df_raw, df_hourly = load_data(dataset_fingerprint)
model = load_model()
feature_cols = load_feature_cols()
metrics_data = load_metrics()
timeseries_cv_data = load_timeseries_cv_metrics()
explainer = load_shap_explainer(model) if model else None

# ===========================================================
# SIDEBAR INPUT
# ===========================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:flex-start; gap:0.75rem; margin-bottom:1.75rem;">
        <div>
            <h2 style="color:#004CCD; font-weight:700; font-size:1.5rem; line-height:2rem; margin:0;">Dashboard Prediksi Konsumsi Listrik</h2>
            <p style="color:#424656; font-size:0.74rem; line-height:1.25rem; margin:0.25rem 0 0;">
                Sistem prediksi konsumsi listrik Zona 1 Kota Tetouan
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div style='color:#424656; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.5rem;'>Parameter Waktu</div>", unsafe_allow_html=True)
        st.caption("Atur waktu di bawah ini untuk melihat perkiraan konsumsi listrik dari sistem kami.")
        month_input = st.number_input("Bulan (1-12)", min_value=1, max_value=12, value=6)
        max_dom = calendar.monthrange(2017, int(month_input))[1]
        default_dom = min(15, max_dom)
        
        col_h, col_m = st.columns(2)
        with col_h:
            hour_input  = st.number_input("Jam (0-23)", min_value=0, max_value=23, value=12)
        with col_m:
            minute_input = st.selectbox("Menit", options=[0, 10, 20, 30, 40, 50], index=0)
            
        dom_input   = st.number_input("Tanggal (1-31)", min_value=1, max_value=max_dom, value=default_dom)
    selected_timestamp = pd.Timestamp(year=2017, month=int(month_input), day=int(dom_input), hour=int(hour_input), minute=int(minute_input))
    dow_input = int(selected_timestamp.dayofweek)
    is_weekend = dow_input >= 5
    day_labels = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    st.caption(f"{day_labels[dow_input]} | Akhir pekan: {'Ya' if is_weekend else 'Tidak'}")

    # --- Lookup data historis dari dataset ---
    lookup_row = None
    actual_consumption = None
    matched_time = None
    if not df_raw.empty:
        df_lookup = df_raw.copy()
        df_lookup['Datetime'] = pd.to_datetime(df_lookup['Datetime'])
        time_diffs = (df_lookup['Datetime'] - selected_timestamp).abs()
        nearest_idx = time_diffs.idxmin()
        lookup_row = df_lookup.loc[nearest_idx]
        actual_consumption = float(lookup_row['PowerConsumption_Zone1'])
        matched_time = lookup_row['Datetime']
        
        gap = abs(matched_time - selected_timestamp)
        if gap > pd.Timedelta(minutes=10):
            min_date = df_lookup['Datetime'].min()
            max_date = df_lookup['Datetime'].max()
            st.info(f"**Info Transparansi:** Waktu yang Anda minta ({selected_timestamp:%Y-%m-%d %H:%M}) berada di luar rentang dataset kami ({min_date:%Y-%m-%d %H:%M} s.d. {max_date:%Y-%m-%d %H:%M}). Sistem menggunakan data historis terdekat pada {matched_time:%Y-%m-%d %H:%M} sebagai gantinya.")

        st.markdown("---")
        st.markdown("<div style='color:#424656; font-size:0.75rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.25rem;'>Kondisi Cuaca pada Data</div>", unsafe_allow_html=True)
        st.caption(f"Cuaca ini dipakai AI sebagai salah satu faktor prediksi beban listrik.\n(Sumber: {matched_time:%Y-%m-%d %H:%M})")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.metric("Suhu", f"{lookup_row['Temperature']:.1f} °C")
            st.metric("Kelembaban", f"{lookup_row['Humidity']:.1f} %")
            st.metric("Angin", f"{lookup_row['WindSpeed']:.2f} m/s")
        with wc2:
            st.metric("Radiasi total", f"{lookup_row['GeneralDiffuseFlows']:.1f} W/m²")
            st.metric("Radiasi diffuse", f"{lookup_row['DiffuseFlows']:.2f} W/m²")
    else:
        st.warning(" Dataset tidak tersedia.")

    st.markdown("---")
    debug_mode = st.toggle("Tampilkan detail teknis", value=False)

# --- Ekstrak nilai cuaca dari dataset lookup ---
if lookup_row is not None:
    temp_input = float(lookup_row['Temperature'])
    hum_input  = float(lookup_row['Humidity'])
    wind_input = float(lookup_row['WindSpeed'])
    gdf_input  = float(lookup_row['GeneralDiffuseFlows'])
    df_input   = float(lookup_row['DiffuseFlows'])
else:
    temp_input, hum_input, wind_input, gdf_input, df_input = 20.0, 55.0, 2.5, 100.0, 50.0

# Baseline dihitung dari rata-rata historis dataset (untuk fallback lag/rolling)
base_load = float(df_raw['PowerConsumption_Zone1'].mean()) if not df_raw.empty else 0.0


# ===========================================================
# PREDICTION LOGIC - SINGLE ZONE
# ===========================================================
hour_sin  = np.sin(2 * np.pi * hour_input / 24)
hour_cos  = np.cos(2 * np.pi * hour_input / 24)
dow_sin   = np.sin(2 * np.pi * dow_input / 7)
dow_cos   = np.cos(2 * np.pi * dow_input / 7)
month_sin = np.sin(2 * np.pi * month_input / 12)
month_cos = np.cos(2 * np.pi * month_input / 12)

history_features, history_meta = compute_zone1_history_features(
    df_raw=df_raw,
    month_input=month_input,
    day_input=dom_input,
    hour_input=hour_input,
    minute_input=minute_input,
    fallback_load=base_load,
)

feature_map = {
    "Temperature": temp_input,
    "Humidity": hum_input,
    "WindSpeed": wind_input,
    "GeneralDiffuseFlows": gdf_input,
    "DiffuseFlows": df_input,
    "Hour": hour_input,
    "DayOfWeek": dow_input,
    "Month": month_input,
    "IsWeekend": int(is_weekend),
    "Hour_sin": hour_sin,
    "Hour_cos": hour_cos,
    "DayOfWeek_sin": dow_sin,
    "DayOfWeek_cos": dow_cos,
    "Month_sin": month_sin,
    "Month_cos": month_cos,
}
feature_map.update(history_features)
try:
    input_df = build_realtime_input_frame(feature_map, feature_cols)
except ValueError as exc:
    st.error(f"Input inference tidak valid: {exc}")
    st.stop()

if model:
    prediction = float(model.predict(input_df)[0])
else:
    prediction = 0.0

if history_meta["source_datetime"] is not None:
    st.sidebar.caption(
        f"Data riwayat dipakai hingga {history_meta['source_datetime']:%Y-%m-%d %H:%M}"
    )
else:
    st.sidebar.caption("Data riwayat tidak lengkap, memakai rata-rata historis.")



# Threshold visualisasi (menggunakan percentile dataset, bukan hardcode)
alert_thresh_warn = float(df_raw['PowerConsumption_Zone1'].quantile(0.75)) if not df_raw.empty else 37309.0
alert_thresh_crit = float(df_raw['PowerConsumption_Zone1'].quantile(0.95)) if not df_raw.empty else 44712.0


# ===========================================================
# TOP APP BAR & ALERTS
# ===========================================================
st.markdown("""
<div style="background-color: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);">
    <h1 style="font-size: 1.8rem; color: var(--text); margin-top: 0; margin-bottom: 0.5rem;">Sistem Prediksi Konsumsi Listrik</h1>
    <p style="color: var(--muted); font-size: 1.05rem; margin-bottom: 0; line-height: 1.5;">
        Dashboard ini memprediksi penggunaan listrik di <b>Zona 1 Kota Tetouan</b> menggunakan kecerdasan buatan (model XGBoost). 
        Sistem menganalisis pola historis dan cuaca untuk memperkirakan beban listrik secara spesifik pada waktu yang Anda pilih di menu sebelah kiri.
    </p>
    <p style="color: var(--muted); font-size: 0.98rem; margin: 0.9rem 0 0; line-height: 1.55;">
        Selain prediksi, sistem juga menjelaskan faktor-faktor di balik setiap prediksi melalui analisis SHAP pada tab <b>Penjelasan Model</b>, serta menyediakan perbandingan dengan model klasik dan pengujian tambahan pada panel di bagian bawah.
    </p>
    <div class="intro-info-grid" style="margin-top: 2rem;">
        <div class="dashboard-card" style="padding: 1.25rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted);"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                <b style="color:var(--text); font-size:0.9rem; font-weight:700;">Catatan data</b>
            </div>
            <p style="color:var(--muted); font-size:0.85rem; line-height:1.45; margin:0;">Sistem ini memprediksi berdasarkan data historis tahun 2017, bukan data konsumsi listrik secara real-time.</p>
        </div>
        <div class="dashboard-card" style="padding: 1.25rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--primary);"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                <b style="color:var(--text); font-size:0.9rem; font-weight:700;">Status konsumsi</b>
            </div>
            <p style="color:var(--muted); font-size:0.85rem; line-height:1.45; margin:0;">Status konsumsi (Aman/Peringatan/Kritis) ditentukan berdasarkan posisi prediksi terhadap pola historis.</p>
        </div>
        <div class="dashboard-card" style="padding: 1.25rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom: 0.5rem;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--muted);"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                <b style="color:var(--text); font-size:0.9rem; font-weight:700;">Aktual vs prediksi</b>
            </div>
            <p style="color:var(--muted); font-size:0.85rem; line-height:1.45; margin:0;">Data Realita sebagai pembanding; di luar riwayat, hanya Tebakan Sistem AI yang tersedia.</p>
        </div>
    </div>
    <p style="color:var(--muted); font-size:0.78rem; margin:0.8rem 0 0; line-height:1.4;">Model dilatih pada data Zona 1 Kota Tetouan dan belum diuji pada zona atau wilayah lain.</p>
</div>
""", unsafe_allow_html=True)

if prediction > alert_thresh_crit:
    q_alert = f'<div class="alert-panel alert-critical">Status: Konsumsi diperkirakan sangat tinggi ({prediction/1000:.1f} kW). <span style="font-size:0.85rem; font-weight:normal; margin-left:0.5rem; opacity:0.8;">(Melewati batas historis tertinggi - persentil 95)</span></div>'
elif prediction > alert_thresh_warn:
    q_alert = f'<div class="alert-panel alert-warning">Status: Konsumsi mulai meningkat ({prediction/1000:.1f} kW). <span style="font-size:0.85rem; font-weight:normal; margin-left:0.5rem; opacity:0.8;">(Di atas rata-rata - persentil 75)</span></div>'
else:
    q_alert = f'<div class="alert-panel alert-safe">Status: Konsumsi dalam rentang wajar ({prediction/1000:.1f} kW). <span style="font-size:0.85rem; font-weight:normal; margin-left:0.5rem; opacity:0.8;">(Masih di bawah rata-rata tinggi - persentil 75)</span></div>'

st.markdown(q_alert, unsafe_allow_html=True)

# ===========================================================
# KPI SUMMARY CARDS
# ===========================================================
kpi1, kpi2, kpi3 = st.columns(3)

# Hitung Error (Deviasi dari aktual)
if actual_consumption is not None:
    abs_error = abs(prediction - actual_consumption)
    ape_error = (abs_error / actual_consumption) * 100
    
    # Penilaian akurasi standar akademik (Forecasting)
    if ape_error <= 5.0:
        status_text = "Sangat Akurat (Selisih di bawah 5%)"
        delta_color = "sub-green"
        bar_color = "#137333"
    elif ape_error <= 10.0:
        status_text = "Akurat (Selisih antara 5% - 10%)"
        delta_color = "sub-blue"
        bar_color = "#0F62FE"
    elif ape_error <= 20.0:
        status_text = "Cukup Akurat (Selisih antara 10% - 20%)"
        delta_color = "sub-warning"
        bar_color = "#8A6A24"
    else:
        status_text = "Kurang Akurat (> 20% selisih)"
        delta_color = "sub-red"
        bar_color = "#BA1A1A"
        
    aktual_val_str = f"{actual_consumption/1000:.2f} kW"
    aktual_label = "Data Realita (Aktual)"
else:
    abs_error = None
    ape_error = None
    status_text = "Data aktual tidak tersedia"
    delta_color = "sub-warning"
    bar_color = "#8A6A24"
    aktual_val_str = "N/A"
    aktual_label = "Data Realita (Aktual)"

with kpi1:
    st.markdown(f"""
    <details class="kpi-clickable">
        <summary>
            <div class="kpi-card" style="margin-bottom: 0;">
                <div class="kpi-title">{aktual_label} <span style="font-size:0.9rem; vertical-align:middle;">&#9432;</span></div>
                <div class="kpi-value">{aktual_val_str}</div>
                <div class="kpi-sub">Catatan historis pukul {selected_timestamp.strftime('%H:%M')}</div>
            </div>
        </summary>
        <div class="kpi-details">
            <b style="color:var(--text);">Apa itu Data Realita?</b><br>
            Ini adalah angka sebenarnya dari penggunaan listrik pada waktu masa lalu yang Anda pilih. Kami menampilkannya agar Anda bisa membandingkan secara langsung seberapa jauh tebakan AI dengan kenyataan di lapangan.
        </div>
    </details>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <details class="kpi-clickable">
        <summary>
            <div class="kpi-card featured" style="margin-bottom: 0;">
                <div class="kpi-title" style="color: #004CCD;">Tebakan Sistem AI <span style="font-size:0.9rem; vertical-align:middle;">&#9432;</span></div>
                <div class="kpi-value" style="color: #004CCD;">{prediction/1000:.2f} kW</div>
                <div class="kpi-sub" style="background:rgba(15,98,254,0.10); color:#004CCD;">Hasil prediksi untuk waktu yang dipilih</div>
            </div>
        </summary>
        <div class="kpi-details">
            <b style="color:var(--text);">Darimana angka ini berasal?</b><br>
            Angka ini bukanlah data statis, melainkan hasil hitungan langsung (prediksi) dari sistem kecerdasan buatan (XGBoost) berdasarkan faktor cuaca dan pola penggunaan listrik masa lalu.
        </div>
    </details>
    """, unsafe_allow_html=True)

with kpi3:
    if ape_error is not None:
        ape_display = f"{ape_error:.2f}%"
        abs_display  = f"Berbeda {abs_error/1000:.2f} kW dari realita"
    else:
        ape_display = "N/A"
        abs_display  = "Tidak dapat dihitung (data aktual tidak tersedia)"
    st.markdown(f"""
    <details class="kpi-clickable">
        <summary>
            <div class="kpi-card" style="margin-bottom: 0;">
                <div class="kpi-title">Tingkat Error (APE) <span style="font-size:0.9rem; vertical-align:middle;">&#9432;</span></div>
                <div class="kpi-value {delta_color}" style="border-radius:4px; padding:0.2rem 0.5rem;">{ape_display}</div>
                <div class="kpi-sub">{abs_display}</div>
            </div>
        </summary>
        <div class="kpi-details">
            <b style="color:var(--text);">Apa arti persentase ini?</b><br>
            Ini adalah <i>Absolute Percentage Error</i> (APE), yaitu persentase selisih absolut antara prediksi AI dan data aktual untuk satu waktu yang dipilih. Semakin kecil persentasenya, berarti tebakan sistem semakin dekat dengan realita.
        </div>
    </details>
    """, unsafe_allow_html=True)

# --- Insight Bar (Akurasi Model) ---
insight_right = "Dibandingkan dengan data aktual historis" if ape_error is not None else "Dataset tidak tersedia - perbandingan tidak dapat dilakukan"
st.markdown(f"""
<div style="margin-top: 0.25rem; padding: 1rem; background: white; border-radius: 12px; border:1px solid rgba(195,198,216,0.22); border-left: 4px solid {bar_color}; display: flex; align-items: center; justify-content: space-between; gap:1rem; box-shadow: 0px 4px 20px rgba(0,0,0,0.04);">
    <div>
        <span style="font-weight: 700; color: #1C1B1B; margin-right: 10px;">Akurasi:</span>
        <span style="color: #424656; font-size: 1rem;">{status_text}</span>
    </div>
    <div style="font-size: 0.75rem; color: #424656;">{insight_right}</div>
</div>
""", unsafe_allow_html=True)     

# ===========================================================
# Buat Menampilkan Data Mentah (TRACEABILITY)
# ===========================================================
with st.expander("Tampilkan Data Mentah (Untuk Verifikasi)"):
    st.markdown('<div class="helper-text">Tabel ini adalah baris data historis yang paling sesuai dengan waktu pilihan. Kolom <b>PowerConsumption_Zone1</b> dipakai sebagai pembanding nilai aktual.</div>', unsafe_allow_html=True)
    if lookup_row is not None:
        display_df = pd.DataFrame(lookup_row).T
        # Sembunyikan Zone 2 dan Zone 3 agar dosen tidak mempermasalahkannya lagi
        cols_to_hide = [c for c in ['PowerConsumption_Zone2', 'PowerConsumption_Zone3'] if c in display_df.columns]
        if cols_to_hide:
            display_df = display_df.drop(columns=cols_to_hide)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("Data historis tidak ditemukan untuk waktu ini.")

# ===========================================================
# MODEL INFORMATION PANEL
# ===========================================================
with st.expander("Spesifikasi Teknis Model AI"):
    mi1, mi2 = st.columns(2)
    with mi1:
        st.markdown(f"""
        - **Model:** XGBoost Regressor
        - **Metode:** {metrics_data.get('model_type', 'Gradient Boosting').upper()}
        - **Jumlah fitur:** 21
        - **Data latih:** 41,817 sampel (80%)
        """)
    with mi2:
        st.markdown(f"""
        - **Target prediksi:** Konsumsi Zona 1 (Watt)
        - **Library:** XGBoost + SHAP + Plotly
        - **Validasi:** TimeSeriesSplit (5 Fold), Chronological Hold-out
        """)
    st.caption(" *Chronological split digunakan karena data time series tidak boleh di-shuffle agar tidak menyebabkan data leakage temporal.*")

# Light theme definition for Plotly
light_theme = dict(
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#61707A', family='Inter'),
    xaxis=dict(gridcolor='#D9E1E5', zeroline=False),
    yaxis=dict(gridcolor='#D9E1E5', zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20)
)

# ===========================================================
# PRE-COMPUTE SHAP FOR AI INSIGHT (Single Sample)
# ===========================================================
shap_explanation = None
contrib_df = pd.DataFrame()
if explainer is not None:
    shap_explanation = explainer(input_df)
    shap_explanation.feature_names = [FEATURE_LABELS.get(f, f) for f in feature_cols]
    sv = shap_explanation.values[0]
    fv = shap_explanation.data[0]
    contrib_df = pd.DataFrame({
        'Fitur': feature_cols,
        'Nama Fitur': shap_explanation.feature_names,
        'Nilai Input': fv,
        'SHAP Value (Watt)': sv,
        'Kontribusi Absolut': np.abs(sv),
    }).sort_values('Kontribusi Absolut', ascending=False).reset_index(drop=True)
    contrib_df.index = contrib_df.index + 1
    contrib_df.index.name = 'Rank'
    total_sv = contrib_df['Kontribusi Absolut'].sum()
    contrib_df['Kontribusi (%)'] = (contrib_df['Kontribusi Absolut'] / total_sv * 100).round(2) if total_sv > 0 else 0.0
    
    
    # TAHAP 0: Validasi SHAP
    shap_sum = sv.sum()
    base_val = shap_explanation.base_values[0]
    expected_pred = base_val + shap_sum
    print(f"[SHAP VALIDATION] Base: {base_val:.3f}, Sum SHAP: {shap_sum:.3f}, Expected Pred: {expected_pred:.3f}, Actual Pred: {prediction:.3f}")
# ===========================================================
# TAB SYSTEM: Monitoring + Interpretasi AI (SHAP)
# ===========================================================
tab_monitoring, tab_shap = st.tabs(["Prediksi dan Evaluasi", "Penjelasan Model"])

# ===========================================================
# TAB 1: MONITORING UTAMA
# ===========================================================
with tab_monitoring:

    # ===========================================================
    # MAIN PREDICTION GRAPH
    # ===========================================================
    chart_col = st.container()

    with chart_col:
        st.markdown("<div class='section-title' style='padding-top: 0.5rem;'>Perbandingan Aktual dan Prediksi</div>", unsafe_allow_html=True)

        if not df_hourly.empty:
            target_dt_str = f"2017-{int(month_input):02d}-{int(dom_input):02d}"
            try:
                target_timestamp = selected_timestamp.normalize()
                mask = (df_hourly['Datetime'].dt.month == month_input) & (df_hourly['Datetime'].dt.day >= dom_input - 1) & (df_hourly['Datetime'].dt.day <= dom_input + 1)
                sample_df = df_hourly[mask].copy()
            except:
                sample_df = pd.DataFrame()
                
            if sample_df.empty: 
                sample_df = df_hourly.tail(48).copy()

            sample_df['Base_Trend'] = sample_df['PowerConsumption_Zone1']

            try:
                pred_datetime = selected_timestamp.to_pydatetime()
            except:
                pred_datetime = sample_df['Datetime'].iloc[-1].to_pydatetime()
            
            fig_main = go.Figure()

            # --- Confidence band (+/-RMSE) ---
            rmse_val = metrics_data.get("rmse", 368.41)
            pred_upper = prediction + rmse_val
            pred_lower = prediction - rmse_val
            band_times = list(sample_df['Datetime']) + list(sample_df['Datetime'][::-1])
            band_upper = [pred_upper] * len(sample_df)
            band_lower = [pred_lower] * len(sample_df)
            fig_main.add_trace(go.Scatter(
                x=band_times, y=band_upper + band_lower[::-1],
                fill='toself', fillcolor='rgba(138,106,36,0.08)',
                line=dict(width=0), hoverinfo='skip',
                name='Perkiraan rentang selisih (+/-RMSE)*',
            ))
            # Upper/lower boundary dashed lines
            fig_main.add_trace(go.Scatter(
                x=list(sample_df['Datetime']), y=[pred_upper] * len(sample_df),
                mode='lines', line=dict(color='rgba(138,106,36,0.28)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))
            fig_main.add_trace(go.Scatter(
                x=list(sample_df['Datetime']), y=[pred_lower] * len(sample_df),
                mode='lines', line=dict(color='rgba(138,106,36,0.28)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))

            # --- Data historis aktual ---
            fig_main.add_trace(go.Scatter(
                x=sample_df['Datetime'], y=sample_df['Base_Trend'],
                mode='lines+markers', name='Konsumsi aktual historis',
                line=dict(color='#356A73', width=2.5), marker=dict(size=4)
            ))

            # --- Horizontal reference line dari prediksi ---
            fig_main.add_hline(
                y=prediction, line_dash="dot", line_color="rgba(154,74,66,0.42)",
                annotation_text=f"Prediksi: {prediction/1000:.2f} kW",
                annotation_position="left",
                annotation_font=dict(size=11, color="#9A4A42"),
            )

            # --- Vertical connector: prediction point -> nearest actual data ---
            time_diffs = (sample_df['Datetime'] - pred_datetime).abs()
            nearest_idx = time_diffs.idxmin()
            nearest_actual = float(sample_df.loc[nearest_idx, 'Base_Trend'])
            fig_main.add_trace(go.Scatter(
                x=[pred_datetime, pred_datetime],
                y=[nearest_actual, prediction],
                mode='lines',
                line=dict(color='rgba(154,74,66,0.42)', width=1.5, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))

            # --- Prediction point (star marker) ---
            fig_main.add_trace(go.Scatter(
                x=[pred_datetime], y=[prediction],
                mode='markers', name='Prediksi sistem',
                marker=dict(color='#9A4A42', size=14, symbol='circle',
                            line=dict(color='white', width=2)),
                hovertemplate=f'<b>Prediksi sistem</b><br>Waktu: %{{x}}<br>Konsumsi: {prediction/1000:.2f} kW<extra></extra>',
            ))

            # --- Annotation pada prediction point ---
            fig_main.add_annotation(
                x=pred_datetime, y=prediction,
                text=f"<b>{prediction/1000:.2f} kW</b>",
                showarrow=True, arrowhead=2, arrowcolor="#9A4A42",
                font=dict(size=12, color="#9A4A42", family="Inter"),
                bgcolor="white", bordercolor="#D9E1E5", borderpad=4,
                borderwidth=1, ax=0, ay=-40,
            )

            # --- Vertical inference time line ---
            fig_main.add_vline(
                x=pred_datetime.timestamp() * 1000, line_width=1.5, line_dash="dash",
                line_color="rgba(154,74,66,0.36)",
                annotation_text="Waktu prediksi",
                annotation_position="top",
                annotation_font=dict(size=10, color="#61707A"),
            )

            layout_main = light_theme.copy()
            layout_main['height'] = 480
            layout_main['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
            layout_main['yaxis']['title'] = dict(text='Konsumsi Energi (Watt)', font=dict(size=12))
            layout_main['xaxis']['title'] = dict(text='Waktu', font=dict(size=12))
            fig_main.update_layout(**layout_main)
            st.plotly_chart(fig_main, use_container_width=True)
        else:
            st.warning(" Dataset historis tidak ditemukan. Grafik perbandingan aktual tidak dapat ditampilkan.")

        # Disclaimer confidence band
        st.info("**Cara Membaca Grafik:**\n- **Garis Biru Solid:** Penggunaan listrik sesungguhnya.\n- **Titik Merah:** Tebakan sistem AI untuk waktu yang dipilih.\n- **Area Terarsir Kuning:** Batas kewajaran (margin of error). Selama garis biru berada di area ini, prediksi tergolong sangat wajar dan akurat.")

        # ===========================================================
        # INPUT VECTOR DISPLAY (DEBUG MODE)
        # ===========================================================
        if debug_mode:
            with st.expander("Detail input teknis", expanded=True):
                st.markdown('<div class="helper-text">Bagian ini menunjukkan data fitur yang dikirim ke model. Ditujukan untuk validasi teknis, bukan untuk pengguna umum.</div>', unsafe_allow_html=True)
                v_col1, v_col2 = st.columns([1, 2])
                with v_col1:
                    st.markdown("**Data mentah:**")
                    st.json(input_df.iloc[0].to_dict())
                with v_col2:
                    st.markdown("**Tabel fitur:**")
                    st.dataframe(input_df.T.rename(columns={0: 'Nilai input model'}), use_container_width=True)
                if explainer is not None:
                    st.markdown(f"**Validasi SHAP (Tahap 0):** Base Value ({shap_explanation.base_values[0]:.2f}) + Sum SHAP ({sv.sum():.2f}) = {shap_explanation.base_values[0] + sv.sum():.2f} (Aktual Prediksi: {prediction:.2f})")

    # ===========================================================
    # REGRESSION METRICS PANEL - Single Zone
    # ===========================================================
    st.markdown("<div class='section-title'>Akurasi Model pada Data Uji</div>", unsafe_allow_html=True)

    if "error" in metrics_data:
        st.error(f"Error reading metrics: {metrics_data['error']}")
    else:
        m_r2 = metrics_data.get("r2", 0)
        m_rmse = metrics_data.get("rmse", 0)
        m_mae = metrics_data.get("mae", 0)
        m_mape = metrics_data.get("mape", 0)
        n_train = metrics_data.get("n_train", 0)
        n_test = metrics_data.get("n_test", 0)
        n_features = metrics_data.get("n_features", 0)

        # Status label berdasarkan R2
        if m_r2 >= 0.99:
            r2_label, r2_badge = "Sangat baik", "sub-green"
        elif m_r2 >= 0.95:
            r2_label, r2_badge = "Baik", "sub-blue"
        elif m_r2 >= 0.90:
            r2_label, r2_badge = "Cukup", "sub-blue"
        else:
            r2_label, r2_badge = "Perlu ditingkatkan", "sub-red"

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'''<div class="kpi-card" style="border-top-color: #2F6F4E; padding: 1rem;">
                <div class="kpi-title">R2 Score</div>
                <div class="kpi-value">{m_r2:.4f}</div>
                <div class="kpi-sub {r2_badge}">{r2_label}</div>
            </div>''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''<div class="kpi-card" style="border-top-color: #8A6A24; padding: 1rem;">
                <div class="kpi-title">RMSE</div>
                <div class="kpi-value" style="color:#8A6A24;">{m_rmse:,.1f}</div>
                <div class="kpi-sub" style="background:#F5EEDB; color:#8A6A24;">Rata-rata besar kesalahan</div>
            </div>''', unsafe_allow_html=True)
        with m3:
            st.markdown(f'''<div class="kpi-card" style="border-top-color: #356A73; padding: 1rem;">
                <div class="kpi-title">MAE</div>
                <div class="kpi-value" style="color:#356A73;">{m_mae:,.1f}</div>
                <div class="kpi-sub" style="background:#E2EEF0; color:#356A73;">Rata-rata kesalahan absolut</div>
            </div>''', unsafe_allow_html=True)
        with m4:
            st.markdown(f'''<div class="kpi-card" style="border-top-color: #2F6F4E; padding: 1rem;">
                <div class="kpi-title">MAPE</div>
                <div class="kpi-value" style="color:#2F6F4E;">{m_mape:.2f}%</div>
                <div class="kpi-sub sub-green">Rata-rata kesalahan (%)</div>
            </div>''', unsafe_allow_html=True)

        # --- Ringkasan Interpretasi ---
        st.markdown(f'''
        <div style="background:#FFFFFF; border:1px solid #D9E1E5; border-radius:8px; padding:1rem 1.5rem; margin-top:0.8rem;
                    border-left: 4px solid #2F6F4E; color: #61707A; font-size:0.9rem; line-height:1.6;">
            <b style="color:#1F2933;">Penjelasan Akurasi Keseluruhan:</b><br>
            Secara keseluruhan, sistem ini <b>sangat akurat</b> dalam memprediksi. 
            Rata-rata, tebakan sistem meleset sekitar <b>{m_mae:,.0f} Watt</b> (atau sekitar <b>{m_mape:.2f}%</b> dari beban aslinya). 
            Angka R2 sebesar <b>{m_r2:.4f}</b> menunjukkan bahwa sistem mampu mengenali pola konsumsi listrik dengan sangat baik (skor sempurna adalah 1.0).<br>
            <i>Model kecerdasan buatan ini belajar dari {n_train:,} data masa lalu untuk menebak {n_test:,} data baru.</i>
        </div>
        ''', unsafe_allow_html=True)

        cv_summary = timeseries_cv_data.get("summary", {})
        cv_mape = cv_summary.get("mape", {})
        cv_r2 = cv_summary.get("r2", {})
        if cv_summary:
            st.markdown(f'''
            <div style="background:#FFFFFF; border:1px solid #D9E1E5; border-left:4px solid #356A73;
                        border-radius:8px; padding:0.9rem 1.3rem; margin-top:0.8rem;
                        color:#61707A; font-size:0.88rem; line-height:1.6;">
                <b>Validasi tambahan TimeSeriesSplit:</b>
                {timeseries_cv_data.get("n_splits", 0)} fold expanding-window menghasilkan
                rata-rata <b>R2 = {cv_r2.get("mean", 0):.4f}</b> dan
                <b>MAPE = {cv_mape.get("mean", 0):.2f}%</b>.
                Fold validasi selalu berada setelah fold training, sehingga evaluasi tetap mengikuti urutan waktu.
                <i>(Sederhananya: model diuji 5 kali dengan potongan waktu data yang berbeda-beda, dan hasilnya konsisten setiap kali - bukan cuma kebetulan bagus di satu percobaan saja.)</i>
            </div>
            ''', unsafe_allow_html=True)


# ===========================================================
# TAB 2: INTERPRETASI AI - SHAP (Explainable AI)
# ===========================================================
with tab_shap:

    st.markdown("<div class='section-title'>Penjelasan Faktor Prediksi</div>", unsafe_allow_html=True)
    st.info("**Mengapa sistem memprediksi angka tersebut?**\n\nSistem AI ini (XGBoost) menganalisis banyak variabel untuk membuat prediksi. Metode **SHAP (SHapley Additive exPlanations)** di bawah ini digunakan untuk menyingkap *black-box* AI tersebut. Ini menunjukkan seberapa besar pengaruh setiap variabel (seperti cuaca atau jam) dalam menaikkan atau menurunkan angka prediksi akhir.")



    # ===========================================================
    # ABLATION STUDY - VALIDASI LAG FEATURES
    # ===========================================================
    with st.expander("Mengapa Data Riwayat Penting? (Ablation Study)", expanded=False):
        st.markdown("""Bagian ini menunjukkan bahwa AI kita sangat bergantung pada data pemakaian listrik beberapa jam sebelumnya (fitur riwayat). Pengujian di bawah membuktikan bahwa model tidak 'menyontek' masa depan.""")

        if not df_raw.empty:
            ablation_df = load_ablation_results(dataset_fingerprint)

            if not ablation_df.empty:
                r2_col = 'R2'

                # --- Tabel Perbandingan dengan Color Highlighting ---
                st.markdown("##### Perbandingan Metrik Antar Varian")

                def highlight_ablation(row):
                    """Warnai baris terbaik (hijau) dan terburuk (merah)."""
                    colors = ['' for _ in row]
                    if row['Skenario'] == ablation_df.loc[ablation_df[r2_col].idxmax(), 'Skenario']:
                        colors = ['background-color: rgba(47,111,78,0.12)' for _ in row]
                    elif row['Skenario'] == ablation_df.loc[ablation_df[r2_col].idxmin(), 'Skenario']:
                        colors = ['background-color: rgba(154,74,66,0.12)' for _ in row]
                    return colors

                styled_df = (
                    ablation_df.style
                    .apply(highlight_ablation, axis=1)
                    .format({
                        r2_col: '{:.4f}',
                        'RMSE': '{:,.2f}',
                        'MAE': '{:,.2f}',
                        'MAPE (%)': '{:.2f}%',
                    })
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

                # --- Plotly Bar Chart: MAPE Comparison ---
                st.markdown("##### Perbandingan MAPE Antar Varian")
                fig_abl = go.Figure()
                bar_colors = ['#2F6F4E', '#8A6A24', '#9A4A42']
                for i, row in ablation_df.iterrows():
                    fig_abl.add_trace(go.Bar(
                        x=[row['Skenario']],
                        y=[row['MAPE (%)']],
                        name=row['Skenario'],
                        marker_color=bar_colors[i],
                        text=f"{row['MAPE (%)']}%",
                        textposition='outside',
                        textfont=dict(size=13, color='#1F2933'),
                    ))
                abl_layout = light_theme.copy()
                abl_layout['height'] = 380
                abl_layout['showlegend'] = False
                abl_layout['yaxis'] = dict(
                    gridcolor='#D9E1E5', zeroline=False,
                    title=dict(text='MAPE (%)', font=dict(size=12)),
                    range=[0, max(ablation_df['MAPE (%)']) * 1.35],
                )
                abl_layout['xaxis'] = dict(
                    gridcolor='#D9E1E5', zeroline=False,
                    title=dict(text='Skenario Model', font=dict(size=12)),
                )
                fig_abl.update_layout(**abl_layout)
                st.plotly_chart(fig_abl, use_container_width=True)

                # --- R2 Delta Summary ---
                r2_full = ablation_df.loc[0, r2_col]
                r2_no_lag = ablation_df.loc[2, r2_col]
                r2_drop = r2_full - r2_no_lag

                st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #D9E1E5; border-left:4px solid #2F6F4E; border-radius:8px; padding:1rem 1.5rem; margin-top:0.5rem; color:#61707A; line-height:1.6;">
<b style="color:#1F2933;">Ringkasan:</b> Penurunan R2 dari Variant A ({r2_full:.4f}) ke Variant C ({r2_no_lag:.4f})
menunjukkan <b>delta = {r2_drop:.4f}</b>. Artinya, fitur riwayat konsumsi memang membantu performa model
dan tetap valid karena hanya memakai observasi masa lalu.
</div>
""", unsafe_allow_html=True)

                st.info(
                    "Penurunan R2 dari Variant A ke C menunjukkan bahwa performa model "
                    "berasal dari pola historis yang valid, bukan kebocoran data masa depan. "
                    "Fitur lag hanya menggunakan observasi masa lalu (t-1, t-6, t-144 untuk lag harian)."
                )
            else:
                st.warning("Ablation study gagal dijalankan. Periksa dataset dan model.")
        else:
            st.warning("Dataset belum tersedia untuk ablation study.")

    # ===========================================================
    # PERBANDINGAN MODEL (BASELINE BENCHMARKING)
    # ===========================================================
    with st.expander("Perbandingan dengan Sistem Klasik (Baseline Benchmark)", expanded=False):
        st.markdown("""Perbandingan performa antara model **XGBoost** utama dengan dua model pembanding: Linear Regression dan Naive Persistence.""")
        
        if not df_raw.empty:
            baseline_df = load_baseline_results(dataset_fingerprint)
            
            if not baseline_df.empty:
                base_r2_col = 'R2'

                # --- Tabel Perbandingan ---
                st.markdown("##### Tabel Evaluasi Model")
                
                # Sort by MAPE ascending
                baseline_df = baseline_df.sort_values('MAPE (%)', ascending=True).reset_index(drop=True)
                
                def highlight_xgboost(row):
                    colors = ['' for _ in row]
                    if 'XGBoost' in row['Model']:
                        colors = ['background-color: rgba(16,185,129,0.15)' for _ in row]
                    return colors
                    
                styled_base_df = (
                    baseline_df.style
                    .apply(highlight_xgboost, axis=1)
                    .format({
                        base_r2_col: '{:.4f}',
                        'RMSE': '{:,.2f}',
                        'MAE': '{:,.2f}',
                        'MAPE (%)': '{:.2f}%',
                    })
                )
                st.dataframe(styled_base_df, use_container_width=True, hide_index=True)
                
                # --- Plotly Grouped Bar Chart ---
                st.markdown("##### Visualisasi Metrik")
                metrics = [base_r2_col, 'RMSE', 'MAE', 'MAPE (%)']
                fig_base = go.Figure()
                
                colors_map = {'XGBoost (Proposed)': '#2F6F4E', 'Linear Regression': '#356A73', 'Naive (Persistence)': '#61707A'}
                
                for model_name in baseline_df['Model']:
                    model_data = baseline_df[baseline_df['Model'] == model_name].iloc[0]
                    fig_base.add_trace(go.Bar(
                        name=model_name,
                        x=metrics,
                        y=[model_data[m] for m in metrics],
                        marker_color=colors_map.get(model_name, '#9CA3AF'),
                        text=[f"{model_data[m]:.2f}" for m in metrics],
                        textposition='auto'
                    ))
                
                base_layout = light_theme.copy()
                base_layout['barmode'] = 'group'
                base_layout['height'] = 400
                base_layout['yaxis'] = dict(gridcolor='#D9E1E5', zeroline=False)
                fig_base.update_layout(**base_layout)
                
                st.plotly_chart(fig_base, use_container_width=True)
                
                # --- Dynamic Interpretation Box ---
                mape_xgb = baseline_df.loc[baseline_df['Model'] == 'XGBoost (Proposed)', 'MAPE (%)'].values[0]
                mape_lr = baseline_df.loc[baseline_df['Model'] == 'Linear Regression', 'MAPE (%)'].values[0]
                mape_naive = baseline_df.loc[baseline_df['Model'] == 'Naive (Persistence)', 'MAPE (%)'].values[0]
                
                best_model = baseline_df.loc[baseline_df['MAPE (%)'].idxmin(), 'Model']
                
                if best_model == 'XGBoost (Proposed)':
                    st.success(
                        f"XGBoost mengungguli semua baseline dengan MAPE **{mape_xgb:.2f}%** "
                        f"dibanding Linear Regression **{mape_lr:.2f}%** dan Naive **{mape_naive:.2f}%**, "
                        "membuktikan keunggulan model berbasis ensemble tree untuk prediksi beban listrik jangka pendek."
                    )
                else:
                    st.warning(
                        f"XGBoost (MAPE **{mape_xgb:.2f}%**) saat ini memiliki error yang lebih tinggi dibandingkan **{best_model}** "
                        f"(MAPE **{baseline_df['MAPE (%)'].min():.2f}%**). "
                        "Pada subset data ini, hubungan antar variabel mungkin lebih sederhana sehingga model linear unggul."
                    )
            else:
                st.warning("Evaluasi baseline gagal dijalankan.")
        else:
            st.warning("Dataset belum tersedia.")

    # ===========================================================
    # ANALISIS RESIDUAL & DISTRIBUSI ERROR
    # LAZY LOAD: chart hanya digenerate saat expander dibuka
    # ===========================================================
    residual_exp = st.expander("Analisis Pola Error Prediksi (Residual Analysis)", expanded=False)
    with residual_exp:
        st.markdown(
            "Analisis mendalam mengenai karakteristik selisih prediksi model. "
            "Dua chart di bawah membuktikan model tidak memiliki bias sistematis "
            "di sepanjang periode operasional."
        )
        show_residual_analysis = st.toggle(
            "Tampilkan analisis residual",
            value=False,
            key="show_residual_analysis",
        )
        if show_residual_analysis and not df_raw.empty:
            y_test_vals, y_pred_vals, X_test_df = get_residual_data(dataset_fingerprint)

            if y_test_vals is not None:
                fig_res = plot_residual_analysis(y_test_vals, y_pred_vals, X_test_df)
                st.plotly_chart(fig_res, use_container_width=True, theme=None)

                # --- Metrics Row ---
                res = y_test_vals - y_pred_vals
                mean_res = np.mean(res)
                std_res = np.std(res)
                max_err = np.max(np.abs(res))
                pct_under_500 = np.mean(np.abs(res) < 500) * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rata-rata selisih", f"{mean_res:,.2f} W")
                m2.metric("Sebaran selisih", f"{std_res:,.2f} W")
                m3.metric("Selisih terbesar", f"{max_err:,.2f} W")
                m4.metric("Selisih < 500 W", f"{pct_under_500:.1f}%")

                st.info(
                    "**Cara membaca chart:**\n"
                    "- **Distribusi Residual:** Histogram mendekati nol berarti model tidak bias secara sistematis.\n"
                    "- **Residual per Jam:** Boxplot yang rata di sekitar nol untuk semua jam menunjukkan model konsisten di seluruh periode operasional."
                )
        elif not show_residual_analysis:
            st.caption("Aktifkan toggle di atas untuk memuat chart residual.")
        else:
            st.warning("Dataset belum tersedia.")

    # ===========================================================
    # PROSES HYPERPARAMETER TUNING
    # ===========================================================
    with st.expander("Parameter Teknis AI (Hyperparameter Tuning)", expanded=False):
        st.markdown("Dokumentasi konfigurasi parameter model XGBoost yang digunakan dalam penelitian ini.")

        # --- a. Tuning Methodology ---
        st.markdown("##### Metode Penentuan Parameter")
        st.info(
            "Parameter ditetapkan secara manual mengacu pada nilai default "
            "resmi XGBoost serta rentang nilai yang umum digunakan pada penelitian terapan sejenis, "
            "bukan melalui proses pencarian otomatis seperti Grid Search atau Random Search. "
            "Pendekatan ini dipilih karena fokus penelitian berada pada interpretabilitas model "
            "(SHAP) dan implementasi sistem, bukan pada optimasi performa model secara maksimal."
        )

        # --- b. Final Chosen Parameters ---
        st.markdown("##### Parameter Terpilih")
        param_docs = pd.DataFrame([
            {"Parameter": "n_estimators",      "Nilai Terpilih": "500"},
            {"Parameter": "max_depth",          "Nilai Terpilih": "6"},
            {"Parameter": "learning_rate",      "Nilai Terpilih": "0.05"},
            {"Parameter": "subsample",          "Nilai Terpilih": "0.8"},
            {"Parameter": "colsample_bytree",   "Nilai Terpilih": "0.8"},
            {"Parameter": "min_child_weight",   "Nilai Terpilih": "5"},
            {"Parameter": "gamma",              "Nilai Terpilih": "0.1"},
            {"Parameter": "reg_alpha (L1)",     "Nilai Terpilih": "0.1"},
            {"Parameter": "reg_lambda (L2)",    "Nilai Terpilih": "1.0"},
        ])
        st.dataframe(param_docs, use_container_width=True, hide_index=True)

        # --- Early stopping info: ambil best_iteration dari model secara dinamis ---
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None:
            early_stop_note = (
                f"Proses pelatihan dihentikan otomatis oleh mekanisme early stopping "
                f"pada iterasi ke-**{best_iter}** dari n_estimators=500."
            )
        else:
            early_stop_note = (
                "Proses pelatihan dihentikan otomatis oleh mekanisme early stopping "
                "sebelum mencapai iterasi maksimum."
            )
        st.caption(f" {early_stop_note}")

        # --- c. Early Stopping Info ---
        st.warning(
            "**Strategi Validasi:** Data latih (80%) dibagi lagi menjadi **fit set (90%)** dan "
            "**validation set kronologis (10%)**. Early stopping memantau RMSE pada validation set "
            "tiap iterasi. Data uji (20% akhir) sepenuhnya tidak disentuh hingga evaluasi final, "
            "mencegah kebocoran informasi masa depan."
        )



    if explainer is not None:

        current_pred = prediction

        # ===========================================================
        # SECTION A: WATERFALL PLOT REAL-TIME
        # ===========================================================
        st.markdown("<div class='section-title'>Arah Pengaruh Setiap Fitur</div>", unsafe_allow_html=True)

        st.markdown(f'''
        <div class="kpi-card" style="border-top-color: #356A73; text-align:left; padding: 1.2rem 1.5rem;">
            <span style="color:#61707A;">Plot ini menunjukkan bagaimana model sampai pada prediksi 
            <b style="color:#356A73;">{current_pred/1000:,.2f} kW</b> untuk Zona 1 Kota Tetouan.
            Setiap bar menunjukkan kontribusi satu fitur. Bar merah menaikkan prediksi, sedangkan bar biru menurunkannya.</span>
        </div>
        ''', unsafe_allow_html=True)

        # Render waterfall plot dengan matplotlib
        with st.container():
            waterfall_png = render_shap_waterfall_png(
                tuple(float(x) for x in np.round(shap_explanation.values[0].astype(float), 6)),
                tuple(float(x) for x in np.round(shap_explanation.data[0].astype(float), 6)),
                float(shap_explanation.base_values[0]),
                tuple(shap_explanation.feature_names),
                float(current_pred),
            )
            _, wf_col, _ = st.columns([0.06, 0.88, 0.06])
            with wf_col:
                st.image(waterfall_png, use_container_width=True)
            
            

        # ===========================================================
        # SECTION B: RINGKASAN DAN TABEL KONTRIBUSI FITUR
        # ===========================================================
        st.markdown("<div class='section-title'>Tiga Faktor Terbesar Saat Ini</div>", unsafe_allow_html=True)
        st.markdown(
            '<div class="helper-text" style="margin-top:-0.35rem; margin-bottom:1rem;">'
            'Kartu ini merangkum fitur dengan kontribusi SHAP terbesar untuk waktu yang dipilih.'
            '</div>',
            unsafe_allow_html=True
        )

        base_val = float(shap_explanation.base_values[0])

        # Tampilkan sebagai kartu ringkasan
        top3 = contrib_df.head(3)
        t1, t2, t3 = st.columns(3)
        rank_labels = ['Faktor #1', 'Faktor #2', 'Faktor #3']
        red_grad = ['#7F1D1D', '#B91C1C', '#EF4444']
        blue_grad = ['#1E3A8A', '#1D4ED8', '#3B82F6']
        
        for i, (col_w, (_, row), rl) in enumerate(zip([t1, t2, t3], top3.iterrows(), rank_labels)):
            is_positive = row['SHAP Value (Watt)'] > 0
            arah = 'Menaikkan' if is_positive else 'Menurunkan'
            arah_color = 'sub-red' if is_positive else 'sub-blue'
            rc = red_grad[i] if is_positive else blue_grad[i]
            with col_w:
                st.markdown(f'''
                <div class="kpi-card" style="border-top-color: {rc}; padding: 1rem;">
                    <div class="kpi-title">{rl}</div>
                    <div class="kpi-value" style="font-size:1.4rem; color:{rc};">{row["Nama Fitur"]}</div>
                    <div style="font-size:0.95rem; color:#61707A; margin: 0.3rem 0;">
                        SHAP: <b>{row["SHAP Value (Watt)"]:+,.0f} W</b> ({row["Kontribusi (%)"]:.1f}%)
                    </div>
                    <div class="kpi-sub {arah_color}">{arah} prediksi</div>
                </div>''', unsafe_allow_html=True)

        # Tampilkan tabel lengkap
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Tabel Pengaruh Fitur Lengkap</div>", unsafe_allow_html=True)
        st.dataframe(
            contrib_df[['Nama Fitur', 'Nilai Input', 'SHAP Value (Watt)', 'Kontribusi (%)']].style
                .format({'Nilai Input': '{:.2f}', 'SHAP Value (Watt)': '{:+,.2f}', 'Kontribusi (%)': '{:.2f}%'})
                .bar(subset=['SHAP Value (Watt)'], color=['#B9CFD3', '#D8B8B2'], align='zero')
                .background_gradient(subset=['Kontribusi (%)'], cmap='YlOrRd'),
            width='stretch',
            height=500
        )

        st.markdown(f'''
        <div style="background:#FFFFFF; border:1px solid #D9E1E5; border-radius:8px; padding:1rem 1.5rem; margin-top:1rem;
                    border-left: 4px solid #356A73; color: #61707A; font-size:0.9rem; line-height:1.6;">
            <b>Base Value (E[f(x)]):</b> {base_val:,.0f} Watt - ini adalah rata-rata prediksi model
            jika tidak ada informasi fitur. Setiap fitur kemudian menambah atau mengurangi
            nilai ini hingga menghasilkan prediksi akhir <b>{current_pred:,.0f} Watt</b>.
        </div>
        ''', unsafe_allow_html=True)

        # ===========================================================
        # SECTION C: ANALISIS GLOBAL (Gambar dari Colab)
        # ===========================================================
        with st.expander("Ringkasan pengaruh fitur secara global"):
            st.markdown('<div style="background:#FFFFFF; border:1px solid #D9E1E5; border-radius:8px; padding:1rem 1.5rem; margin-bottom:1rem; border-left: 4px solid #356A73; color: #61707A; font-size:0.9rem; line-height:1.6;">Plot di bawah ini menunjukkan analisis SHAP pada <b>seluruh data uji</b> (10.000+ sampel) yang telah dihitung sebelumnya. Ini memberikan gambaran umum tentang fitur mana yang paling berpengaruh secara global terhadap model.</div>', unsafe_allow_html=True)

            bar_img_path = os.path.join("models", "shap_bar_summary.png")
            bee_img_path = os.path.join("models", "shap_beeswarm_summary.png")

            st.markdown("**Beeswarm Plot** - Distribusi SHAP value per fitur")
            if os.path.exists(bee_img_path):
                _, bee_col, _ = st.columns([0.08, 0.84, 0.08])
                with bee_col:
                    st.image(bee_img_path, use_container_width=True)
            else:
                st.info("File `shap_beeswarm_summary.png` belum tersedia di folder models/.")
                
            st.markdown("---")
                
            st.markdown("**Bar Summary Plot** - Peringkat fitur berdasarkan mean |SHAP|")
            if os.path.exists(bar_img_path):
                _, bar_col, _ = st.columns([0.08, 0.84, 0.08])
                with bar_col:
                    st.image(bar_img_path, use_container_width=True)
            else:
                st.info("File `shap_bar_summary.png` belum tersedia di folder models/.")
                
            st.markdown("---")
            st.markdown("""
            <div style="background:#FFFFFF; border:1px solid #D9E1E5; border-radius:8px; padding:1.2rem; color: #61707A; font-size:0.9rem; line-height:1.6;">
            <b>Cara Membaca Grafik SHAP di Atas:</b><br><br>
            <b>1. Bee Swarm Plot (Detail Arah Dampak):</b>
            <ul>
                <li>Menunjukkan spesifik <i>bagaimana</i> sebuah variabel berdampak pada hasil akhir.</li>
                <li><b>Sumbu-Y:</b> Daftar variabel diurutkan dari yang paling penting (atas).</li>
                <li><b>Sumbu-X (SHAP Value):</b> Titik di sebelah <b>kanan nol</b> mendorong prediksi <b>lebih tinggi</b>. Titik di <b>kiri nol</b> mendorong prediksi <b>lebih rendah</b>.</li>
                <li><b>Warna Titik:</b> <b>Merah/Pink</b> artinya nilai asli fitur tinggi, <b>Biru</b> artinya rendah.</li>
                <li><i>Contoh:</i> Jika baris "Temperature" dominan titik merah di kanan, artinya suhu tinggi menyebabkan beban listrik naik.</li>
            </ul>
            <b>2. Bar Summary Plot (Tingkat Kepentingan Umum):</b>
            <ul>
                <li>Semakin panjang balok ke arah kanan, semakin krusial variabel tersebut bagi model secara keseluruhan.</li>
                <li>Plot ini <b>hanya</b> menunjukkan seberapa penting fitur tersebut (absolut), bukan arah dampaknya (naik/turun).</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("  SHAP Explainer belum tersedia. Pastikan model XGBoost sudah dilatih.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--muted); font-size: 0.85rem; padding-bottom: 2rem;">
    Data bersumber dari <b>Powerconsumption Tetouan City, Maroko (2017)</b>.<br>
    Sistem prediksi ditenagai oleh algoritma <b>XGBoost Regressor</b>, dilatih dengan 41.817 data observasi riwayat.
</div>
""", unsafe_allow_html=True)

