# ⚡ Tetouan Power Consumption Prediction

Prediksi konsumsi energi listrik kota Tetouan (Maroko) menggunakan **XGBoost Regressor** berbasis data time-series 10 menit. Proyek ini mencakup pipeline lengkap mulai dari preprocessing data, feature engineering, training model, hingga deployment aplikasi web interaktif dengan **Streamlit**.

---

## 📌 Latar Belakang

Konsumsi energi listrik bersifat sangat dinamis — dipengaruhi oleh suhu udara, kelembapan, waktu dalam sehari, hingga hari dalam seminggu. Kemampuan memprediksi konsumsi energi secara akurat menjadi kunci dalam manajemen distribusi listrik yang efisien.

Dataset yang digunakan adalah [Tetouan City Power Consumption](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) dari UCI Machine Learning Repository, berisi data konsumsi energi per 10 menit sepanjang tahun 2017 untuk 3 zona distribusi. Penelitian ini berfokus pada **Zone 1** sebagai target prediksi.

---

## 🎯 Hasil Model

| Metrik | Test Set | CV (5-fold, mean) |
|--------|----------|-------------------|
| **R²** | 0.9968 | 0.9958 ± 0.0020 |
| **RMSE** | 347.50 W | 422.13 ± 118.89 W |
| **MAE** | 227.13 W | 274.73 ± 69.91 W |
| **MAPE** | 0.78% | 0.84% ± 0.14% |

Validasi menggunakan **expanding-window TimeSeriesSplit** (5 fold) untuk memastikan tidak ada data leakage dari masa depan ke masa lalu.

### Perbandingan dengan Baseline

| Model | R² | RMSE | MAPE |
|-------|----|------|------|
| Naive Persistence | 0.9921 | 550.07 W | 1.27% |
| Linear Regression | 0.9984 | 245.13 W | 0.54% |
| **XGBoost (Proposed)** | **0.9968** | **347.50 W** | **0.78%** |

---

## 🗂️ Struktur Proyek

```
02-energy-prediction-project/
├── app.py                      # Aplikasi Streamlit (main entry point)
├── config.yaml                 # Konfigurasi path, fitur, dan hyperparameter
├── requirements.txt
│
├── dataset/
│   └── powerconsumption.csv    # Raw dataset dari UCI
│
├── data/
│   └── processed/              # Hasil preprocessing (auto-generated)
│
├── models/                     # Artefak model (auto-generated saat training)
│   ├── model_tetouan_xgb.json
│   ├── tetouan_metrics.json
│   ├── tetouan_feature_cols.json
│   ├── tetouan_drift_baseline.json
│   └── tetouan_timeseries_cv.json
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_modeling_tetouan.ipynb  # Notebook training utama
│   └── *.png                   # Visualisasi hasil (SHAP, feature importance, dll.)
│
└── src/
    ├── data/
    │   ├── make_dataset.py
    │   └── preprocessing.py    # Cleaning + feature engineering
    ├── models/
    │   ├── train.py            # Pipeline training XGBoost
    │   └── evaluate.py         # Evaluasi + plot + drift baseline
    └── utils/
        ├── config_loader.py
        ├── drift_monitoring.py     # Deteksi data drift realtime
        ├── time_series_validation.py
        ├── inference.py
        ├── realtime_features.py
        ├── ablation.py
        ├── baseline_models.py
        └── residual_analysis.py
```

---

## ⚙️ Feature Engineering

Model dilatih dengan **21 fitur** yang dibagi menjadi empat kelompok:

| Kelompok | Fitur |
|----------|-------|
| **Cuaca** | Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows |
| **Temporal** | Hour, DayOfWeek, Month, IsWeekend |
| **Cyclic Encoding** | Hour_sin/cos, DayOfWeek_sin/cos, Month_sin/cos |
| **Lag & Rolling** | Zone1_lag1, Zone1_lag6, Zone1_lag24, Zone1_roll3, Zone1_roll6, Zone1_roll24 |

Cyclic encoding digunakan agar model memahami bahwa jam 23 dan jam 0 sebenarnya berdekatan secara siklus. Lag features menangkap autokorelasi temporal jangka pendek dan jangka panjang (hingga 1 hari ke belakang).

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.10+
- pip

### 1. Clone & Setup Environment

```bash
git clone <repo-url>
cd 02-energy-prediction-project

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Jalankan Preprocessing

```bash
python src/data/preprocessing.py
```

Output akan tersimpan di `data/processed/train_ready.csv`.

### 3. Training Model

```bash
python src/models/train.py
```

Model disimpan di `models/model_tetouan_xgb.json`.

### 4. Evaluasi Model

```bash
python src/models/evaluate.py
```

Menghasilkan metrik, plot feature importance, actual vs predicted, dan baseline drift.

### 5. Jalankan Aplikasi Web

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

> **Catatan:** Jika model sudah tersedia di folder `models/`, bisa langsung lompat ke langkah 5 tanpa perlu training ulang.

---

## 📊 Aplikasi Streamlit

Aplikasi web interaktif ini menyajikan:

- **Dashboard Overview** — ringkasan performa model dan dataset
- **Prediksi Realtime** — input manual kondisi cuaca dan waktu untuk mendapatkan prediksi konsumsi
- **Analisis SHAP** — interpretasi model berbasis SHAP values (feature importance global & lokal)
- **Drift Monitoring** — deteksi pergeseran distribusi data input dibandingkan baseline training
- **Residual Analysis** — analisis error model secara visual
- **Time Series CV** — visualisasi hasil validasi silang expanding-window

---

## 🛠️ Tech Stack

| Komponen | Library |
|----------|---------|
| Model | XGBoost 2.x |
| Data Processing | Pandas, NumPy |
| Visualisasi | Matplotlib, Seaborn, Plotly |
| Interpretability | SHAP |
| Web App | Streamlit |
| Config | PyYAML |

---

## 📁 Dataset

**Sumber:** [UCI Machine Learning Repository — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city)

- **Periode:** 1 Januari 2017 – 30 Desember 2017
- **Frekuensi sampling:** 10 menit
- **Total baris:** ±52.416 observasi
- **Zona:** 3 zona distribusi listrik (penelitian ini menggunakan Zone 1)

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan portofolio. Dataset berasal dari UCI ML Repository dan tunduk pada ketentuan penggunaan masing-masing.
