# Portofolio Proyek — Data Science & Machine Learning

Dua proyek di bawah ini diambil dari repository [projek-portofolio-rakha](https://github.com/Rkhkumara/projek-portofolio-rakha), folder `projects/`. Keduanya dikerjakan dari nol: mulai dari data mentah, proses cleaning, sampai model dan (untuk proyek kedua) aplikasi yang bisa langsung dicoba.

---

## 1. Analisis Perilaku Pelanggan Olist (E-Commerce Brasil)

Proyek ini menggali data transaksi Olist, salah satu platform e-commerce terbesar di Brasil, dengan lebih dari 100 ribu transaksi nyata dari periode 2016–2018. Fokusnya ada di dua hal: seberapa besar masalah retensi pelanggan, dan siapa saja pelanggan yang paling bernilai buat bisnis.

**Yang ditemukan cukup mengejutkan.** Retensi pelanggan anjlok drastis di bulan pertama — rata-rata kurang dari 2% pelanggan kembali belanja setelah pembelian pertama mereka. Artinya platform ini sebenarnya bertahan bukan karena loyalitas pelanggan, tapi karena terus-menerus menarik pembeli baru. Model bisnis seperti ini mahal dan sulit dipertahankan jangka panjang.

Data juga menunjukkan jumlah pelanggan yang "hilang" (Lost Customers) hampir sama banyaknya dengan pelanggan baru — sekitar 36 ribu di masing-masing kelompok. Sementara itu, segmen Champions cuma sekitar 1.200 orang, tapi nilai belanja rata-rata mereka jauh di atas segmen lain. Kehilangan satu pelanggan Champions setara dengan kehilangan puluhan pelanggan baru kalau dilihat dari sisi pendapatan.

**Metode yang dipakai:**
- Cohort Retention Analysis untuk melihat pola retensi bulanan (M1–M12)
- RFM Segmentation (Recency, Frequency, Monetary) untuk mengelompokkan pelanggan
- SQL (SQLite) sebagai mesin analisis utama — dipilih karena lebih efisien untuk operasi grouping dan window function dibanding pandas biasa
- Pandas untuk loading dan cleaning data, Matplotlib/Seaborn untuk visualisasi

**Rekomendasi yang dihasilkan** juga langsung dipetakan ke aksi bisnis, misalnya kampanye email untuk pelanggan baru di hari ke-7 setelah barang sampai, program win-back untuk pelanggan yang mulai jarang belanja, sampai program VIP khusus untuk segmen Champions.

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle, lisensi CC BY-NC-SA 4.0)

---

## 2. Prediksi Konsumsi Listrik Kota Tetouan, Maroko

Proyek kedua ini beda arah — lebih ke machine learning terapan dengan pipeline lengkap sampai deployment. Tujuannya memprediksi konsumsi listrik di Kota Tetouan menggunakan data time-series per 10 menit sepanjang tahun 2017, dengan XGBoost sebagai model utama.

Konsumsi listrik itu naik-turun terus tergantung suhu, kelembapan, jam berapa, bahkan hari apa dalam seminggu. Kalau bisa diprediksi dengan akurat, distribusi listrik jadi bisa diatur lebih efisien. Data yang dipakai berasal dari UCI Machine Learning Repository, mencakup tiga zona distribusi, tapi proyek ini fokus ke Zone 1.

**Hasil modelnya cukup solid.** Di test set, model mencapai R² sebesar 0.9968 dengan RMSE 347.5 Watt dan MAPE hanya 0.78%. Validasi dilakukan pakai expanding-window TimeSeriesSplit 5-fold supaya tidak ada kebocoran data dari masa depan ke masa lalu — kesalahan umum yang sering terjadi di analisis time-series. Sebagai pembanding, model ini juga diuji melawan baseline naive persistence dan linear regression.

**Feature engineering-nya cukup detail**, ada 21 fitur yang dibagi jadi empat kelompok: fitur cuaca (suhu, kelembapan, kecepatan angin), fitur temporal (jam, hari, bulan), cyclic encoding (supaya model paham jam 23 dan jam 0 itu sebenarnya berdekatan), dan fitur lag/rolling untuk menangkap pola konsumsi dari beberapa jam hingga sehari sebelumnya.

Yang bikin proyek ini lebih lengkap dari sekadar notebook: ada aplikasi web interaktif pakai Streamlit, lengkap dengan dashboard performa model, fitur prediksi realtime, analisis SHAP untuk interpretasi model, drift monitoring untuk mendeteksi kalau data input mulai bergeser dari data training, dan analisis residual. Ada juga unit test untuk preprocessing, inference, dan validasi time-series-nya.

**Stack yang dipakai:** XGBoost, Pandas, NumPy, SHAP, Streamlit, Matplotlib/Seaborn/Plotly.

Dataset: [Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (UCI ML Repository)

---

## Ringkasan Teknis

| | Olist Customer Analytics | Tetouan Power Prediction |
|---|---|---|
| Jenis masalah | Analisis perilaku pelanggan | Prediksi time-series |
| Metode utama | Cohort analysis, RFM segmentation | XGBoost Regressor |
| Tools inti | SQL, Pandas, Matplotlib | XGBoost, SHAP, Streamlit |
| Output | Notebook analisis + rekomendasi bisnis | Model + aplikasi web interaktif |
| Skala data | 100k+ transaksi (2016–2018) | ±52.400 baris data 10-menitan (2017) |

---

*Kode lengkap, notebook, dan detail implementasi masing-masing proyek bisa dilihat langsung di [repository GitHub](https://github.com/Rkhkumara/projek-portofolio-rakha/tree/main/projects).*
