# Flight Delay BI

**Predictive Business Intelligence for Flight Disruption Management Using Machine Learning and Deep Learning**

Proyek riset ini menggunakan data historis penerbangan untuk memprediksi keterlambatan dan pembatalan, menjelaskan faktor yang berkaitan dengan prediksi, serta menyajikan hasilnya melalui dashboard Business Intelligence (BI). Sistem yang direncanakan mendukung analisis kinerja maskapai, bandara, dan rute penerbangan.

README ini merangkum [rencana penelitian](flight_disruption_research_plan.md). Implementasi saat ini telah menyelesaikan pemahaman dataset serta preprocessing dan feature engineering untuk eksperimen klasifikasi keterlambatan (E1). Pelatihan model dan dashboard masih menjadi tahap berikutnya.

## Tujuan penelitian

- Memprediksi apakah penerbangan mengalami keterlambatan yang signifikan.
- Memperkirakan durasi keterlambatan dan probabilitas pembatalan.
- Membandingkan pendekatan Machine Learning (ML) dan Deep Learning (DL).
- Menjelaskan kontribusi fitur terhadap prediksi menggunakan feature importance dan SHAP.
- Mengembangkan indeks risiko gangguan dan dashboard untuk mendukung keputusan operasional.

## Dataset

Dataset penelitian berasal dari [Flight Delay and Cancellation Dataset (2019–2023) — Kaggle, akun patrickzel](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/code). Unduh dataset melalui tab **Data** pada halaman tersebut. Dataset tidak disertakan dalam repository karena dikecualikan melalui `.gitignore`.

Penelitian ini memakai sampel 3 juta penerbangan dari dataset tersebut. Berdasarkan [hasil audit sampel penelitian](dataset_research_feasibility_review.md), cakupannya adalah:

| Aspek | Hasil |
| --- | --- |
| Jumlah baris | 3.000.000 |
| Jumlah kolom | 32 |
| Periode aktual | 1 Januari 2019–31 Agustus 2023 |
| Maskapai | 18 |
| Bandara asal / tujuan | 380 / 380 |
| Rute berarah | 7.785 |
| Penerbangan dibatalkan | 79.140 (2,638% seluruh baris) |
| Populasi E1 | 2.920.860 penerbangan non-cancelled dengan label departure delay tersedia |
| Departure delay ≥15 menit pada populasi E1 | 531.033 (sekitar 18,18%) |

Kolom mencakup tanggal, maskapai, bandara asal/tujuan, jadwal dan waktu aktual, durasi, jarak, keterlambatan, pembatalan, pengalihan, serta kategori penyebab delay. Statistik di atas menggambarkan sampel yang dianalisis dalam penelitian; volume penerbangan tidak dapat langsung dianggap sebagai jumlah seluruh operasi bandara.

Setelah diunduh dari Kaggle, tempatkan dataset raw di root proyek agar dapat dibaca langsung oleh skrip. Dataset hasil preprocessing dan artefak transformer juga dikecualikan dari Git dan dibuat saat skrip dijalankan.

## Rencana eksperimen

E1–E4 adalah penamaan eksperimen dalam penelitian.

| Eksperimen | Tugas | Kandidat model dalam rencana |
| --- | --- | --- |
| **E1** | Klasifikasi keterlambatan keberangkatan ≥15 menit | Logistic Regression, Random Forest, XGBoost, MLP |
| **E2** | Regresi durasi keterlambatan | Linear Regression, Random Forest Regressor, XGBoost Regressor, MLP Regressor |
| **E3** | Klasifikasi pembatalan penerbangan | Logistic Regression, Random Forest, XGBoost, MLP |
| **E4** | Peramalan tingkat keterlambatan bandara pada periode berikutnya | XGBoost, LSTM, GRU |

E4 memerlukan agregasi deret waktu dan pemilihan bandara dengan observasi memadai. Banyak pasangan bandara–tanggal memiliki sedikit penerbangan dalam sampel, sehingga cakupan E4 perlu dibatasi berdasarkan data train.

Evaluasi klasifikasi direncanakan menggunakan precision, recall, F1, ROC-AUC, PR-AUC, dan confusion matrix; accuracy tidak menjadi satu-satunya ukuran. Evaluasi regresi menggunakan MAE, RMSE, dan R². Perbandingan ML–DL dilakukan pada pembagian data dan prosedur evaluasi yang setara.

## Alur penelitian

1. Memahami dataset, kamus fitur, dan kualitas data.
2. Menetapkan target, populasi eksperimen, waktu prediksi, dan fitur yang tersedia saat prediksi.
3. Melakukan cleaning, eksplorasi data, dan feature engineering.
4. Membagi data berdasarkan waktu; mempelajari parameter preprocessing hanya dari train.
5. Melatih baseline, membandingkan model, dan mengevaluasi generalisasi ke periode berikutnya.
6. Menjelaskan prediksi dan menguji indeks risiko gangguan.
7. Mengintegrasikan hasil ke dashboard BI dan menyusun interpretasi operasional.

Prediksi sebelum keberangkatan tidak menggunakan informasi setelah kejadian, seperti waktu aktual, penyebab delay, atau label target sebagai fitur. Fitur historis dan rolling yang ditambahkan kemudian harus hanya memakai informasi yang telah tersedia sebelum waktu prediksi.

## Implementasi E1 saat ini

Skrip `classification/e1-data-preparation.py` menggabungkan cleaning, feature engineering, split waktu, encoding, imputasi, dan scaling dalam satu proses.

- **Populasi:** `CANCELLED == 0` dan `DEP_DELAY` valid; penerbangan diverted tetap masuk jika memenuhi syarat.
- **Target:** `DELAYED_15 = 1` jika `DEP_DELAY >= 15`, selain itu 0.
- **Fitur dasar:** kode maskapai, bandara asal/tujuan, kalender, jadwal keberangkatan/kedatangan, durasi terjadwal, dan jarak.
- **Transformasi:** one-hot encoding kategori; sin/cos untuk kalender dan waktu; indikator missing; imputasi median dan standardisasi durasi/jarak.
- **Pencegahan leakage:** encoder, median imputer, dan scaler dipelajari hanya dari train, kemudian diterapkan ke validation/test.

| Subset | Periode | Baris E1 |
| --- | --- | ---: |
| Train | 2019–2021 | 1.795.773 |
| Validation | 2022 | 669.412 |
| Test | Januari–Agustus 2023 | 455.675 |

Hasil pemrosesan saat ini menghasilkan **790 fitur** dalam matriks sparse. CSV setelah cleaning/FE disimpan untuk inspeksi, sedangkan matriks hasil seluruh preprocessing siap digunakan untuk pelatihan. Belum ada hasil akurasi model yang dilaporkan.

Lihat [laporan preprocessing E1](classification/outputs/e1-data-preparation-report.md) untuk pemilihan kolom, parameter transformasi, validasi hasil, serta head dan tail setiap subset.

## Menjalankan proyek

Gunakan Python 3.10 atau lebih baru. Unduh dataset dari [Kaggle](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/code), lalu letakkan sampel `flights_sample_3m.csv` di root proyek. Dari root, pasang dependensi:

```powershell
python -m pip install -r classification/requirements.txt
```

Untuk membuat laporan pemahaman dataset, pastikan kamus fitur juga tersedia di root, lalu jalankan:

```powershell
python dataset-understanding.py
```

Untuk menjalankan seluruh preprocessing dan FE E1:

```powershell
python classification/e1-data-preparation.py
```

Tidak diperlukan argumen CLI tambahan. Jika terminal sudah berada di folder `classification`, cukup jalankan `python e1-data-preparation.py`. Panduan keluaran tersedia pada [README classification](classification/README.md).

## Struktur proyek

```text
flight-delay-BI/
├── README.md
├── flight_disruption_research_plan.md
├── dataset_research_feasibility_review.md
├── flights_sample_3m.csv                 # Unduh dari Kaggle; tidak disertakan dalam Git
├── flights_sample_3m_features_dictionary.csv
├── audit_dataset.py
├── dataset-understanding.py
├── outputs/                             # Hasil pemahaman dataset
└── classification/
    ├── e1-data-preparation.py
    ├── requirements.txt
    ├── README.md
    ├── datasets/
    │   ├── prepared/                    # CSV setelah cleaning dan FE
    │   └── model-ready/                 # Matriks X (.npz) dan label y (.npy)
    ├── artifacts/                       # Transformer hasil fit train
    ├── outputs/                         # Laporan, kontrak fitur, preview head/tail
    └── tests/
```

Folder hasil dibuat oleh skrip. Pembacaan raw dilakukan bertahap, tetapi pengurutan dan transformasi memuat satu subset ke RAM pada satu waktu.

## Arah pengembangan BI

Dashboard yang direncanakan mencakup ringkasan operasional, kinerja maskapai/bandara/rute, prediksi risiko penerbangan, dan simulasi skenario. Keluaran model akan disajikan bersama penjelasan fitur serta cakupan data yang digunakan.

Implementasi data saat ini menggunakan Python, pandas, NumPy, scikit-learn, SciPy, dan joblib. Rencana pengembangan mencakup XGBoost, PyTorch, SHAP, backend FastAPI atau Flask, penyimpanan DuckDB atau PostgreSQL, serta antarmuka React atau HTML/JavaScript dengan ECharts atau Plotly. Pilihan stack aplikasi belum ditetapkan; Power BI tidak menjadi keharusan.

Indeks risiko gabungan dalam research plan masih merupakan rancangan yang perlu diuji bobot dan interpretasinya. Penjelasan SHAP menunjukkan kontribusi terhadap prediksi, bukan bukti sebab-akibat. Prediksi penerbangan mendatang secara operasional memerlukan jadwal dan pembaruan data tambahan di luar dataset historis ini.
