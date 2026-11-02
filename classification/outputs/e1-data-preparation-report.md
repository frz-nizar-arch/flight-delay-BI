# E1 — Preprocessing dan feature engineering

Dibuat (UTC): 2026-11-02T05:14:12.947587+00:00

Cleaning, pembentukan fitur, pembagian waktu, encoding, imputasi, dan scaling selesai. Encoder, median imputer, dan scaler dipelajari hanya dari train. Belum ada model klasifikasi yang dilatih.

Target DELAYED_15 = 1 jika DEP_DELAY >= 15 menit; selain itu 0. Populasi memakai CANCELLED == 0 dengan DEP_DELAY valid. Diverted tetap masuk bila memenuhi syarat. Probabilitas model nantinya bersyarat pada populasi non-cancelled.

Skenario prediksi memakai jadwal sebelum keberangkatan. CSV tidak menyimpan riwayat revisi jadwal, sehingga ketersediaan jadwal pada horizon tertentu masih merupakan asumsi.

## Rekonsiliasi baris

| Pemeriksaan | Baris |
| --- | --- |
| input_rows | 3,000,000 |
| excluded_invalid_cancelled | 0 |
| excluded_cancelled | 79,140 |
| excluded_missing_or_invalid_target | 0 |
| excluded_invalid_date | 0 |
| excluded_outside_split_period | 0 |
| retained_rows | 2,920,860 |

Pengecualian saling eksklusif. Jumlah retained_rows + seluruh excluded_* sama dengan input_rows. Dataset raw di root hanya dibaca.

## Pemilihan kolom

| Kolom sumber | Keputusan | Alasan |
| --- | --- | --- |
| FL_DATE | dipakai | Metadata split waktu; diturunkan menjadi MONTH dan DAY_OF_WEEK. |
| AIRLINE | tidak dipakai | Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara. |
| AIRLINE_DOT | tidak dipakai | Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara. |
| AIRLINE_CODE | dipakai | Identitas maskapai; gunakan satu representasi identitas untuk baseline. |
| DOT_CODE | tidak dipakai | Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara. |
| FL_NUMBER | ditunda | Nomor penerbangan dapat diuji setelah baseline; jangan diperlakukan sebagai besaran kontinu. |
| ORIGIN | dipakai | Bandara asal yang tercantum pada jadwal. |
| ORIGIN_CITY | tidak dipakai | Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara. |
| DEST | dipakai | Bandara tujuan yang tercantum pada jadwal. |
| DEST_CITY | tidak dipakai | Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara. |
| CRS_DEP_TIME | dipakai | Jadwal keberangkatan HHMM; diubah menjadi menit sejak tengah malam lokal. |
| DEP_TIME | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DEP_DELAY | pendukung | Pembentuk target DELAYED_15 dengan ambang >=15 menit; bukan fitur model. |
| TAXI_OUT | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| WHEELS_OFF | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| WHEELS_ON | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| TAXI_IN | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| CRS_ARR_TIME | dipakai | Jadwal kedatangan HHMM; diubah menjadi menit sejak tengah malam lokal. |
| ARR_TIME | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| ARR_DELAY | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| CANCELLED | pendukung | Filter populasi: hanya nilai 0 yang digunakan; bukan fitur model. |
| CANCELLATION_CODE | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DIVERTED | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| CRS_ELAPSED_TIME | dipakai | Durasi perjalanan terjadwal dalam menit. |
| ELAPSED_TIME | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| AIR_TIME | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DISTANCE | dipakai | Jarak rute dalam mil. |
| DELAY_DUE_CARRIER | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DELAY_DUE_WEATHER | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DELAY_DUE_NAS | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DELAY_DUE_SECURITY | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |
| DELAY_DUE_LATE_AIRCRAFT | tidak dipakai | Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan. |

## Cleaning

| Pemeriksaan | Jumlah |
| --- | --- |
| AIRLINE_CODE_normalized | 0 |
| ORIGIN_normalized | 0 |
| DEST_normalized | 0 |
| CRS_DEP_TIME_2400_normalized | 0 |
| CRS_DEP_TIME_invalid_to_missing | 0 |
| CRS_ARR_TIME_2400_normalized | 13 |
| CRS_ARR_TIME_invalid_to_missing | 0 |
| CRS_ELAPSED_TIME_invalid_to_missing | 0 |
| DISTANCE_invalid_to_missing | 0 |
| scheduled_duration_under_10_retained | 1 |
| negative_delay_retained | 1,787,101 |
| delay_over_24h_retained | 130 |

Kategori di-trim dan diubah menjadi huruf kapital. Jam tidak valid dan durasi/jarak tidak positif menjadi missing. Delay negatif dan ekstrem dipertahankan; durasi terjadwal positif di bawah 10 menit ditandai untuk tinjauan. Missing target tidak diimputasi. Skrip tidak menghapus duplikasi; audit sebelumnya tidak menemukan kandidat duplikat. Audit ulang diperlukan bila sumber berubah.

## Feature engineering

- MONTH: Januari=1 hingga Desember=12; DAY_OF_WEEK: Senin=0 hingga Minggu=6.
- SCHEDULED_*_MINUTES: menit sejak tengah malam lokal; 1530 menjadi 930 dan 2400 menjadi 0. Tidak dibuat perkiraan tanggal kedatangan atau konversi zona waktu.
- Kalender dan waktu diubah menjadi pasangan sin/cos: sudut = 2*pi*(nilai-offset)/periode. Periode bulan=12, hari=7, waktu=1440; offset bulan=1, lainnya=0. Representasi ini menjaga kedekatan akhir dan awal siklus.
- Waktu yang missing memakai pasangan (0,0) dengan indikator missing=1. Durasi/jarak juga mendapat indikator missing. Aturan ini tetap dan tidak memerlukan fit.
- Sin/cos sudah berada pada rentang [-1,1]; indikator missing dan hasil one-hot adalah 0/1, sehingga ketiganya tidak distandardisasi.

## Transformasi yang dipelajari dari train

Maskapai dan bandara: missing diisi token __MISSING__, lalu one-hot encoding. Kategori baru pada validation/test menghasilkan semua nol pada kelompok kategori terkait. Jumlah kategori baru dilaporkan di bawah.

Durasi terjadwal dan jarak: missing diisi median train, kemudian StandardScaler menggunakan mean dan skala hasil imputasi train. Validation/test tidak ikut menentukan parameter. Kolom numerik yang seluruhnya kosong pada train menghentikan proses.

| Fitur | Median train | Mean setelah imputasi | Skala StandardScaler |
| --- | --- | --- | --- |
| CRS_ELAPSED_TIME | 125.0 | 141.36564532376866 | 70.95881820105598 |
| DISTANCE | 648.0 | 802.4031222209043 | 582.6894191351927 |

## Pembagian waktu dan matriks model

| Subset | Periode aktual | Baris | Label 0 | Label 1 | Delay rate (%) | Fitur akhir |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2019-01-01 s.d. 2021-12-31 | 1,795,773 | 1,511,379 | 284,394 | 15.8369 | 790 |
| validation | 2022-01-01 s.d. 2022-12-31 | 669,412 | 526,729 | 142,683 | 21.3147 | 790 |
| test | 2023-01-01 s.d. 2023-08-31 | 455,675 | 351,719 | 103,956 | 22.8136 | 790 |

Batas tetap: train 2019–2021; validation 2022; test Januari–Agustus 2023. Tiap subset diurutkan berdasarkan tanggal, waktu keberangkatan lokal, lalu nomor record sumber. Urutan lokal lintas bandara bukan urutan timestamp UTC.

## Missing sebelum imputasi

| Fitur dasar | Train | Validation | Test |
| --- | --- | --- | --- |
| AIRLINE_CODE | 0 | 0 | 0 |
| ORIGIN | 0 | 0 | 0 |
| DEST | 0 | 0 | 0 |
| MONTH | 0 | 0 | 0 |
| DAY_OF_WEEK | 0 | 0 | 0 |
| SCHEDULED_DEP_MINUTES | 0 | 0 | 0 |
| SCHEDULED_ARR_MINUTES | 0 | 0 | 0 |
| CRS_ELAPSED_TIME | 0 | 0 | 0 |
| DISTANCE | 0 | 0 | 0 |

## Kategori baru di luar train

| Kolom | Kategori hasil fit train | Baris baru validation | Baris baru test |
| --- | --- | --- | --- |
| AIRLINE_CODE | 18 | 0 | 0 |
| ORIGIN | 379 | 8 | 0 |
| DEST | 379 | 15 | 0 |

Kategori baru berarti nilai (termasuk token missing jika belum pernah ada saat train) yang tidak terdapat pada encoder train. Jumlah baris dilaporkan per kolom dan dapat saling tumpang tindih.

## Validasi hasil

- Jumlah baris input, pengecualian, dan hasil telah direkonsiliasi.
- Label hanya 0/1; SOURCE_ROW_NUMBER unik di masing-masing subset.
- Semua matriks memiliki urutan fitur yang sama dan tidak mengandung NaN/infinity.
- Metadata, CANCELLED, dan DEP_DELAY tidak menjadi masukan model. Tidak ada resampling atau fitur historis pada versi ini.

## Berkas keluaran

- datasets/prepared/{train,validation,test}.csv: data setelah cleaning/FE, sebelum transformasi yang dipelajari; untuk audit dan model dengan kebutuhan preprocessing lain.
- datasets/model-ready/X_{train,validation,test}.npz: matriks fitur sparse CSR float32 setelah seluruh preprocessing.
- datasets/model-ready/y_{train,validation,test}.npy: label int8. Baris X dan y sejajar dengan CSV prepared subset yang sama.
- artifacts/preprocessor.joblib: transformer hasil fit train, menerima fitur engineered sesuai feature_contract.json.
- outputs/feature_names.csv: nama dan urutan kolom matriks X; feature_contract.json: definisi fitur dan target.
- outputs/train_transformed_preview.csv: sepuluh baris pertama hasil transformasi train untuk inspeksi.
- outputs/preprocessing_summary.json dan e1-data-preparation-report.md: ringkasan lengkap.

Matriks menggunakan format sparse agar one-hot encoding tidak menghasilkan CSV padat berukuran besar. Nilai nol tersirat di dalam sparse bukan missing. X dibaca dengan scipy.sparse.load_npz; y dengan numpy.load.

## Sumber dan pemakaian ulang

Raw: D:\Coding\#bigdata\flight-delay-BI\flights_sample_3m.csv
SHA-256: 66c338ff6c99db352ceee1af0d3e41329cf466c1c203abee52aec1dc56fe7dc4
Waktu proses: 129.0 detik; chunk: 100,000 baris.

Raw tidak disalin. Pembacaan awal bertahap, tetapi pengurutan dan preprocessing memuat satu subset ke RAM pada satu waktu. File sementara serta hasil lama dapat memerlukan ruang disk bersamaan saat menjalankan ulang.

Langkah berikutnya adalah training baseline dan evaluasi. Jangan fit ulang preprocessing pada validation/test. Jika nanti menggunakan cross-validation di dalam train, preprocessing harus di-fit ulang di setiap fold dari CSV prepared, bukan memakai X yang telah di-fit pada seluruh train.
