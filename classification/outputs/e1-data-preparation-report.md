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

## Head dan tail dataset setelah preprocessing dan FE

Head adalah 5 baris pertama dan tail adalah 5 baris terakhir dari setiap subset, mengikuti urutan dataset hasil pengolahan. Nilai fitur diambil langsung dari matriks X setelah encoding, imputasi, scaling, dan FE; label diambil dari y. Jika subset berisi kurang dari 5 baris, semua baris ditampilkan dan head/tail dapat berulang.

DATASET_ROW adalah posisi baris dalam subset (mulai 1). SOURCE_ROW_NUMBER dan FL_DATE hanya metadata pelacakan; DELAYED_15 adalah target, bukan fitur. Kolom one-hot diringkas sebagai daftar nama kolom bernilai 1; kolom one-hot lainnya bernilai 0. Kategori baru dapat menghasilkan seluruh nol dalam kelompoknya.

Untuk keterbacaan, fitur dibagi menjadi tabel identitas/encoding, scaling/indikator missing, dan fitur siklik. Prefix category__, continuous__, cyclic__, dan missing__ disingkat pada tampilan. Nilai desimal dibulatkan hingga 6 tempat hanya dalam laporan. Seluruh kolom dan nilai tanpa pembulatan tampilan tersedia di outputs/{train,validation,test}_head_tail.csv.

### Train

#### Head — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 1 | 2052433 | 2019-01-01 | 0 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_ORD |
| 2 | 1869934 | 2019-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_CLT |
| 3 | 945942 | 2019-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_PHX; DEST_DCA |
| 4 | 2113750 | 2019-01-01 | 0 | AIRLINE_CODE_NK; ORIGIN_IAG; DEST_FLL |
| 5 | 181563 | 2019-01-01 | 0 | AIRLINE_CODE_B6; ORIGIN_BQN; DEST_MCO |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.474579 | 1.791000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 1.869174 | 2.269814 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 1.390023 | 2.019252 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 0.600832 | 0.641160 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 0.459906 | 0.560499 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.000000 | 1.000000 | 0.781832 | 0.623490 | 0.082808 | 0.996566 | 0.994056 | -0.108867 |
| 2 | 0.000000 | 1.000000 | 0.781832 | 0.623490 | 0.173648 | 0.984808 | 0.833886 | -0.551937 |
| 3 | 0.000000 | 1.000000 | 0.781832 | 0.623490 | 0.321439 | 0.946930 | 0.946930 | -0.321439 |
| 4 | 0.000000 | 1.000000 | 0.781832 | 0.623490 | 0.636078 | 0.771625 | 0.996917 | 0.078459 |
| 5 | 0.000000 | 1.000000 | 0.781832 | 0.623490 | 0.639439 | 0.768842 | 0.928810 | 0.370557 |

#### Tail — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 1795769 | 87103 | 2021-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_ANC; DEST_DEN |
| 1795770 | 1717827 | 2021-12-31 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_CLT |
| 1795771 | 686799 | 2021-12-31 | 0 | AIRLINE_CODE_UA; ORIGIN_SEA; DEST_ORD |
| 1795772 | 35853 | 2021-12-31 | 1 | AIRLINE_CODE_NK; ORIGIN_LAS; DEST_CLE |
| 1795773 | 1081407 | 2021-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_LAX; DEST_IAH |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 1795769 | 2.334232 | 2.750345 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1795770 | 1.897359 | 2.269814 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1795771 | 1.291374 | 1.576478 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1795772 | 1.333652 | 1.753244 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1795773 | 0.713574 | 0.989544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1795769 | -0.500000 | 0.866025 | -0.433884 | -0.900969 | -0.043619 | 0.999048 | 0.969231 | -0.246153 |
| 1795770 | -0.500000 | 0.866025 | -0.433884 | -0.900969 | -0.039260 | 0.999229 | 0.928810 | -0.370557 |
| 1795771 | -0.500000 | 0.866025 | -0.433884 | -0.900969 | -0.021815 | 0.999762 | 0.998630 | 0.052336 |
| 1795772 | -0.500000 | 0.866025 | -0.433884 | -0.900969 | -0.004363 | 0.999990 | 0.971342 | -0.237686 |
| 1795773 | -0.500000 | 0.866025 | -0.433884 | -0.900969 | -0.004363 | 0.999990 | 0.977231 | 0.212178 |

### Validation

#### Head — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 1 | 909625 | 2022-01-01 | 0 | AIRLINE_CODE_F9; ORIGIN_LAS; DEST_CLE |
| 2 | 1930992 | 2022-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_PHL |
| 3 | 1096188 | 2022-01-01 | 1 | AIRLINE_CODE_F9; ORIGIN_DEN; DEST_MCO |
| 4 | 1324334 | 2022-01-01 | 1 | AIRLINE_CODE_DL; ORIGIN_FAI; DEST_SEA |
| 5 | 2077976 | 2022-01-01 | 1 | AIRLINE_CODE_AS; ORIGIN_FAI; DEST_SEA |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.390023 | 1.753244 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 2.094656 | 2.745196 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 1.065891 | 1.276146 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 1.122262 | 1.253836 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 1.009520 | 1.253836 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.000000 | 1.000000 | -0.974928 | -0.222521 | 0.021815 | 0.999762 | 0.960050 | -0.279829 |
| 2 | 0.000000 | 1.000000 | -0.974928 | -0.222521 | 0.130526 | 0.991445 | 0.819152 | -0.573576 |
| 3 | 0.000000 | 1.000000 | -0.974928 | -0.222521 | 0.229200 | 0.973379 | 0.991445 | -0.130526 |
| 4 | 0.000000 | 1.000000 | -0.974928 | -0.222521 | 0.362438 | 0.932008 | 0.999657 | -0.026177 |
| 5 | 0.000000 | 1.000000 | -0.974928 | -0.222521 | 0.370557 | 0.928810 | 1.000000 | 0.000000 |

#### Tail — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 669408 | 2890217 | 2022-12-31 | 0 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_SAT |
| 669409 | 197329 | 2022-12-31 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_MIA |
| 669410 | 630497 | 2022-12-31 | 1 | AIRLINE_CODE_B6; ORIGIN_PHX; DEST_FLL |
| 669411 | 86760 | 2022-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_MCI |
| 669412 | 219514 | 2022-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_AUS |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 669408 | 0.967242 | 1.166311 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 669409 | 2.193305 | 2.642226 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 669410 | 1.333652 | 2.007239 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 669411 | 0.924964 | 1.195486 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 669412 | 0.953149 | 1.204067 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 669408 | -0.500000 | 0.866025 | -0.974928 | -0.222521 | -0.065403 | 0.997859 | 0.980785 | 0.195090 |
| 669409 | -0.500000 | 0.866025 | -0.974928 | -0.222521 | -0.030539 | 0.999534 | 0.887011 | -0.461749 |
| 669410 | -0.500000 | 0.866025 | -0.974928 | -0.222521 | -0.026177 | 0.999657 | 0.999048 | 0.043619 |
| 669411 | -0.500000 | 0.866025 | -0.974928 | -0.222521 | -0.021815 | 0.999762 | 0.986286 | 0.165048 |
| 669412 | -0.500000 | 0.866025 | -0.974928 | -0.222521 | -0.004363 | 0.999990 | 0.990268 | 0.139173 |

### Test

#### Head — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 1 | 217507 | 2023-01-01 | 0 | AIRLINE_CODE_UA; ORIGIN_PHX; DEST_ORD |
| 2 | 2635363 | 2023-01-01 | 1 | AIRLINE_CODE_F9; ORIGIN_LAS; DEST_PHL |
| 3 | 2548690 | 2023-01-01 | 0 | AIRLINE_CODE_WN; ORIGIN_LAX; DEST_MDW |
| 4 | 1473188 | 2023-01-01 | 0 | AIRLINE_CODE_B6; ORIGIN_SYR; DEST_JFK |
| 5 | 2070628 | 2023-01-01 | 0 | AIRLINE_CODE_B6; ORIGIN_BNA; DEST_FLL |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.896779 | 1.094231 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 1.883266 | 2.357340 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 1.460486 | 1.626247 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | -0.752065 | -1.018387 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | -0.019246 | -0.016137 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.000000 | 1.000000 | -0.781832 | 0.623490 | 0.254602 | 0.967046 | 0.987688 | 0.156434 |
| 2 | 0.000000 | 1.000000 | -0.781832 | 0.623490 | 0.333807 | 0.942641 | 0.728371 | -0.685183 |
| 3 | 0.000000 | 1.000000 | -0.781832 | 0.623490 | 0.965926 | 0.258819 | 0.237686 | -0.971342 |
| 4 | 0.000000 | 1.000000 | -0.781832 | 0.623490 | 0.968148 | 0.250380 | 0.991445 | -0.130526 |
| 5 | 0.000000 | 1.000000 | -0.781832 | 0.623490 | 0.971342 | 0.237686 | 0.806445 | -0.591310 |

#### Tail — 5 baris

Identitas, target, dan hasil encoding kategori:

| DATASET_ROW | SOURCE_ROW_NUMBER | FL_DATE | DELAYED_15 | One-hot aktif (=1) |
| --- | --- | --- | --- | --- |
| 455671 | 352227 | 2023-08-31 | 0 | AIRLINE_CODE_AS; ORIGIN_PDX; DEST_BOS |
| 455672 | 2470515 | 2023-08-31 | 0 | AIRLINE_CODE_DL; ORIGIN_BOI; DEST_ATL |
| 455673 | 472065 | 2023-08-31 | 0 | AIRLINE_CODE_UA; ORIGIN_DEN; DEST_BWI |
| 455674 | 982074 | 2023-08-31 | 0 | AIRLINE_CODE_DL; ORIGIN_SLC; DEST_MIA |
| 455675 | 2695437 | 2023-08-31 | 0 | AIRLINE_CODE_B6; ORIGIN_SFO; DEST_JFK |

Hasil scaling dan indikator missing:

| DATASET_ROW | CRS_ELAPSED_TIME | DISTANCE | SCHEDULED_DEP_MINUTES_MISSING | SCHEDULED_ARR_MINUTES_MISSING | CRS_ELAPSED_TIME_MISSING | DISTANCE_MISSING |
| --- | --- | --- | --- | --- | --- | --- |
| 455671 | 2.601993 | 2.976881 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 455672 | 1.404115 | 1.777271 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 455673 | 1.009520 | 1.181756 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 455674 | 1.996008 | 2.206316 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 455675 | 2.672457 | 3.060973 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Fitur siklik hasil FE:

| DATASET_ROW | MONTH_SIN | MONTH_COS | DAY_OF_WEEK_SIN | DAY_OF_WEEK_COS | SCHEDULED_DEP_MINUTES_SIN | SCHEDULED_DEP_MINUTES_COS | SCHEDULED_ARR_MINUTES_SIN | SCHEDULED_ARR_MINUTES_COS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 455671 | -0.500000 | -0.866025 | 0.433884 | -0.900969 | -0.021815 | 0.999762 | 0.816642 | -0.577145 |
| 455672 | -0.500000 | -0.866025 | 0.433884 | -0.900969 | -0.013090 | 0.999914 | 0.999962 | 0.008727 |
| 455673 | -0.500000 | -0.866025 | 0.433884 | -0.900969 | -0.004363 | 0.999990 | 0.992546 | 0.121869 |
| 455674 | -0.500000 | -0.866025 | 0.433884 | -0.900969 | -0.004363 | 0.999990 | 0.983255 | -0.182236 |
| 455675 | -0.500000 | -0.866025 | 0.433884 | -0.900969 | -0.004363 | 0.999990 | 0.793353 | -0.608761 |
