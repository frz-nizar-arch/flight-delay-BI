# Laporan Pemodelan — Klasifikasi Keterlambatan Keberangkatan

## 1. Majority Baseline

Laporan ini mendokumentasikan tahap pertama pemodelan: baseline kelas mayoritas. Angka evaluasi berasal dari [hasil majority baseline](modelling/output/%231-majority-baseline/majority-baseline-results.json) yang dijalankan pada 7 November 2026. Cuplikan dataset diambil dari hasil preprocessing yang sama dan diverifikasi terhadap label input baseline.

### 1.1 Mengapa proyek ini menggunakan majority baseline?

Penelitian bertujuan mendeteksi penerbangan yang terlambat berangkat minimal 15 menit. Pada data train, sebagian besar penerbangan berada pada kelas 0, yaitu departure delay kurang dari 15 menit. Model yang selalu memilih kelas 0 dapat memperoleh accuracy tinggi tanpa mendeteksi keterlambatan.

Majority baseline memberikan acuan untuk menilai manfaat model berikutnya. Baseline ini menjawab: **berapa hasil evaluasi jika setiap penerbangan diprediksi sebagai kelas yang paling sering muncul pada train?** Dengan acuan tersebut, peningkatan model dapat dinilai melalui keterlambatan yang berhasil dideteksi, peringatan keliru, dan ketepatan prediksi.

Baseline juga memeriksa alur label dan evaluasi sebelum melatih model berbasis fitur. Hasilnya menjadi pembanding untuk Logistic Regression dan model lain pada validation yang sama.

### 1.2 Dataset yang digunakan

Label berasal dari dataset penerbangan yang telah melalui [preprocessing dan feature engineering](outputs/e1-data-preparation-report.md). Populasi yang digunakan adalah penerbangan dengan `CANCELLED == 0` dan nilai `DEP_DELAY` yang valid. Penerbangan diverted tetap masuk jika memenuhi kedua syarat tersebut.

| Label | Definisi |
| --- | --- |
| 0 | `DEP_DELAY < 15` menit, termasuk keberangkatan lebih awal dan tepat waktu. |
| 1 | `DEP_DELAY >= 15` menit; kelas positif yang ingin dideteksi. |

Baseline membaca dua array label berikut. Path dihitung dari folder `classification`:

| Berkas | Periode | Jumlah baris | Penggunaan |
| --- | --- | --- | --- |
| `datasets/model-ready/y_train.npy` | 2019–2021 | 1.795.773 | Menentukan kelas mayoritas. |
| `datasets/model-ready/y_validation.npy` | 2022 | 669.412 | Membandingkan label aktual dengan prediksi konstan. |

Setiap array berisi satu label 0/1 per penerbangan, dengan urutan yang sejajar dengan CSV prepared pada subset yang sama. Matriks fitur `X_train` dan `X_validation` tidak diperlukan karena keputusan baseline tidak bergantung pada maskapai, bandara, waktu, jarak, atau fitur lainnya. Label dan fitur test tidak digunakan pada tahap ini.

### 1.3 Head dan tail dataset input

Berikut adalah 5 label pertama dan 5 label terakhir dari train serta validation. **Kolom label aktual adalah input baseline.** Tanggal dan nomor record sumber ditambahkan dari metadata hasil preprocessing untuk memudahkan penelusuran; keduanya tidak digunakan untuk menentukan prediksi.

`Posisi pada subset` dihitung mulai 1 dan mengikuti urutan array `.npy`. `Nomor record sumber` adalah `SOURCE_ROW_NUMBER`, yaitu urutan record data pada CSV raw tanpa menghitung header. Baris telah diurutkan dalam preprocessing sehingga nomor record sumber tidak harus berurutan.

Nilai label pada tabel telah dicocokkan langsung dengan `y_train.npy` dan `y_validation.npy`. Kolom prediksi validation diturunkan dari aturan baseline, yaitu selalu memprediksi kelas mayoritas train.

#### Train — menentukan kelas mayoritas

**Head: 5 baris pertama**

| Posisi pada subset | Nomor record sumber | Tanggal | Label aktual `DELAYED_15` |
| --- | --- | --- | --- |
| 1 | 2052433 | 2019-01-01 | 0 |
| 2 | 1869934 | 2019-01-01 | 0 |
| 3 | 945942 | 2019-01-01 | 0 |
| 4 | 2113750 | 2019-01-01 | 0 |
| 5 | 181563 | 2019-01-01 | 0 |

**Tail: 5 baris terakhir**

| Posisi pada subset | Nomor record sumber | Tanggal | Label aktual `DELAYED_15` |
| --- | --- | --- | --- |
| 1795769 | 87103 | 2021-12-31 | 1 |
| 1795770 | 1717827 | 2021-12-31 | 0 |
| 1795771 | 686799 | 2021-12-31 | 0 |
| 1795772 | 35853 | 2021-12-31 | 1 |
| 1795773 | 1081407 | 2021-12-31 | 1 |

#### Validation — mengevaluasi prediksi

**Head: 5 baris pertama**

| Posisi pada subset | Nomor record sumber | Tanggal | Label aktual `DELAYED_15` | Prediksi baseline |
| --- | --- | --- | --- | --- |
| 1 | 909625 | 2022-01-01 | 0 | 0 |
| 2 | 1930992 | 2022-01-01 | 0 | 0 |
| 3 | 1096188 | 2022-01-01 | 1 | 0 |
| 4 | 1324334 | 2022-01-01 | 1 | 0 |
| 5 | 2077976 | 2022-01-01 | 1 | 0 |

**Tail: 5 baris terakhir**

| Posisi pada subset | Nomor record sumber | Tanggal | Label aktual `DELAYED_15` | Prediksi baseline |
| --- | --- | --- | --- | --- |
| 669408 | 2890217 | 2022-12-31 | 0 | 0 |
| 669409 | 197329 | 2022-12-31 | 0 | 0 |
| 669410 | 630497 | 2022-12-31 | 1 | 0 |
| 669411 | 86760 | 2022-12-31 | 1 | 0 |
| 669412 | 219514 | 2022-12-31 | 1 | 0 |

Head dan tail hanya memperlihatkan batas awal/akhir dataset. Distribusi kelas dihitung dari seluruh label, sehingga tidak dapat disimpulkan dari cuplikan kecil ini saja.

### 1.4 Distribusi kelas dan aturan prediksi

| Subset | Label 0 | Label 1 | Total | Proporsi kelas 1 |
| --- | --- | --- | --- | --- |
| train | 1.511.379 | 284.394 | 1.795.773 | 15,8369% |
| validation | 526.729 | 142.683 | 669.412 | 21,3147% |

![Distribusi kelas pada train dan validation](modelling/output/%231-majority-baseline/majority-baseline-class-distribution.png)

**Gambar 1.** Persentase kedua kelas pada masing-masing subset; `n` menunjukkan jumlah penerbangan. Panjang setiap batang dinormalisasi menjadi 100% agar komposisi train dan validation dapat dibandingkan meskipun jumlah penerbangannya berbeda.

Pada train, kelas 0 berjumlah **1.511.379 penerbangan (84,1631%)**, sehingga kelas mayoritas yang dipilih adalah **0**. Aturan prediksinya: **setiap penerbangan pada validation diprediksi memiliki departure delay kurang dari 15 menit**.

Kelas mayoritas ditentukan hanya dari train. Jika jumlah kedua kelas sama, skrip memilih kelas 0 sebagai aturan tetap. Tidak ada optimasi koefisien, tuning threshold, class weighting, atau resampling dalam baseline ini.

Proporsi keterlambatan meningkat dari **15,8369%** pada train menjadi **21,3147%** pada validation. Perubahan distribusi ini perlu diperhatikan saat menilai generalisasi model ke periode berikutnya.

### 1.5 Hasil evaluasi pada validation

| Metrik | Hasil | Interpretasi |
| --- | --- | --- |
| Accuracy | 78,6853% | Proporsi seluruh prediksi yang benar. |
| Precision kelas 1 | Tidak terdefinisi; disajikan sebagai 0 dalam keluaran skrip | Tidak ada prediksi kelas 1, sehingga penyebut precision bernilai nol. |
| Recall kelas 1 | 0% | Tidak ada penerbangan terlambat yang terdeteksi. |
| F1 kelas 1 | 0 | Tidak ada true positive; nilai F1 pada data ini terdefinisi sebagai nol. |

ROC-AUC dan PR-AUC tidak dievaluasi pada keluaran baseline ini, yang hanya memprediksi satu kelas tetap dan tidak memberikan skor untuk membedakan risiko antar-penerbangan. Baseline ini tetap berguna sebagai acuan confusion matrix dan metrik keputusan kelas.

### 1.6 Confusion matrix dan grafik hasil

Baris menunjukkan label aktual, sedangkan kolom menunjukkan prediksi baseline.

| Aktual / Prediksi | 0: delay <15 menit | 1: delay >=15 menit |
| --- | --- | --- |
| 0: delay <15 menit | 526.729 — TN | 0 — FP |
| 1: delay >=15 menit | 142.683 — FN | 0 — TP |

![Confusion matrix majority baseline pada validation](modelling/output/%231-majority-baseline/majority-baseline-confusion-matrix.png)

**Gambar 2.** Seluruh prediksi berada pada kolom kelas 0. Nilai besar pada sel TN menjelaskan accuracy yang tinggi, sedangkan sel FN menunjukkan keterlambatan yang tidak terdeteksi.

- **True negative (TN): 526.729** penerbangan kelas 0 dikenali dengan benar.
- **False positive (FP): 0** peringatan keterlambatan yang keliru.
- **False negative (FN): 142.683** penerbangan terlambat diprediksi sebagai kelas 0.
- **True positive (TP): 0** penerbangan terlambat berhasil dideteksi.

Perhitungan metrik utama:

```text
Accuracy  = (TN + TP) / seluruh validation
          = (526.729 + 0) / 669.412
          = 78,6853%

Recall    = TP / (TP + FN) = 0 / 142.683 = 0
Precision = TP / (TP + FP) = 0 / 0 → tidak terdefinisi
F1        = 2TP / (2TP + FP + FN) = 0 / 142.683 = 0
```

### 1.7 Makna hasil untuk tahap pemodelan berikutnya

Baseline memperoleh accuracy **78,6853%**, tetapi melewatkan seluruh **142.683** penerbangan terlambat pada validation. Ketiadaan false positive terjadi karena baseline tidak pernah mengeluarkan peringatan keterlambatan.

Untuk tujuan deteksi keterlambatan, hasil ini belum menyediakan informasi yang membedakan risiko antar-penerbangan. Model berikutnya perlu dinilai dari jumlah keterlambatan yang terdeteksi beserta jumlah peringatan keliru yang ditimbulkan. Kenaikan accuracy saja tidak cukup untuk menyatakan perbaikan operasional; recall yang lebih tinggi juga harus dibaca bersama precision dan beban peringatan.

Baseline ini dipertahankan sebagai acuan pada validation. Evaluasi test dilakukan setelah pilihan model, parameter, dan aturan keputusan ditetapkan.

## 2. Training Logistic Regression

### 2.1 Mengapa menggunakan Logistic Regression?

Setelah majority baseline menetapkan acuan prediksi konstan, Logistic Regression digunakan untuk mempelajari hubungan antara karakteristik penerbangan dan label keterlambatan. Model menghasilkan probabilitas kelas terlambat sehingga penerbangan dapat diurutkan menurut skor risikonya.

Logistic Regression menjadi baseline berbasis fitur dalam penelitian ini. Bentuk modelnya linear pada log-odds dari fitur yang sudah ditransformasi. Hasil awal ini membantu menilai informasi yang dapat ditangkap oleh maskapai, bandara, kalender, jadwal, durasi, dan jarak sebelum beralih ke model dengan hubungan yang lebih kompleks.

### 2.2 Dataset yang digunakan

Training memakai matriks fitur dan label yang sudah dihasilkan oleh preprocessing. Encoder, median imputer, dan scaler sebelumnya dipelajari hanya dari train. Pada tahap pemodelan ini tidak dilakukan fit ulang preprocessing.

| Berkas relatif terhadap classification | Ukuran | Periode | Peran |
| --- | --- | --- | --- |
| `datasets/model-ready/X_train.npz` | 1.795.773 baris × 790 fitur | 2019–2021 | Masukan untuk mempelajari koefisien model. |
| `datasets/model-ready/y_train.npy` | 1.795.773 label | 2019–2021 | Jawaban aktual bagi setiap baris train. |
| `datasets/model-ready/X_validation.npz` | 669.412 baris × 790 fitur | 2022 | Masukan untuk menghasilkan prediksi evaluasi. |
| `datasets/model-ready/y_validation.npy` | 669.412 label | 2022 | Pembanding bagi prediksi model. |

Label 0/1 dan populasi non-cancelled mengikuti definisi pada bagian majority baseline. File `.npz` menyimpan matriks sparse, sedangkan `.npy` menyimpan array label. Baris X dan y mengikuti urutan yang sama pada setiap subset; file test tidak digunakan.

Sebanyak **790 fitur akhir** terdiri dari 776 kolom one-hot untuk maskapai dan bandara, 2 fitur kontinu yang distandardisasi, 8 fitur sin/cos, dan 4 indikator missing. Daftar serta urutan lengkapnya tersedia pada [feature_names.csv](outputs/feature_names.csv). Nomor record, tanggal mentah, `DEP_DELAY`, dan `CANCELLED` bukan masukan model.

### 2.3 Head dan tail dataset input Logistic Regression

Cuplikan berikut berasal dari 5 baris pertama dan 5 baris terakhir matriks X masing-masing subset, dipasangkan dengan label y. Nilai menunjukkan hasil akhir preprocessing yang benar-benar digunakan model. Tanggal ditambahkan sebagai metadata pembaca dan tidak menjadi fitur.

Untuk menjaga keterbacaan, kolom one-hot ditampilkan sebagai nama-nama kolom yang bernilai 1; seluruh kolom one-hot lainnya pada baris tersebut bernilai 0. Prefix teknis nama fitur disingkat. Durasi dan jarak ditampilkan dalam satuan standardisasi `z = (nilai setelah imputasi − mean train) / skala train`, sehingga nilainya dapat negatif dan tidak lagi bersatuan menit atau mil.

Posisi baris dihitung mulai 1 dalam subset masing-masing. Nilai z dibulatkan menjadi 6 angka desimal dan sin/cos menjadi 4 angka desimal hanya untuk tampilan. Cuplikan lengkap dengan seluruh kolom tersedia dalam [train_head_tail.csv](outputs/train_head_tail.csv) dan [validation_head_tail.csv](outputs/validation_head_tail.csv).

#### Train — masukan untuk mempelajari koefisien

**Head: 5 baris pertama**

Identitas baris, label, encoding kategori, dan fitur yang telah distandardisasi:

| Posisi | Tanggal | Label | One-hot aktif (=1) | Durasi terjadwal (z) | Jarak (z) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2019-01-01 | 0 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_ORD | 1,474579 | 1,791000 |
| 2 | 2019-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_CLT | 1,869174 | 2,269814 |
| 3 | 2019-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_PHX; DEST_DCA | 1,390023 | 2,019252 |
| 4 | 2019-01-01 | 0 | AIRLINE_CODE_NK; ORIGIN_IAG; DEST_FLL | 0,600832 | 0,641160 |
| 5 | 2019-01-01 | 0 | AIRLINE_CODE_B6; ORIGIN_BQN; DEST_MCO | 0,459906 | 0,560499 |

Fitur siklik untuk baris yang sama; setiap sel berisi pasangan `(sin; cos)`:

| Posisi | Bulan | Hari dalam minggu | Waktu keberangkatan | Waktu kedatangan |
| --- | --- | --- | --- | --- |
| 1 | (0,0000; 1,0000) | (0,7818; 0,6235) | (0,0828; 0,9966) | (0,9941; -0,1089) |
| 2 | (0,0000; 1,0000) | (0,7818; 0,6235) | (0,1736; 0,9848) | (0,8339; -0,5519) |
| 3 | (0,0000; 1,0000) | (0,7818; 0,6235) | (0,3214; 0,9469) | (0,9469; -0,3214) |
| 4 | (0,0000; 1,0000) | (0,7818; 0,6235) | (0,6361; 0,7716) | (0,9969; 0,0785) |
| 5 | (0,0000; 1,0000) | (0,7818; 0,6235) | (0,6394; 0,7688) | (0,9288; 0,3706) |

**Tail: 5 baris terakhir**

Identitas baris, label, encoding kategori, dan fitur yang telah distandardisasi:

| Posisi | Tanggal | Label | One-hot aktif (=1) | Durasi terjadwal (z) | Jarak (z) |
| --- | --- | --- | --- | --- | --- |
| 1795769 | 2021-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_ANC; DEST_DEN | 2,334232 | 2,750345 |
| 1795770 | 2021-12-31 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_CLT | 1,897359 | 2,269814 |
| 1795771 | 2021-12-31 | 0 | AIRLINE_CODE_UA; ORIGIN_SEA; DEST_ORD | 1,291374 | 1,576478 |
| 1795772 | 2021-12-31 | 1 | AIRLINE_CODE_NK; ORIGIN_LAS; DEST_CLE | 1,333652 | 1,753244 |
| 1795773 | 2021-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_LAX; DEST_IAH | 0,713574 | 0,989544 |

Fitur siklik untuk baris yang sama; setiap sel berisi pasangan `(sin; cos)`:

| Posisi | Bulan | Hari dalam minggu | Waktu keberangkatan | Waktu kedatangan |
| --- | --- | --- | --- | --- |
| 1795769 | (-0,5000; 0,8660) | (-0,4339; -0,9010) | (-0,0436; 0,9990) | (0,9692; -0,2462) |
| 1795770 | (-0,5000; 0,8660) | (-0,4339; -0,9010) | (-0,0393; 0,9992) | (0,9288; -0,3706) |
| 1795771 | (-0,5000; 0,8660) | (-0,4339; -0,9010) | (-0,0218; 0,9998) | (0,9986; 0,0523) |
| 1795772 | (-0,5000; 0,8660) | (-0,4339; -0,9010) | (-0,0044; 1,0000) | (0,9713; -0,2377) |
| 1795773 | (-0,5000; 0,8660) | (-0,4339; -0,9010) | (-0,0044; 1,0000) | (0,9772; 0,2122) |

#### Validation — masukan untuk mengukur generalisasi

**Head: 5 baris pertama**

Identitas baris, label, encoding kategori, dan fitur yang telah distandardisasi:

| Posisi | Tanggal | Label | One-hot aktif (=1) | Durasi terjadwal (z) | Jarak (z) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2022-01-01 | 0 | AIRLINE_CODE_F9; ORIGIN_LAS; DEST_CLE | 1,390023 | 1,753244 |
| 2 | 2022-01-01 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_PHL | 2,094656 | 2,745196 |
| 3 | 2022-01-01 | 1 | AIRLINE_CODE_F9; ORIGIN_DEN; DEST_MCO | 1,065891 | 1,276146 |
| 4 | 2022-01-01 | 1 | AIRLINE_CODE_DL; ORIGIN_FAI; DEST_SEA | 1,122262 | 1,253836 |
| 5 | 2022-01-01 | 1 | AIRLINE_CODE_AS; ORIGIN_FAI; DEST_SEA | 1,009520 | 1,253836 |

Fitur siklik untuk baris yang sama; setiap sel berisi pasangan `(sin; cos)`:

| Posisi | Bulan | Hari dalam minggu | Waktu keberangkatan | Waktu kedatangan |
| --- | --- | --- | --- | --- |
| 1 | (0,0000; 1,0000) | (-0,9749; -0,2225) | (0,0218; 0,9998) | (0,9600; -0,2798) |
| 2 | (0,0000; 1,0000) | (-0,9749; -0,2225) | (0,1305; 0,9914) | (0,8192; -0,5736) |
| 3 | (0,0000; 1,0000) | (-0,9749; -0,2225) | (0,2292; 0,9734) | (0,9914; -0,1305) |
| 4 | (0,0000; 1,0000) | (-0,9749; -0,2225) | (0,3624; 0,9320) | (0,9997; -0,0262) |
| 5 | (0,0000; 1,0000) | (-0,9749; -0,2225) | (0,3706; 0,9288) | (1,0000; 0,0000) |

**Tail: 5 baris terakhir**

Identitas baris, label, encoding kategori, dan fitur yang telah distandardisasi:

| Posisi | Tanggal | Label | One-hot aktif (=1) | Durasi terjadwal (z) | Jarak (z) |
| --- | --- | --- | --- | --- | --- |
| 669408 | 2022-12-31 | 0 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_SAT | 0,967242 | 1,166311 |
| 669409 | 2022-12-31 | 0 | AIRLINE_CODE_AA; ORIGIN_LAX; DEST_MIA | 2,193305 | 2,642226 |
| 669410 | 2022-12-31 | 1 | AIRLINE_CODE_B6; ORIGIN_PHX; DEST_FLL | 1,333652 | 2,007239 |
| 669411 | 2022-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_MCI | 0,924964 | 1,195486 |
| 669412 | 2022-12-31 | 1 | AIRLINE_CODE_UA; ORIGIN_SFO; DEST_AUS | 0,953149 | 1,204067 |

Fitur siklik untuk baris yang sama; setiap sel berisi pasangan `(sin; cos)`:

| Posisi | Bulan | Hari dalam minggu | Waktu keberangkatan | Waktu kedatangan |
| --- | --- | --- | --- | --- |
| 669408 | (-0,5000; 0,8660) | (-0,9749; -0,2225) | (-0,0654; 0,9979) | (0,9808; 0,1951) |
| 669409 | (-0,5000; 0,8660) | (-0,9749; -0,2225) | (-0,0305; 0,9995) | (0,8870; -0,4617) |
| 669410 | (-0,5000; 0,8660) | (-0,9749; -0,2225) | (-0,0262; 0,9997) | (0,9990; 0,0436) |
| 669411 | (-0,5000; 0,8660) | (-0,9749; -0,2225) | (-0,0218; 0,9998) | (0,9863; 0,1650) |
| 669412 | (-0,5000; 0,8660) | (-0,9749; -0,2225) | (-0,0044; 1,0000) | (0,9903; 0,1392) |

Keempat indikator missing (`SCHEDULED_DEP_MINUTES_MISSING`, `SCHEDULED_ARR_MINUTES_MISSING`, `CRS_ELAPSED_TIME_MISSING`, dan `DISTANCE_MISSING`) bernilai **0 pada seluruh baris cuplikan di atas**. Hal tersebut tidak boleh digeneralisasi hanya dari head/tail; distribusi missing keseluruhan tercatat pada laporan preprocessing.

### 2.4 Konfigurasi dan pelaksanaan training

| Pengaturan | Nilai | Alasan pada percobaan awal |
| --- | --- | --- |
| Solver | `lbfgs` | Mengoptimalkan koefisien Logistic Regression. |
| Regularisasi | L2; `C = 1.0` | Membatasi besarnya koefisien dengan pengaturan awal yang tetap. |
| Bobot kelas | `class_weight = None` | Mencatat performa awal sebelum percobaan pembobotan kelas. |
| Batas iterasi | 1.000 | Memberi ruang bagi solver untuk mencapai kriteria berhenti. |
| Toleransi | `tol = 0.0001` | Kriteria numerik konvergensi. |
| Random state | 42 | Dicatat untuk reproduksibilitas; solver lbfgs tidak mengandalkan pengacakan ini. |
| Threshold keputusan | 0,5 | Acuan pertama: probabilitas kelas 1 ≥0,5 menghasilkan prediksi 1. |

Tidak ada tuning hyperparameter, perubahan fitur, atau resampling pada percobaan ini. Parameter model dipelajari dari train, kemudian probabilitas dan metrik dihitung pada validation.

Training selesai dalam **7,722 detik**, dengan **76 iterasi**. Solver mencapai konvergensi dan tidak mencatat warning. Konvergensi berarti proses optimasi memenuhi kriteria berhenti, bukan jaminan bahwa model sudah memenuhi tujuan deteksi keterlambatan.

### 2.5 Hasil evaluasi dan perbandingan baseline

| Metrik validation | Majority baseline | Logistic Regression |
| --- | --- | --- |
| Accuracy | 78,6853% | 78,6847% |
| Precision kelas terlambat | Tidak terdefinisi; ditampilkan 0 pada keluaran | 46,2963% |
| Recall kelas terlambat | 0% | 0,0175% |
| F1 kelas terlambat | 0 | 0,000350 |
| ROC-AUC | Tidak dievaluasi | 0,661688 |
| PR-AUC (trapesium) | Tidak dievaluasi | 0,342042 |
| Average Precision (AP) | Tidak dievaluasi | 0,342054 |

Accuracy, precision, dan recall pada tabel dinyatakan sebagai persen. F1, ROC-AUC, PR-AUC, dan AP memakai skala 0–1. Precision, recall, dan F1 mengacu pada label 1. Perbandingan menggunakan label validation yang sama.

Accuracy Logistic Regression hampir sama dengan baseline mayoritas. Model memperoleh 25 true positive tambahan, tetapi juga 29 false positive; jumlah prediksi benar menjadi empat lebih sedikit daripada baseline. Perbedaan accuracy yang sangat kecil ini tidak menunjukkan peningkatan kemampuan operasional.

PR-AUC (trapesium) merupakan luas kurva precision–recall dengan interpolasi linear. AP merangkum precision dengan bobot kenaikan recall tanpa interpolasi linear. Keduanya dihitung dari skor probabilitas dan dapat berbeda; keduanya tidak sama dengan accuracy.

### 2.6 Confusion matrix dan grafik evaluasi

| Aktual / Prediksi | 0: delay <15 menit | 1: delay >=15 menit |
| --- | --- | --- |
| 0: delay <15 menit | 526.700 — TN | 29 — FP |
| 1: delay >=15 menit | 142.658 — FN | 25 — TP |

![Confusion matrix Logistic Regression pada validation](modelling/output/%232-LR-train/logistic-regression-confusion-matrix.png)

**Gambar 3.** Confusion matrix pada threshold 0,5. Model hanya mengeluarkan **54 prediksi terlambat**: 25 benar dan 29 keliru. Sebanyak 142.658 penerbangan terlambat lainnya tidak terdeteksi.

![Kurva ROC dan precision-recall Logistic Regression pada validation](modelling/output/%232-LR-train/logistic-regression-curves.png)

**Gambar 4.** Kurva ROC dan precision–recall dari probabilitas validation. ROC-AUC **0,6617** berada di atas acuan pemeringkatan acak 0,5. AP **0,3421** berada di atas proporsi keterlambatan validation **0,2131**, yang menjadi acuan tingkat positif pada kurva PR. Ini menunjukkan adanya informasi untuk membedakan risiko, meskipun pada threshold 0,5 deteksinya masih sangat terbatas.

### 2.7 Mengapa recall pada threshold 0,5 sangat rendah?

Sebagian besar probabilitas yang dihasilkan model berada di bawah 0,5. Ringkasan berikut dihitung dari probabilitas validation yang tersimpan:

| Ringkasan probabilitas kelas terlambat | Nilai |
| --- | --- |
| Median | 14,7103% |
| Persentil ke-95 | 29,4304% |
| Persentil ke-99 | 35,2988% |
| Maksimum | 54,0884% |
| Rata-rata prediksi | 16,0061% |
| Proporsi terlambat aktual | 21,3147% |

Dengan ambang 50%, hanya **54** baris yang menjadi prediksi kelas 1. Recall **0,0175%** berarti hanya 25 dari 142.683 penerbangan terlambat berhasil ditemukan. Precision **46,2963%** tersebut dihitung dari 54 peringatan, sehingga tidak dapat dibaca terpisah dari recall yang rendah.

Rata-rata probabilitas 16,0061% lebih rendah daripada tingkat keterlambatan aktual 21,3147%. Model meremehkan tingkat keterlambatan secara agregat pada validation. Perbedaan rata-rata ini menjadi alasan untuk memeriksa kalibrasi; belum menjelaskan kualitas kalibrasi pada setiap rentang skor atau kelompok penerbangan.

Menurunkan threshold akan mengubah keputusan kelas dan pertukaran antara false negative dan false positive. Perubahan threshold saja tidak memperbaiki kemampuan pemeringkatan ROC-AUC/AP, sehingga keterbatasan model dan fitur tetap perlu dievaluasi.

## 3. Langkah selanjutnya

### 3.1 Analisis threshold pada validation

Langkah terdekat adalah menggunakan probabilitas yang sudah tersimpan untuk membandingkan beberapa threshold, misalnya 0,10; 0,15; 0,20; 0,25; 0,30; 0,40; dan 0,50. Langkah ini tidak memerlukan training ulang.

Untuk setiap threshold, catat precision, recall, F1, accuracy, TP, FP, FN, serta jumlah penerbangan yang mendapat peringatan. Sajikan kurva metrik terhadap threshold dan tabel pertukaran deteksi dengan peringatan keliru.

Pemilihan threshold perlu mengikuti tujuan evaluasi. F1 tertinggi pada validation dapat menjadi kandidat pembanding awal penelitian; untuk penggunaan operasional perlu ditetapkan kebutuhan minimum recall atau batas peringatan keliru. Threshold belum ditetapkan sebagai pilihan akhir pada laporan ini. Hasil pada validation yang dipakai memilih threshold juga belum merupakan estimasi akhir performa pada data baru.

### 3.2 Periksa kalibrasi dan perubahan antarperiode

Periksa hubungan probabilitas prediksi dengan proporsi keterlambatan aktual melalui reliability plot dan ukuran seperti Brier score. Analisis per bulan dapat membantu melihat apakah kesenjangan rata-rata skor dan label terkonsentrasi pada periode tertentu. Jika dilakukan kalibrasi, siapkan bagian data untuk mempelajari kalibrator dan bagian evaluasi yang terpisah secara waktu; jangan menggunakan test untuk memilih kalibrator.

### 3.3 Bandingkan Logistic Regression dengan bobot kelas

Setelah hasil threshold baseline dipahami, lakukan percobaan terpisah dengan `class_weight="balanced"`. Pertahankan fitur serta split waktu agar pengaruh pembobotan dapat dibandingkan. Tinjau skor pemeringkatan, precision/recall pada kriteria keputusan yang sama, serta kalibrasi. Pembobotan dapat mengubah koefisien dan probabilitas; peningkatan performa tidak diasumsikan sebelumnya.

### 3.4 Perluas perbandingan model dan lakukan evaluasi akhir

Bandingkan model dalam research plan, seperti Random Forest, XGBoost, dan MLP, setelah baseline Logistic Regression terdokumentasi. Gunakan train untuk pembelajaran dan validation untuk pemilihan model, hyperparameter, dan aturan keputusan.

Setelah pilihan tersebut dikunci, lakukan evaluasi pada test Januari–Agustus 2023. Jika memakai cross-validation selama pengembangan, preprocessing perlu dipelajari ulang pada bagian train setiap fold dari CSV prepared. Test tidak digunakan untuk memilih threshold, bobot kelas, fitur, atau konfigurasi model.

Bagian langkah berikutnya merupakan rencana kerja. Laporan ini mencatat hasil training awal pada threshold 0,5; belum menambahkan hasil training dengan pembobotan kelas, kalibrasi, atau evaluasi test.

