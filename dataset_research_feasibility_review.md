# Audit kesesuaian dataset dan kelayakan rencana riset

Tanggal pemeriksaan: 30 Oktober 2026.

**Keputusan: layak dikerjakan sebagai riset dan prototipe Predictive BI. E1–E3 didukung data; E4 perlu dibatasi pada bandara dengan observasi memadai. Kamus fitur dan beberapa definisi metodologi perlu diperbaiki sebelum pelatihan.** Kelayakan data tidak membuktikan akurasi model atau manfaat operasional; keduanya masih harus diuji.

Audit mencakup seluruh `flights_sample_3m.csv`, seluruh kamus fitur, dan `flight_disruption_research_plan.md`. Skrip `audit_dataset.py` menghasilkan `dataset_audit_results.json`. Berkas sumber tidak diubah. Belum dilakukan pelatihan model, pengukuran kebutuhan perangkat keras, atau validasi terhadap file sumber asli BTS/Kaggle.

## 1. Bukti dari seluruh dataset

| Pemeriksaan | Hasil |
|---|---:|
| Baris / kolom dataset | 3.000.000 / 32 |
| Fitur yang dijelaskan kamus | 30 |
| Kolom daftar utama bagian 6 research plan yang tidak tersedia | 0 |
| Periode aktual | 1 Januari 2019–31 Agustus 2023 |
| Tanggal teramati | 1.704; tidak ada tanggal kalender yang hilang secara global |
| Maskapai / bandara asal / bandara tujuan | 18 / 380 / 380 |
| Rute berarah | 7.785 |
| Cancelled | 79.140 (2,638% seluruh baris) |
| Diverted | 7.056 (0,2352% seluruh baris) |
| Non-cancelled dengan `DEP_DELAY` tersedia | 2.920.860 |
| `DEP_DELAY >= 15` pada kelompok tersebut | 531.033 (18,1807%) |
| Tanggal gagal diparsing | 0 |
| Kandidat duplikat seluruh baris | 0 |
| Kandidat duplikat kunci penerbangan | 0 |

Pemeriksaan duplikat memakai hash seluruh baris dan hash kunci `(FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, DEST, CRS_DEP_TIME)`, tanpa indeks. Tidak ada hash berulang. Dataset tidak terurut kronologis; urutkan berdasarkan waktu sebelum membuat fitur historis atau membagi data.

## 2. Kesesuaian kamus fitur

Semua 30 nama kolom dalam kamus tersedia. Tipe hasil pembacaan CSV konsisten dengan tipe yang dicantumkan, tetapi tipe penyimpanan perlu dibedakan dari tipe untuk pemodelan.

| Temuan | Perbaikan yang diperlukan |
|---|---|
| `AIRLINE` tidak dijelaskan | Tambahkan sebagai nama maskapai, string/kategori. |
| `AIRLINE_DOT` tidak dijelaskan | Nilai aktual berbentuk `United Air Lines Inc.: UA`; deskripsikan sebagai label nama maskapai dan kode, bukan nomor DOT. |
| Deskripsi `FL_DATE` menyebut `yyyymmdd` | Format file aktual adalah `YYYY-MM-DD`; parse menjadi tanggal. |
| `DOT_CODE` dan `FL_NUMBER` bertipe integer | Perlakukan sebagai pengenal/kategori jika digunakan; besar angkanya tidak menunjukkan besarnya risiko. |
| `CANCELLED` dan `DIVERTED` bertipe float | Nilai aktual hanya 0 dan 1; tipe logisnya indikator biner. |
| Jam bertipe integer/float | Nilai memakai HHMM waktu lokal, bukan angka jam desimal atau durasi menit. |
| `CANCELLATION_CODE` tanpa penjelasan kode | Tambahkan pemetaan kode dan aturan null, setelah memeriksa provenance dataset. |
| Peran variabel belum dijelaskan | Tambahkan satuan, nullable, aturan missing, peran target/fitur/diagnostik, dan waktu ketersediaan. |

Dalam sampel, empat kolom identitas maskapai berkorespondensi satu-ke-satu. Pilih satu identitas utama untuk model; nama/label lain dapat dipakai di dashboard.

Nama dan arti fitur pokok sesuai dengan [kamus resmi BTS](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ), termasuk delay negatif untuk keberangkatan lebih awal. Namun, kesesuaian skema tidak membuktikan file lokal identik dengan sumber BTS.

## 3. Kelayakan tiap komponen research plan

| Komponen | Status | Ketentuan |
|---|---|---|
| E1: klasifikasi delay | Layak | Gunakan `DEP_DELAY >= 15` pada penerbangan non-cancelled dengan label tersedia. |
| E2: regresi durasi delay | Layak | Tetapkan apakah target adalah selisih bertanda `DEP_DELAY` atau durasi nonnegatif `max(DEP_DELAY, 0)`. |
| E3: klasifikasi pembatalan | Layak | Semua 3 juta baris memiliki `CANCELLED`; fokus pada precision, recall, PR-AUC dan kalibrasi. |
| E4: forecasting bandara | Layak bersyarat | Pilih bandara dengan cakupan memadai; nilai adalah statistik sampel, bukan seluruh operasi bandara. |
| Perbandingan ML dan MLP | Layak | Gunakan split dan prosedur tuning yang setara; kemenangan DL belum dapat ditentukan. |
| LSTM/GRU | Layak sebagai eksperimen | Gunakan deret agregat teratur dengan baseline sederhana; jumlah baris penerbangan tidak sama dengan jumlah timestep. |
| SHAP / feature importance | Layak | Menjelaskan kontribusi terhadap prediksi; tidak membuktikan sebab-akibat. |
| Dashboard maskapai, bandara, rute | Layak | Cantumkan cakupan sampel, periode, denominator KPI dan ukuran sampel kelompok. |
| Peta bandara | Perlu data tambahan | Koordinat lintang/bujur tidak tersedia dalam CSV. |
| Prediksi upcoming flights secara langsung | Perlu data tambahan | Memerlukan jadwal terbaru dan pembaruan fitur historis; CSV mendukung simulasi historis/form input. |
| What-if waktu/rute/maskapai | Layak sebagai eksplorasi model | Batasi pada kombinasi masuk akal; perubahan prediksi bukan estimasi efek kausal. |

## 4. Revisi metodologi yang perlu diprioritaskan

### 4.1 Tetapkan waktu prediksi dan daftar fitur yang diizinkan

Bagian 8 research plan menggunakan nama `ACTUAL_DEP_TIME` dan `ACTUAL_ARR_TIME`, sedangkan nama sebenarnya adalah `DEP_TIME` dan `ARR_TIME`.

Untuk prediksi sebelum keberangkatan, kandidat fitur dasar adalah:

```text
FL_DATE -> fitur kalender
AIRLINE_CODE atau DOT_CODE
FL_NUMBER (opsional, kategorikal)
ORIGIN, DEST -> route
CRS_DEP_TIME, CRS_ARR_TIME
CRS_ELAPSED_TIME
DISTANCE
statistik historis yang tersedia sebelum waktu prediksi
```

Tetapkan horizon, misalnya H-24 jam, dan asumsi bahwa jadwal pada file sudah diketahui saat itu. CSV tidak memiliki riwayat revisi jadwal, sehingga asumsi ini belum bisa diverifikasi.

Jangan masukkan kolom berikut sebagai prediktor sebelum keberangkatan:

```text
DEP_TIME, DEP_DELAY, TAXI_OUT, WHEELS_OFF,
WHEELS_ON, TAXI_IN, ARR_TIME, ARR_DELAY,
ELAPSED_TIME, AIR_TIME,
CANCELLED, CANCELLATION_CODE, DIVERTED,
semua DELAY_DUE_*
```

Kolom target tetap dipakai sebagai label sesuai eksperimen. Pola null dari kolom hasil penerbangan juga dapat membocorkan pembatalan; jangan jadikan missing indicator-nya sebagai fitur pra-keberangkatan.

### 4.2 Pisahkan populasi target

- **E1/E2:** `CANCELLED == 0` dan `DEP_DELAY.notna()`, menghasilkan 2.920.860 baris. Semua penerbangan non-cancelled memiliki `DEP_DELAY`.
- **Diverted:** 7.056 baris memiliki departure delay, sehingga dapat dipertahankan jika targetnya keberangkatan. Jika penelitian mengecualikannya, ukuran E1/E2 menjadi 2.913.804 dan kebijakan harus konsisten.
- **E3:** gunakan seluruh 3.000.000 baris, termasuk cancelled.
- Sebanyak 1.496 penerbangan cancelled masih memiliki `DEP_DELAY`. Karena itu, filter `DEP_DELAY.notna()` saja tidak cukup untuk populasi E1/E2 yang disarankan.
- Untuk target arrival delay, buat populasi tersendiri: seluruh cancelled, seluruh diverted, dan dua penerbangan non-cancelled/non-diverted tidak memiliki `ARR_DELAY`.

Jangan mengubah target delay yang kosong menjadi 0 atau membentuk label langsung dari perbandingan NaN, karena itu akan memasukkan pembatalan sebagai penerbangan tidak terlambat.

Untuk istilah **durasi keterlambatan**, gunakan `DelayMinutes = max(DEP_DELAY, 0)` pada populasi E2. Jika memilih nilai bertanda, jelaskan bahwa target mencakup keberangkatan lebih awal. Jangan mengganti target dengan durasi arrival tanpa mengubah definisi E1, KPI, dan interpretasi.

### 4.3 Gunakan split waktu yang sesuai periode nyata

| Subset | Periode | Baris sebelum filter target |
|---|---|---:|
| Train | 2019–2021 | 1.848.656 |
| Validation | 2022 | 687.860 |
| Test | Januari–Agustus 2023 | 463.484 |

Split contoh dalam research plan dapat digunakan, tetapi test bukan seluruh tahun 2023. Jangan menyimpulkan performa September–Desember dari hasil ini. Laporkan hasil per bulan/tahun karena prevalensi target berubah cukup besar: cancellation rate tahun 2020 sekitar 6,0%, dibanding sekitar 1,7% pada Januari–Agustus 2023.

Encoder, imputasi, scaling, resampling dan pemilihan fitur harus dipelajari dari train. Pilih hiperparameter, threshold, dan kalibrasi menggunakan validation atau subset kalibrasi yang sesuai; test digunakan untuk evaluasi akhir. Untuk klaim ML vs DL lebih baik secara signifikan, sertakan variasi seed dan interval ketidakpastian yang memperhatikan pengelompokan waktu.

### 4.4 Hitung fitur historis dengan batas ketersediaan yang jelas

Rolling 7/30 hari harus menggunakan hasil penerbangan yang sudah diketahui pada waktu prediksi. Jangan menghitung agregat seluruh dataset lalu memasukkannya kembali sebagai fitur. Penggeseran satu baris saja tidak cukup bila urutan atau kelompok tanggal tidak ditangani dengan benar.

Skenario replay historis boleh memperbarui fitur memakai outcome hari-hari sebelumnya yang sudah tersedia, tetapi tidak boleh menggunakan outcome masa depan dalam horizon prediksi. File tidak memuat timestamp publikasi outcome. Tentukan jeda ketersediaan secara eksplisit; bahkan outcome penerbangan tanggal kemarin dapat belum selesai jika delay panjang. Bila sumber hanya tersedia sebagai publikasi berkala, gunakan lag yang sesuai publikasinya.

Untuk rute/bandara dengan sedikit observasi, tambahkan jumlah observasi historis, smoothing, dan fallback ke maskapai/bandara/global dari data masa lalu. Data tanpa observasi bukan berarti delay rate 0.

### 4.5 Batasi E4 berdasarkan kepadatan sampel

Terdapat 338.264 pasangan bandara asal–tanggal yang teramati. Median hanya **3 penerbangan sampel per pasangan**; **261.492 (77,30%)** memiliki kurang dari 10. Sebanyak 4.239 pasangan memiliki nol penerbangan yang memenuhi populasi departure-delay non-cancelled.

Sebanyak 37 bandara memiliki setidaknya satu observasi pada seluruh 1.704 hari. Contoh cakupan seluruh periode: ATL rata-rata 90,1 penerbangan sampel/hari, DFW 76,5, ORD 71,8. JFK memiliki rata-rata 29,6 dan satu hari tanpa penerbangan non-cancelled yang eligible.

Usulan implementasi:

1. Pilih kandidat 5–10 bandara dengan volume dan cakupan baik berdasarkan **train**, kemudian laporkan cakupan validation/test. Statistik seluruh periode di atas hanya hasil audit.
2. Buat kalender harian lengkap; tandai denominator nol/missing dan jangan mengisinya sebagai delay rate nol.
3. Definisikan target besok, misalnya `delayed_non_cancelled / eligible_non_cancelled`, beserta horizon dan jeda tersedianya label.
4. Bandingkan dengan persistence dan rata-rata historis/seasonal naive sebelum menilai XGBoost, LSTM atau GRU.
5. Pertimbangkan agregasi mingguan bila sampel harian terlalu sedikit; ini mengurangi jumlah timestep sehingga evaluasi DL juga perlu disesuaikan.

Tanpa dokumentasi cara sampling atau data populasi pembanding, jangan menafsirkan `Total Flights`, volume kemacetan, atau cancellation count sebagai jumlah aktual seluruh bandara. Bahkan rate sampel memerlukan penilaian representativitas. Label dashboard sebaiknya menyebut **penerbangan dalam sampel**.

## 5. Kualitas data dan interpretasi

| Temuan aktual | Penanganan |
|---|---|
| `DEP_DELAY` kosong pada 77.644 baris, semuanya cancelled | Missing struktural; jangan imputasi target sebagai nol. |
| `ARR_DELAY`, `ELAPSED_TIME`, `AIR_TIME` masing-masing kosong pada 86.198 baris | 79.140 cancelled + 7.056 diverted + 2 lainnya; periksa dua sisanya. |
| `CRS_ELAPSED_TIME` kosong pada 14 baris, semuanya cancelled | Untuk E3, imputasi dari train dengan aturan terdokumentasi. |
| `CANCELLATION_CODE` kosong pada 2.920.860 baris | Tepat pada seluruh non-cancelled; konsisten secara struktural. |
| Semua lima `DELAY_DUE_*` kosong pada 2.466.137 baris (82,20%) | Tidak ada missing parsial antarkolom penyebab; jangan langsung membuang baris atau menganggap seluruhnya data rusak. |
| Setiap `ARR_DELAY >= 15` memiliki kolom penyebab; tidak ada penyebab terisi pada `ARR_DELAY < 15` | Analisis penyebab harus mengikuti konteks arrival delay, walaupun E1 menargetkan departure delay. |
| `DEP_DELAY` berkisar -90 sampai 2.966 menit | Nilai negatif berarti lebih awal; nilai ekstrem perlu inspeksi, bukan otomatis dibuang. |
| Kuantil `DEP_DELAY` non-cancelled: median -2; P95 72; P99 191; P99,9 652 menit | Laporkan MAE dan RMSE serta error menurut tingkat delay; evaluasi dampak ekor panjang. |
| Jam `2400` muncul, termasuk 14 `CRS_ARR_TIME` | Normalisasi batas tengah malam secara eksplisit; jangan parse sebagai jam 24 biasa. Tanggal arrival dan zona waktu tidak tersedia langsung. |
| Tidak ada HHMM lain yang gagal pemeriksaan rentang/menit | Validitas format tidak menjamin konsistensi seluruh timestamp. |
| Satu `CRS_ELAPSED_TIME < 10`, minimum 1 menit | Periksa rekaman asal sebelum menetapkan aturan pembersihan durasi. |

Pelaporan penyebab keterlambatan untuk kedatangan terlambat setidaknya 15 menit dijelaskan oleh [BTS](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting). Ini membantu menjelaskan pola missing yang ditemukan; `DELAY_DUE_WEATHER` adalah menit delay yang dikaitkan dengan cuaca setelah kejadian, bukan pengukuran cuaca untuk prediksi.

**Temuan yang perlu verifikasi sumber:** kode pembatalan D berjumlah 24.417, dan 24.221 di antaranya berada pada 2020. Dalam [dokumentasi BTS, D berarti Security](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting). Konsentrasi ini perlu ditelusuri pada sumber dataset dan proses transformasinya sebelum menampilkan narasi sebab pembatalan. Audit lokal tidak membuktikan salah label dan tidak cukup untuk menggantinya menjadi kategori lain. Target biner `CANCELLED` tetap dapat dipakai untuk eksperimen dengan batasan provenance tersebut.

## 6. Risk score dan penjelasan dashboard

- Jika E1 dilatih pada non-cancelled, `P_d` adalah probabilitas delay **bersyarat tidak cancelled**, bukan probabilitas delay seluruh penerbangan. Jelaskan makna ini pada integrasi model.
- Formula bobot bagian 21 dapat digunakan sebagai indeks heuristik, tetapi bukan probabilitas gangguan yang tervalidasi. Bobot, kalibrasi, dan threshold kategori perlu diuji.
- Gunakan `D_n = min(max(D, 0) / 120, 1)` untuk menjaga komponen durasi tetap dalam rentang 0–1.
- Kategori harus mencakup skor pecahan: misalnya `[0,25]`, `(25,50]`, `(50,75]`, `(75,100]`. Rentang 0–25, 26–50, dan seterusnya pada rencana menyisakan celah.
- Persentase kontribusi contoh SHAP tidak boleh langsung dianggap perubahan probabilitas atau bukti penyebab. Nyatakan satuan output explainer dan baseline-nya.
- Data tidak memuat cuaca aktual/prakiraan, nomor registrasi pesawat, rotasi pesawat, kapasitas bandara, atau kejadian ATC. Jadi klaim penyebab operasional dan prediksi gangguan spesifik memiliki batasan.
- Dashboard prediksi harus menyediakan seluruh fitur model atau cara memperoleh nilai jadwalnya. Form contoh rencana belum memasukkan scheduled arrival, elapsed time dan distance jika ketiganya dipakai model.

## 7. Urutan pengerjaan yang disarankan

1. Lengkapi kamus dan catatan sumber/sampling; tetapkan departure delay sebagai target utama, populasi, horizon, fitur yang diizinkan, dan kebijakan missing.
2. Bangun preprocessing serta split waktu; mulai baseline E1/E2/E3 dan satu model tabular pembanding.
3. Tambahkan fitur historis setelah validasi ketersediaan waktu; evaluasi generalisasi, kalibrasi dan error tiap periode/kelompok.
4. Tambahkan MLP untuk perbandingan ML–DL, lalu SHAP dan indeks risiko.
5. Bangun dashboard historis dan simulasi prediksi dengan periode/ukuran sampel yang jelas.
6. Kerjakan E4 pada subset bandara yang memenuhi kriteria, dengan baseline dan LSTM/GRU sesuai kapasitas eksperimen.

Rencana dapat dikerjakan dengan dataset ini setelah penyesuaian tersebut. Prioritas utamanya adalah definisi target dan pencegahan leakage; hambatan data terbesar berada pada representativitas sampel, forecasting bandara kecil, provenance kode pembatalan, dan klaim penggunaan operasional terkini.
