# Dataset Understanding — Bagian 6

Dibuat (UTC): 2026-11-01T10:23:24.915265+00:00

Seluruh baris dibaca. CSV sumber tidak dibersihkan atau diubah. Statistik menggambarkan sampel, bukan seluruh operasi penerbangan.

## Ringkasan

| Aspek | Hasil |
| --- | --- |
| Baris | 3,000,000 |
| Kolom | 32 |
| Ukuran CSV (MiB) | 585.6902 |
| Tanggal awal | 2019-01-01 |
| Tanggal akhir | 2023-08-31 |
| Tanggal teramati | 1,704 |
| Hari kalender tanpa data secara global | 0 |
| Maskapai | 18 |
| Bandara asal | 380 |
| Bandara tujuan | 380 |
| Rute berarah dengan asal/tujuan terisi | 7,785 |
| Cancelled | 79,140 |
| Cancellation rate (% seluruh baris) | 2.6380 |
| Diverted (termasuk jika juga cancelled) | 7,056 |
| Populasi E1/E2: non-cancelled, DEP_DELAY tersedia | 2,920,860 |
| Departure delay >=15 menit pada populasi E1 | 531,033 |
| Delay rate (% populasi E1) | 18.1807 |

### Penjelasan istilah pada ringkasan

- **Hari kalender tanpa data secara global** adalah jumlah tanggal di antara tanggal awal dan akhir dataset yang tidak memiliki satu pun baris penerbangan. Kata “global” berarti seluruh dataset diperiksa bersama, tanpa dipisahkan per bandara, maskapai, atau rute. Nilai 0 berarti setiap tanggal memiliki setidaknya satu catatan; bukan berarti setiap bandara memiliki data setiap hari atau seluruh penerbangan tercatat.

- **Cancelled** adalah jumlah baris penerbangan yang ditandai dibatalkan (`CANCELLED == 1`). Baris ini tetap diperlukan untuk penelitian pembatalan (E3), tetapi dikeluarkan dari populasi E1/E2 yang digunakan dalam laporan.

- **Cancellation rate (% seluruh baris)** adalah persentase penerbangan yang ditandai dibatalkan terhadap seluruh baris dataset: `jumlah CANCELLED == 1 / jumlah seluruh baris × 100%`. Ini adalah tingkat pembatalan dalam sampel, bukan probabilitas hasil prediksi model. Jika status pembatalan ada yang kosong atau tidak valid, baris tersebut tetap masuk penyebut sesuai definisi “seluruh baris”; jumlahnya dapat diperiksa pada bagian kualitas data.

- **Diverted (termasuk jika juga cancelled)** adalah jumlah baris yang ditandai mengalami pengalihan penerbangan (`DIVERTED == 1`), misalnya dialihkan ke bandara lain dari tujuan yang dijadwalkan. Penghitungan ini tidak memfilter `CANCELLED`, sehingga baris dengan kedua indikator bernilai 1 akan tetap dihitung jika ada. Frasa dalam kurung menjelaskan aturan hitung, bukan menyatakan bahwa kedua status tersebut pasti muncul bersamaan.

- **Populasi E1/E2: non-cancelled, DEP_DELAY tersedia** adalah kumpulan baris yang memenuhi `CANCELLED == 0` dan nilai `DEP_DELAY` tidak kosong. E1 adalah eksperimen klasifikasi apakah keterlambatan keberangkatan mencapai 15 menit; E2 adalah eksperimen prediksi jumlah menit keterlambatan. `DEP_DELAY` menyimpan selisih menit antara keberangkatan aktual dan terjadwal. Populasi ini mencakup penerbangan lebih awal, tepat waktu, dan terlambat; penerbangan diverted tetap masuk jika memenuhi kedua syarat. Kata “populasi” di sini merujuk pada data yang memenuhi syarat eksperimen di dalam sampel, sebelum pembagian train/validation/test.

## Cakupan per tahun

| Periode | Baris | Cancelled | Eligible E1 | Delay >=15 | Delay rate (%) |
| --- | --- | --- | --- | --- | --- |
| 2019 | 757,673 | 13,594 | 744,079 | 138,677 | 18.6374 |
| 2020 | 479,350 | 28,757 | 450,593 | 40,939 | 9.0856 |
| 2021 | 611,633 | 10,532 | 601,101 | 104,778 | 17.4310 |
| 2022 | 687,860 | 18,448 | 669,412 | 142,683 | 21.3147 |
| 2023 | 463,484 | 7,809 | 455,675 | 103,956 | 22.8136 |

## Cakupan per bulan

| Periode | Baris | Cancelled | Eligible E1 | Delay >=15 | Delay rate (%) |
| --- | --- | --- | --- | --- | --- |
| 2019-01 | 59,412 | 1,666 | 57,746 | 10,028 | 17.3657 |
| 2019-02 | 54,565 | 1,562 | 53,003 | 11,441 | 21.5856 |
| 2019-03 | 64,894 | 1,315 | 63,579 | 10,905 | 17.1519 |
| 2019-04 | 62,080 | 1,486 | 60,594 | 11,038 | 18.2163 |
| 2019-05 | 64,966 | 1,372 | 63,594 | 12,507 | 19.6669 |
| 2019-06 | 64,777 | 1,260 | 63,517 | 15,281 | 24.0581 |
| 2019-07 | 67,213 | 1,338 | 65,875 | 13,792 | 20.9366 |
| 2019-08 | 67,469 | 1,127 | 66,342 | 13,387 | 20.1788 |
| 2019-09 | 61,987 | 956 | 61,031 | 8,205 | 13.4440 |
| 2019-10 | 64,869 | 487 | 64,382 | 10,282 | 15.9703 |
| 2019-11 | 61,662 | 461 | 61,201 | 8,769 | 14.3282 |
| 2019-12 | 63,779 | 564 | 63,215 | 13,042 | 20.6312 |
| 2020-01 | 62,160 | 752 | 61,408 | 8,349 | 13.5959 |
| 2020-02 | 58,479 | 529 | 57,950 | 8,384 | 14.4676 |
| 2020-03 | 66,305 | 11,181 | 55,124 | 5,171 | 9.3807 |
| 2020-04 | 32,074 | 13,350 | 18,724 | 841 | 4.4916 |
| 2020-05 | 18,645 | 1,232 | 17,413 | 892 | 5.1226 |
| 2020-06 | 23,147 | 73 | 23,074 | 1,452 | 6.2928 |
| 2020-07 | 36,088 | 280 | 35,808 | 2,605 | 7.2749 |
| 2020-08 | 38,528 | 392 | 38,136 | 2,620 | 6.8701 |
| 2020-09 | 32,792 | 174 | 32,618 | 1,922 | 5.8925 |
| 2020-10 | 35,935 | 187 | 35,748 | 2,525 | 7.0633 |
| 2020-11 | 37,313 | 203 | 37,110 | 2,309 | 6.2220 |
| 2020-12 | 37,884 | 404 | 37,480 | 3,869 | 10.3228 |
| 2021-01 | 36,914 | 398 | 36,516 | 3,153 | 8.6346 |
| 2021-02 | 33,742 | 1,866 | 31,876 | 3,803 | 11.9306 |
| 2021-03 | 45,143 | 546 | 44,597 | 4,553 | 10.2092 |
| 2021-04 | 45,735 | 247 | 45,488 | 4,894 | 10.7589 |
| 2021-05 | 50,421 | 226 | 50,195 | 7,040 | 14.0253 |
| 2021-06 | 55,393 | 897 | 54,496 | 13,197 | 24.2165 |
| 2021-07 | 59,653 | 942 | 58,711 | 14,952 | 25.4671 |
| 2021-08 | 59,484 | 1,829 | 57,655 | 13,184 | 22.8671 |
| 2021-09 | 55,127 | 672 | 54,455 | 7,809 | 14.3403 |
| 2021-10 | 57,396 | 1,166 | 56,230 | 10,781 | 19.1730 |
| 2021-11 | 55,885 | 375 | 55,510 | 8,868 | 15.9755 |
| 2021-12 | 56,740 | 1,368 | 55,372 | 12,544 | 22.6540 |
| 2022-01 | 54,895 | 3,437 | 51,458 | 10,053 | 19.5363 |
| 2022-02 | 50,685 | 2,249 | 48,436 | 9,629 | 19.8798 |
| 2022-03 | 57,748 | 938 | 56,810 | 12,270 | 21.5983 |
| 2022-04 | 56,739 | 1,286 | 55,453 | 12,633 | 22.7815 |
| 2022-05 | 59,051 | 1,184 | 57,867 | 12,238 | 21.1485 |
| 2022-06 | 59,151 | 1,809 | 57,342 | 14,368 | 25.0567 |
| 2022-07 | 60,744 | 1,043 | 59,701 | 14,304 | 23.9594 |
| 2022-08 | 60,169 | 1,446 | 58,723 | 13,409 | 22.8343 |
| 2022-09 | 57,022 | 853 | 56,169 | 9,488 | 16.8919 |
| 2022-10 | 58,590 | 512 | 58,078 | 9,613 | 16.5519 |
| 2022-11 | 56,080 | 629 | 55,451 | 10,381 | 18.7210 |
| 2022-12 | 56,986 | 3,062 | 53,924 | 14,297 | 26.5132 |
| 2023-01 | 55,321 | 1,094 | 54,227 | 11,436 | 21.0891 |
| 2023-02 | 51,026 | 982 | 50,044 | 9,242 | 18.4677 |
| 2023-03 | 59,009 | 760 | 58,249 | 13,389 | 22.9858 |
| 2023-04 | 57,289 | 1,027 | 56,262 | 12,786 | 22.7258 |
| 2023-05 | 59,447 | 327 | 59,120 | 10,912 | 18.4574 |
| 2023-06 | 58,668 | 1,217 | 57,451 | 15,437 | 26.8699 |
| 2023-07 | 61,270 | 1,494 | 59,776 | 17,322 | 28.9782 |
| 2023-08 | 61,454 | 908 | 60,546 | 13,432 | 22.1848 |

## Profil kolom

Tipe teramati adalah hasil inferensi pandas per chunk. Missing mengikuti parser CSV pandas; string kosong dan penanda NA bawaan dihitung missing.

| Kolom | Tipe teramati | Missing | Missing (%) | Minimum | Maksimum | Mean | Std sampel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FL_DATE | object | 0 | 0.0000 | — | — | — | — |
| AIRLINE | object | 0 | 0.0000 | — | — | — | — |
| AIRLINE_DOT | object | 0 | 0.0000 | — | — | — | — |
| AIRLINE_CODE | object | 0 | 0.0000 | — | — | — | — |
| DOT_CODE | int64 | 0 | 0.0000 | 19,393.0000 | 20,452.0000 | 19,976.2941 | 377.2846 |
| FL_NUMBER | int64 | 0 | 0.0000 | 1.0000 | 9,562.0000 | 2,511.5355 | 1,747.2580 |
| ORIGIN | object | 0 | 0.0000 | — | — | — | — |
| ORIGIN_CITY | object | 0 | 0.0000 | — | — | — | — |
| DEST | object | 0 | 0.0000 | — | — | — | — |
| DEST_CITY | object | 0 | 0.0000 | — | — | — | — |
| CRS_DEP_TIME | int64 | 0 | 0.0000 | 1.0000 | 2,359.0000 | 1,327.0620 | 485.8789 |
| DEP_TIME | float64 | 77,615 | 2.5872 | 1.0000 | 2,400.0000 | 1,329.7759 | 499.3101 |
| DEP_DELAY | float64 | 77,644 | 2.5881 | -90.0000 | 2,966.0000 | 10.1233 | 49.2518 |
| TAXI_OUT | float64 | 78,806 | 2.6269 | 1.0000 | 184.0000 | 16.6430 | 9.1929 |
| WHEELS_OFF | float64 | 78,806 | 2.6269 | 1.0000 | 2,400.0000 | 1,352.3610 | 500.8727 |
| WHEELS_ON | float64 | 79,944 | 2.6648 | 1.0000 | 2,400.0000 | 1,462.4996 | 527.2368 |
| TAXI_IN | float64 | 79,944 | 2.6648 | 1.0000 | 249.0000 | 7.6790 | 6.2696 |
| CRS_ARR_TIME | int64 | 0 | 0.0000 | 1.0000 | 2,400.0000 | 1,490.5607 | 511.5476 |
| ARR_TIME | float64 | 79,942 | 2.6647 | 1.0000 | 2,400.0000 | 1,466.5112 | 531.8383 |
| ARR_DELAY | float64 | 86,198 | 2.8733 | -96.0000 | 2,934.0000 | 4.2609 | 51.1748 |
| CANCELLED | float64 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0264 | 0.1603 |
| CANCELLATION_CODE | object | 2,920,860 | 97.3620 | — | — | — | — |
| DIVERTED | float64 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0024 | 0.0484 |
| CRS_ELAPSED_TIME | float64 | 14 | 0.0005 | 1.0000 | 705.0000 | 142.2758 | 71.5567 |
| ELAPSED_TIME | float64 | 86,198 | 2.8733 | 15.0000 | 739.0000 | 136.6205 | 71.6758 |
| AIR_TIME | float64 | 86,198 | 2.8733 | 8.0000 | 692.0000 | 112.3108 | 69.7548 |
| DISTANCE | float64 | 0 | 0.0000 | 29.0000 | 5,812.0000 | 809.3616 | 587.8939 |
| DELAY_DUE_CARRIER | float64 | 2,466,137 | 82.2046 | 0.0000 | 2,934.0000 | 24.7591 | 71.7718 |
| DELAY_DUE_WEATHER | float64 | 2,466,137 | 82.2046 | 0.0000 | 1,653.0000 | 3.9853 | 32.4108 |
| DELAY_DUE_NAS | float64 | 2,466,137 | 82.2046 | 0.0000 | 1,741.0000 | 13.1647 | 33.1611 |
| DELAY_DUE_SECURITY | float64 | 2,466,137 | 82.2046 | 0.0000 | 1,185.0000 | 0.1459 | 3.5821 |
| DELAY_DUE_LATE_AIRCRAFT | float64 | 2,466,137 | 82.2046 | 0.0000 | 2,557.0000 | 25.4713 | 55.7669 |

Mean/std untuk kode identitas, indikator, dan HHMM bukan ukuran durasi operasional.

## Distribusi departure delay

Kuantil eksak pada non-cancelled dengan DEP_DELAY tersedia; nilai negatif tetap dipertahankan.

| Kuantil | Menit |
| --- | --- |
| 0.0 | -90.0000 |
| 0.01 | -14.0000 |
| 0.25 | -6.0000 |
| 0.5 | -2.0000 |
| 0.75 | 6.0000 |
| 0.9 | 36.0000 |
| 0.95 | 72.0000 |
| 0.99 | 191.0000 |
| 0.999 | 652.0000 |
| 1.0 | 2,966.0000 |

### Arti kuantil eksak

**Kuantil** adalah nilai batas pada posisi tertentu setelah seluruh nilai `DEP_DELAY` yang memenuhi syarat diurutkan dari yang terkecil hingga terbesar. Kuantil 0,95 sama dengan persentil ke-95 (P95): secara umum, sekitar 95% nilai berada pada atau di bawah batas tersebut. Persentase aktual dapat berbeda karena banyak penerbangan memiliki nilai delay yang sama.

**Eksak (exact)** berarti perhitungan menggunakan seluruh nilai `DEP_DELAY` dari penerbangan non-cancelled yang nilainya tersedia dalam dataset ini, tanpa mengambil sampel tambahan atau memakai algoritme kuantil perkiraan. Dataset asal tetap merupakan sampel penerbangan; kata “eksak” tidak berarti hasilnya mewakili seluruh penerbangan di dunia nyata.

Cara membaca kuantil pada tabel:

- **0,0:** nilai minimum, yaitu keberangkatan paling awal relatif terhadap jadwal.
- **0,25 (P25):** batas seperempat bagian bawah data.
- **0,5 (P50/median):** nilai tengah data. Jika nilainya -2 menit, titik tengah distribusi berada pada keberangkatan 2 menit lebih awal dari jadwal.
- **0,95 (P95):** jika nilainya 72 menit, sekitar 95% penerbangan dalam kelompok ini memiliki `DEP_DELAY` paling tinggi 72 menit; sisanya berada di ekor keterlambatan yang lebih panjang.
- **0,99 (P99):** jika nilainya 191 menit, batas untuk sekitar 99% data adalah 191 menit.
- **1,0:** nilai maksimum, yaitu keterlambatan keberangkatan terbesar yang tercatat.

Nilai negatif berarti berangkat lebih awal, nol berarti sesuai jadwal, dan positif berarti terlambat. Perhitungan menggunakan interpolasi linear bila posisi kuantil berada di antara dua nilai yang berurutan, sehingga hasil kuantil dapat berupa pecahan dan tidak harus sama dengan nilai satu penerbangan tertentu. Tampilan tabel dibulatkan hingga empat angka desimal.

## Status dan ketersediaan label

| Kelompok | Baris | DEP_DELAY missing | ARR_DELAY missing |
| --- | --- | --- | --- |
| cancelled | 79,140 | 77,644 | 79,140 |
| noncancelled_diverted | 7,056 | 0 | 7,056 |
| noncancelled_not_diverted | 2,913,804 | 0 | 2 |
| unknown_status | 0 | 0 | 0 |

## Pemeriksaan kualitas

| Pemeriksaan | Jumlah |
| --- | --- |
| invalid_dates | 0 |
| date_order_inversions | 1,499,403 |
| cancelled_with_dep_delay | 1,496 |
| noncancelled_missing_dep_delay | 0 |
| negative_dep_delay | 1,787,569 |
| invalid_CANCELLED | 0 |
| invalid_DIVERTED | 0 |
| CRS_DEP_TIME_2400 | 0 |
| CRS_DEP_TIME_invalid_hhmm_except_2400 | 0 |
| DEP_TIME_2400 | 242 |
| DEP_TIME_invalid_hhmm_except_2400 | 0 |
| WHEELS_OFF_2400 | 401 |
| WHEELS_OFF_invalid_hhmm_except_2400 | 0 |
| WHEELS_ON_2400 | 1,109 |
| WHEELS_ON_invalid_hhmm_except_2400 | 0 |
| CRS_ARR_TIME_2400 | 14 |
| CRS_ARR_TIME_invalid_hhmm_except_2400 | 0 |
| ARR_TIME_2400 | 1,416 |
| ARR_TIME_invalid_hhmm_except_2400 | 0 |
| DISTANCE_nonpositive | 0 |
| CRS_ELAPSED_TIME_nonpositive | 0 |
| ELAPSED_TIME_nonpositive | 0 |
| AIR_TIME_nonpositive | 0 |
| full_row_hash_duplicate_candidates | 0 |
| flight_key_hash_duplicate_candidates | 0 |

Duplikat adalah kandidat berdasarkan hash 64-bit, bukan verifikasi kesamaan seluruh baris. Kunci: FL_DATE, AIRLINE_CODE, FL_NUMBER, ORIGIN, DEST, CRS_DEP_TIME.

Tanggal mundur dihitung antara baris bertetangga. Jam 2400 dicatat terpisah dari HHMM tidak valid; normalisasi dilakukan pada tahap cleaning.

## Kategori dan rute teratas

### AIRLINE_CODE

| Nilai | Baris |
| --- | --- |
| WN | 576,470 |
| DL | 395,239 |
| AA | 383,106 |
| OO | 343,737 |
| UA | 254,504 |
| YX | 143,107 |
| MQ | 121,256 |
| B6 | 112,844 |
| 9E | 112,463 |
| OH | 107,050 |

### ORIGIN

| Nilai | Baris |
| --- | --- |
| ATL | 153,556 |
| DFW | 130,334 |
| ORD | 122,296 |
| DEN | 119,919 |
| CLT | 94,304 |
| LAX | 85,872 |
| PHX | 74,815 |
| LAS | 73,470 |
| SEA | 70,906 |
| MCO | 63,883 |

### DEST

| Nilai | Baris |
| --- | --- |
| ATL | 153,569 |
| DFW | 129,770 |
| ORD | 123,334 |
| DEN | 119,592 |
| CLT | 95,413 |
| LAX | 85,621 |
| PHX | 75,605 |
| LAS | 73,462 |
| SEA | 70,832 |
| MCO | 63,818 |

### CANCELLED

| Nilai | Baris |
| --- | --- |
| 0.0 | 2,920,860 |
| 1.0 | 79,140 |

### DIVERTED

| Nilai | Baris |
| --- | --- |
| 0.0 | 2,992,944 |
| 1.0 | 7,056 |

### CANCELLATION_CODE

| Nilai | Baris |
| --- | --- |
| B | 28,772 |
| D | 24,417 |
| A | 19,476 |
| C | 6,475 |

### ROUTE

| Nilai | Baris |
| --- | --- |
| SFO -> LAX | 5,358 |
| LAX -> SFO | 5,212 |
| LGA -> ORD | 4,735 |
| OGG -> HNL | 4,656 |
| ORD -> LGA | 4,634 |
| LAX -> LAS | 4,627 |
| HNL -> OGG | 4,554 |
| LAS -> LAX | 4,550 |
| LAX -> JFK | 4,403 |
| JFK -> LAX | 4,305 |

## Tinjauan kamus fitur

| Kolom | Peran usulan | Catatan |
| --- | --- | --- |
| FL_DATE | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Format yang diperiksa YYYY-MM-DD; deskripsi kamus perlu diselaraskan |
| AIRLINE | Label/identitas untuk pelaporan; periksa redundansi | Belum tersedia dalam kamus sumber; Usulan: nama maskapai; verifikasi sumber |
| AIRLINE_DOT | Label/identitas untuk pelaporan; periksa redundansi | Belum tersedia dalam kamus sumber; Usulan: label gabungan nama/kode maskapai; bukan nomor DOT; verifikasi sumber |
| AIRLINE_CODE | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DOT_CODE | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Pengenal kategorikal, bukan besaran kontinu |
| FL_NUMBER | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Pengenal kategorikal, bukan besaran kontinu |
| ORIGIN | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| ORIGIN_CITY | Label/identitas untuk pelaporan; periksa redundansi | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DEST | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DEST_CITY | Label/identitas untuk pelaporan; periksa redundansi | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| CRS_DEP_TIME | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| DEP_TIME | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| DEP_DELAY | Target E1/E2; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| TAXI_OUT | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| WHEELS_OFF | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| WHEELS_ON | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| TAXI_IN | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| CRS_ARR_TIME | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| ARR_TIME | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | HHMM waktu lokal; 2400 memerlukan aturan normalisasi |
| ARR_DELAY | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| CANCELLED | Target E3; filter populasi E1/E2; bukan fitur | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| CANCELLATION_CODE | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Arti kode perlu verifikasi provenance; null dapat bersifat struktural |
| DIVERTED | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| CRS_ELAPSED_TIME | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| ELAPSED_TIME | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| AIR_TIME | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DISTANCE | Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi) | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DELAY_DUE_CARRIER | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DELAY_DUE_WEATHER | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DELAY_DUE_NAS | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DELAY_DUE_SECURITY | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |
| DELAY_DUE_LATE_AIRCRAFT | Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan | Peran adalah usulan; konfirmasi pada rancangan eksperimen |

Kolom kamus yang tidak ditemukan di CSV: tidak ada.

## Keputusan yang diperlukan sebelum cleaning/modeling

- Tetapkan horizon prediksi dan asumsi ketersediaan jadwal; dataset tidak menyimpan riwayat revisi jadwal.
- E1/E2 menggunakan CANCELLED == 0 dan DEP_DELAY tersedia. Jangan mengisi target missing dengan nol; diverted dengan label keberangkatan tersedia dapat tetap dianalisis.
- E3 menggunakan label CANCELLED yang valid. Status dan waktu aktual serta DELAY_DUE_* tidak boleh menjadi prediktor sebelum keberangkatan.
- Tentukan target E2: selisih bertanda atau durasi max(DEP_DELAY, 0). Nilai ekstrem perlu ditinjau sebelum dibuang.
- Pisahkan data secara waktu. Statistik laporan ini untuk pemahaman data; parameter preprocessing dipelajari dari train dan test disimpan untuk evaluasi akhir.
- Verifikasi sumber, metode sampling, dan pemetaan kode pembatalan. Laporan lokal tidak membuktikan representativitas atau sebab-akibat.

## Metode dan keterbatasan

Chunk: 100,000 baris. Waktu pemindaian dan peringkasan: 45.8 detik.

CSV dibaca bertahap. Hash duplikasi dan nilai DEP_DELAY eligible disimpan sepanjang pemindaian, sehingga sebagian penggunaan RAM tetap bertambah dengan jumlah baris. Identitas input memakai path, ukuran, dan waktu modifikasi; bukan checksum kriptografis.

Kirim laporan Markdown ini untuk analisis lanjutan. JSON menyimpan rincian numerik dan CSV menyimpan tinjauan kamus fitur.
