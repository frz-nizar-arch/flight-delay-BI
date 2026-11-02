# E1: preprocessing dan feature engineering

Dari folder `classification`, cukup jalankan:

```powershell
python e1-data-preparation.py
```

Jika dependensi belum tersedia, jalankan sekali dari folder yang sama:

```powershell
python -m pip install -r requirements.txt
```

Skrip membaca `flights_sample_3m.csv` langsung dari root proyek. Tidak membuat salinan raw. Jalankan dari root dengan `python classification/e1-data-preparation.py` bila diperlukan; lokasi input/output tetap sama.

Alur satu skrip: pilih kolom dan populasi E1 → cleaning → target dan fitur kalender/jadwal → split waktu → fit encoder, imputer, scaler pada train → transform ketiga subset → simpan artefak dan laporan. Model klasifikasi belum dilatih.

Maskapai/bandara memakai one-hot encoding. Bulan, hari dalam minggu, dan waktu memakai sin/cos. Durasi terjadwal/jarak diimputasi menggunakan median train lalu distandardisasi. Missing fitur memiliki indikator; kategori baru di luar train ditangani oleh encoder. Validation/test tidak memengaruhi parameter preprocessing.

Keluaran setelah menjalankan skrip:

```text
classification/
  e1-data-preparation.py
  requirements.txt
  datasets/
    prepared/{train,validation,test}.csv
    model-ready/X_{train,validation,test}.npz
    model-ready/y_{train,validation,test}.npy
  artifacts/preprocessor.joblib
  outputs/
    e1-data-preparation-report.md
    preprocessing_summary.json
    feature_contract.json
    feature_selection.csv
    feature_names.csv
    train_transformed_preview.csv
```

CSV prepared berisi data setelah cleaning/FE untuk audit. Matriks X sudah melewati seluruh preprocessing dan disimpan sebagai sparse CSR float32, agar one-hot tidak menghasilkan CSV padat yang besar. Label y berformat int8. Urutan baris X/y sama dengan CSV prepared subset masing-masing. Nama kolom matriks ada pada `feature_names.csv`. Sepuluh contoh hasil transformasi tersedia di `train_transformed_preview.csv`.

`preprocessor.joblib` menyimpan transformer yang menerima kolom engineered pada `feature_contract.json`. Fungsi `clean_chunk` dan `engineer` di skrip mendokumentasikan aturan deterministik pembentukan fitur; transformer tersimpan tidak membaca CSV raw atau membentuk label sendiri.

Train: 2019–2021; validation: 2022; test: Januari–Agustus 2023. Subset kosong atau fitur kontinu tanpa satu pun nilai valid pada train menghentikan proses. Raw hanya dibaca; rerun memperbarui hasil. Pengurutan dan transformasi memuat satu subset ke RAM, meskipun pembacaan awal memakai chunk. Sediakan ruang disk untuk hasil sementara dan hasil sebelumnya secara bersamaan.

Kirim `outputs/e1-data-preparation-report.md` ke chat setelah selesai. Tidak perlu menjalankan training terlebih dahulu. Jika nantinya menggunakan cross-validation dalam train, fit preprocessing di setiap fold dari CSV prepared untuk menjaga pemisahan data.
