"""E1: cleaning, feature engineering, encoding, imputasi, dan scaling.

Jalankan: python e1-data-preparation.py
Raw dibaca dari root proyek. Model klasifikasi belum dilatih.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

try:
    import joblib
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import sparse
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    raise SystemExit("Dependensi belum tersedia. Jalankan: python -m pip install -r requirements.txt")


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "flights_sample_3m.csv"
CHUNK_SIZE = 100_000
INPUT_COLUMNS = [
    "FL_DATE", "AIRLINE_CODE", "ORIGIN", "DEST", "CRS_DEP_TIME", "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME", "DISTANCE", "CANCELLED", "DEP_DELAY",
]
FEATURES = [
    "AIRLINE_CODE", "ORIGIN", "DEST", "MONTH", "DAY_OF_WEEK",
    "SCHEDULED_DEP_MINUTES", "SCHEDULED_ARR_MINUTES", "CRS_ELAPSED_TIME", "DISTANCE",
]
TARGET = "DELAYED_15"
METADATA = ["SOURCE_ROW_NUMBER", "FL_DATE"]
OUTPUT_COLUMNS = METADATA + FEATURES + [TARGET]
SPLITS = {
    "train": ("2019-01-01", "2021-12-31"),
    "validation": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2023-08-31"),
}
REASONS = {
    "FL_DATE": "Metadata split waktu; diturunkan menjadi MONTH dan DAY_OF_WEEK.",
    "AIRLINE_CODE": "Identitas maskapai; gunakan satu representasi identitas untuk baseline.",
    "ORIGIN": "Bandara asal yang tercantum pada jadwal.",
    "DEST": "Bandara tujuan yang tercantum pada jadwal.",
    "CRS_DEP_TIME": "Jadwal keberangkatan HHMM; diubah menjadi menit sejak tengah malam lokal.",
    "CRS_ARR_TIME": "Jadwal kedatangan HHMM; diubah menjadi menit sejak tengah malam lokal.",
    "CRS_ELAPSED_TIME": "Durasi perjalanan terjadwal dalam menit.",
    "DISTANCE": "Jarak rute dalam mil.",
    "CANCELLED": "Filter populasi: hanya nilai 0 yang digunakan; bukan fitur model.",
    "DEP_DELAY": "Pembentuk target DELAYED_15 dengan ambang >=15 menit; bukan fitur model.",
}


def numeric(series):
    result = pd.to_numeric(series, errors="coerce")
    return result.where(np.isfinite(result))


def clean_chunk(frame, start_row):
    """Aturan tetap; tidak mempelajari statistik dari train/validation/test."""
    stats = Counter(input_rows=len(frame))
    dates = pd.to_datetime(frame.FL_DATE, format="%Y-%m-%d", errors="coerce")
    cancelled, delay = numeric(frame.CANCELLED), numeric(frame.DEP_DELAY)
    # Pengecualian saling eksklusif agar jumlah baris dapat direkonsiliasi.
    valid = pd.Series(True, index=frame.index)
    for reason, condition in [
        ("excluded_invalid_cancelled", ~cancelled.isin([0, 1])),
        ("excluded_cancelled", cancelled.eq(1)),
        ("excluded_missing_or_invalid_target", delay.isna()),
        ("excluded_invalid_date", dates.isna()),
        ("excluded_outside_split_period", ~dates.between("2019-01-01", "2023-08-31")),
    ]:
        rejected = valid & condition
        stats[reason] += int(rejected.sum())
        valid &= ~rejected
    kept = frame.loc[valid]
    dates = dates.loc[valid]
    result = pd.DataFrame(index=kept.index)
    result["SOURCE_ROW_NUMBER"] = pd.Series(np.arange(start_row, start_row + len(frame)), index=frame.index).loc[valid]
    result["FL_DATE"] = dates.dt.strftime("%Y-%m-%d")
    for column in ("AIRLINE_CODE", "ORIGIN", "DEST"):
        original = kept[column]
        cleaned = original.str.strip().str.upper().replace("", pd.NA)
        stats[column + "_normalized"] += int((original.fillna("") != cleaned.fillna("")).sum())
        result[column] = cleaned
    result["MONTH"] = dates.dt.month.astype("int8")
    result["DAY_OF_WEEK"] = dates.dt.dayofweek.astype("int8")
    for source, destination in [("CRS_DEP_TIME", "SCHEDULED_DEP_MINUTES"), ("CRS_ARR_TIME", "SCHEDULED_ARR_MINUTES")]:
        time = numeric(kept[source])
        valid_time = (time.between(0, 2359) & time.mod(100).lt(60) & time.mod(1).eq(0)).fillna(False)
        midnight = time.eq(2400).fillna(False)
        stats[source + "_2400_normalized"] += int(midnight.sum())
        stats[source + "_invalid_to_missing"] += int((kept[source].notna() & ~(valid_time | midnight)).sum())
        result[destination] = ((time // 100) * 60 + time % 100).where(valid_time).mask(midnight, 0)
    for column in ("CRS_ELAPSED_TIME", "DISTANCE"):
        value = numeric(kept[column])
        acceptable = value.gt(0).fillna(False)
        stats[column + "_invalid_to_missing"] += int((kept[column].notna() & ~acceptable).sum())
        result[column] = value.where(acceptable)
    stats["scheduled_duration_under_10_retained"] += int(result.CRS_ELAPSED_TIME.lt(10).sum())
    result[TARGET] = delay.loc[valid].ge(15).astype("int8")
    stats["negative_delay_retained"] += int(delay.loc[valid].lt(0).sum())
    stats["delay_over_24h_retained"] += int(delay.loc[valid].gt(1440).sum())
    stats["retained_rows"] += len(result)
    return result[OUTPUT_COLUMNS], stats


def feature_decisions(columns):
    records = []
    for column in columns:
        if column in REASONS:
            role = "pendukung" if column in ("CANCELLED", "DEP_DELAY") else "dipakai"
            reason = REASONS[column]
        elif column in ("AIRLINE", "AIRLINE_DOT", "DOT_CODE", "ORIGIN_CITY", "DEST_CITY"):
            role, reason = "tidak dipakai", "Label/identitas tambahan; baseline menggunakan kode maskapai dan bandara."
        elif column == "FL_NUMBER":
            role, reason = "ditunda", "Nomor penerbangan dapat diuji setelah baseline; jangan diperlakukan sebagai besaran kontinu."
        else:
            role, reason = "tidak dipakai", "Informasi setelah kejadian atau belum disetujui untuk fitur pra-keberangkatan."
        records.append({"source_column": column, "decision": role, "reason": reason})
    return records


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
    ])



CATEGORICAL = ["AIRLINE_CODE", "ORIGIN", "DEST"]
CONTINUOUS = ["CRS_ELAPSED_TIME", "DISTANCE"]
CYCLES = {
    "MONTH": (12, 1), "DAY_OF_WEEK": (7, 0),
    "SCHEDULED_DEP_MINUTES": (1440, 0), "SCHEDULED_ARR_MINUTES": (1440, 0),
}
CYCLIC_FEATURES = [f"{column}_{suffix}" for column in CYCLES for suffix in ("SIN", "COS")]
MISSING_FLAGS = [f"{column}_MISSING" for column in
                 ["SCHEDULED_DEP_MINUTES", "SCHEDULED_ARR_MINUTES"] + CONTINUOUS]
MODEL_FEATURES = CATEGORICAL + CONTINUOUS + CYCLIC_FEATURES + MISSING_FLAGS
PREPARED_COLUMNS = OUTPUT_COLUMNS + CYCLIC_FEATURES + MISSING_FLAGS


def engineer(frame):
    """Transformasi tetap; tidak mempelajari parameter dari data mana pun."""
    frame = frame.copy()
    for column in ["SCHEDULED_DEP_MINUTES", "SCHEDULED_ARR_MINUTES"] + CONTINUOUS:
        frame[column + "_MISSING"] = frame[column].isna().astype("int8")
    for column, (period, offset) in CYCLES.items():
        phase = 2 * np.pi * (frame[column].astype(float) - offset) / period
        # (0, 0) menandai waktu tidak diketahui; flag missing tetap disertakan.
        frame[column + "_SIN"] = np.sin(phase).fillna(0.0)
        frame[column + "_COS"] = np.cos(phase).fillna(0.0)
    return frame[PREPARED_COLUMNS]


def make_preprocessor():
    return ColumnTransformer([
        ("category", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]), CATEGORICAL),
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), CONTINUOUS),
        ("cyclic", "passthrough", CYCLIC_FEATURES),
        ("missing", "passthrough", MISSING_FLAGS),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)


def model_input(frame):
    result = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL:
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def head_tail_preview(matrix, labels, feature_names, metadata, positions):
    """Hanya mendensifikasi sepuluh baris; posisi mengacu pada urutan matriks."""
    preview = pd.DataFrame(matrix[positions].toarray(), columns=feature_names)
    preview.insert(0, "FL_DATE", metadata.FL_DATE.to_numpy())
    preview.insert(0, "SOURCE_ROW_NUMBER", metadata.SOURCE_ROW_NUMBER.to_numpy())
    preview.insert(0, "DATASET_ROW", np.asarray(positions) + 1)
    size = len(positions) // 2
    preview.insert(0, "PREVIEW_PART", ["head"] * size + ["tail"] * size)
    preview[TARGET] = labels[positions]
    return preview


def preview_positions(rows, size=5):
    size = min(rows, size)
    return list(range(size)) + list(range(rows - size, rows))


def render_head_tail(previews):
    lines = [
        "## Head dan tail dataset setelah preprocessing dan FE", "",
        "Head adalah 5 baris pertama dan tail adalah 5 baris terakhir dari setiap subset, mengikuti urutan dataset hasil pengolahan. Nilai fitur diambil langsung dari matriks X setelah encoding, imputasi, scaling, dan FE; label diambil dari y. Jika subset berisi kurang dari 5 baris, semua baris ditampilkan dan head/tail dapat berulang.", "",
        "DATASET_ROW adalah posisi baris dalam subset (mulai 1). SOURCE_ROW_NUMBER dan FL_DATE hanya metadata pelacakan; DELAYED_15 adalah target, bukan fitur. Kolom one-hot diringkas sebagai daftar nama kolom bernilai 1; kolom one-hot lainnya bernilai 0. Kategori baru dapat menghasilkan seluruh nol dalam kelompoknya.", "",
        "Untuk keterbacaan, fitur dibagi menjadi tabel identitas/encoding, scaling/indikator missing, dan fitur siklik. Prefix category__, continuous__, cyclic__, dan missing__ disingkat pada tampilan. Nilai desimal dibulatkan hingga 6 tempat hanya dalam laporan. Seluruh kolom dan nilai tanpa pembulatan tampilan tersedia di outputs/{train,validation,test}_head_tail.csv.", "",
    ]
    for name, preview in previews.items():
        lines += [f"### {name.capitalize()}", ""]
        one_hot = [c for c in preview.columns if c.startswith("category__")]
        for part in ("head", "tail"):
            block = preview.loc[preview.PREVIEW_PART.eq(part)]
            lines += [f"#### {part.capitalize()} — {len(block)} baris", "",
                      "Identitas, target, dan hasil encoding kategori:", "",
                      markdown_table(["DATASET_ROW", "SOURCE_ROW_NUMBER", "FL_DATE", TARGET, "One-hot aktif (=1)"], [
                          (int(row.DATASET_ROW), int(row.SOURCE_ROW_NUMBER), row.FL_DATE, int(row[TARGET]),
                           "; ".join(c.removeprefix("category__") for c in one_hot if row[c] == 1) or "Tidak ada")
                          for _, row in block.iterrows()
                      ]), ""]
            for label, prefixes in (("Hasil scaling dan indikator missing", ("continuous__", "missing__")),
                                    ("Fitur siklik hasil FE", ("cyclic__",))):
                columns = [c for c in preview.columns if c.startswith(prefixes)]
                lines += [label + ":", "", markdown_table(
                    ["DATASET_ROW"] + [c.split("__", 1)[1] for c in columns],
                    [(int(row.DATASET_ROW), *(f"{float(row[c]):.6f}" for c in columns)) for _, row in block.iterrows()]
                ), ""]
    return "\n".join(lines)


def render_report(summary, decisions, previews):
    counts, splits = summary["counts"], summary["splits"]
    fitting = summary["fitted_on_train"]
    return "\n".join([
        "# E1 — Preprocessing dan feature engineering", "",
        f"Dibuat (UTC): {summary['created_utc']}", "",
        "Cleaning, pembentukan fitur, pembagian waktu, encoding, imputasi, dan scaling selesai. Encoder, median imputer, dan scaler dipelajari hanya dari train. Belum ada model klasifikasi yang dilatih.", "",
        "Target DELAYED_15 = 1 jika DEP_DELAY >= 15 menit; selain itu 0. Populasi memakai CANCELLED == 0 dengan DEP_DELAY valid. Diverted tetap masuk bila memenuhi syarat. Probabilitas model nantinya bersyarat pada populasi non-cancelled.", "",
        "Skenario prediksi memakai jadwal sebelum keberangkatan. CSV tidak menyimpan riwayat revisi jadwal, sehingga ketersediaan jadwal pada horizon tertentu masih merupakan asumsi.", "",
        "## Rekonsiliasi baris", "",
        markdown_table(["Pemeriksaan", "Baris"], [(k, f"{v:,}") for k, v in counts.items()
            if k in ("input_rows", "retained_rows") or k.startswith("excluded_")]), "",
        "Pengecualian saling eksklusif. Jumlah retained_rows + seluruh excluded_* sama dengan input_rows. Dataset raw di root hanya dibaca.", "",
        "## Pemilihan kolom", "",
        markdown_table(["Kolom sumber", "Keputusan", "Alasan"],
            [(d["source_column"], d["decision"], d["reason"]) for d in decisions]), "",
        "## Cleaning", "",
        markdown_table(["Pemeriksaan", "Jumlah"], [(k, f"{v:,}") for k, v in counts.items()
            if k not in ("input_rows", "retained_rows") and not k.startswith("excluded_")]), "",
        "Kategori di-trim dan diubah menjadi huruf kapital. Jam tidak valid dan durasi/jarak tidak positif menjadi missing. Delay negatif dan ekstrem dipertahankan; durasi terjadwal positif di bawah 10 menit ditandai untuk tinjauan. Missing target tidak diimputasi. Skrip tidak menghapus duplikasi; audit sebelumnya tidak menemukan kandidat duplikat. Audit ulang diperlukan bila sumber berubah.", "",
        "## Feature engineering", "",
        "- MONTH: Januari=1 hingga Desember=12; DAY_OF_WEEK: Senin=0 hingga Minggu=6.",
        "- SCHEDULED_*_MINUTES: menit sejak tengah malam lokal; 1530 menjadi 930 dan 2400 menjadi 0. Tidak dibuat perkiraan tanggal kedatangan atau konversi zona waktu.",
        "- Kalender dan waktu diubah menjadi pasangan sin/cos: sudut = 2*pi*(nilai-offset)/periode. Periode bulan=12, hari=7, waktu=1440; offset bulan=1, lainnya=0. Representasi ini menjaga kedekatan akhir dan awal siklus.",
        "- Waktu yang missing memakai pasangan (0,0) dengan indikator missing=1. Durasi/jarak juga mendapat indikator missing. Aturan ini tetap dan tidak memerlukan fit.",
        "- Sin/cos sudah berada pada rentang [-1,1]; indikator missing dan hasil one-hot adalah 0/1, sehingga ketiganya tidak distandardisasi.", "",
        "## Transformasi yang dipelajari dari train", "",
        "Maskapai dan bandara: missing diisi token __MISSING__, lalu one-hot encoding. Kategori baru pada validation/test menghasilkan semua nol pada kelompok kategori terkait. Jumlah kategori baru dilaporkan di bawah.", "",
        "Durasi terjadwal dan jarak: missing diisi median train, kemudian StandardScaler menggunakan mean dan skala hasil imputasi train. Validation/test tidak ikut menentukan parameter. Kolom numerik yang seluruhnya kosong pada train menghentikan proses.", "",
        markdown_table(["Fitur", "Median train", "Mean setelah imputasi", "Skala StandardScaler"],
            [(c, fitting["numeric"][c]["median"], fitting["numeric"][c]["mean"], fitting["numeric"][c]["scale"]) for c in CONTINUOUS]), "",
        "## Pembagian waktu dan matriks model", "",
        markdown_table(["Subset", "Periode aktual", "Baris", "Label 0", "Label 1", "Delay rate (%)", "Fitur akhir"],
            [(name, f"{s['date_min']} s.d. {s['date_max']}", f"{s['rows']:,}", f"{s['label_0']:,}",
              f"{s['label_1']:,}", f"{s['delay_rate_pct']:.4f}", s["matrix_shape"][1])
             for name, s in splits.items()]), "",
        "Batas tetap: train 2019–2021; validation 2022; test Januari–Agustus 2023. Tiap subset diurutkan berdasarkan tanggal, waktu keberangkatan lokal, lalu nomor record sumber. Urutan lokal lintas bandara bukan urutan timestamp UTC.", "",
        "## Missing sebelum imputasi", "",
        markdown_table(["Fitur dasar", "Train", "Validation", "Test"],
            [(c, *(splits[name]["missing_before"][c] for name in SPLITS)) for c in FEATURES]), "",
        "## Kategori baru di luar train", "",
        markdown_table(["Kolom", "Kategori hasil fit train", "Baris baru validation", "Baris baru test"],
            [(c, fitting["category_counts"][c], splits["validation"]["unknown_category_rows"][c],
              splits["test"]["unknown_category_rows"][c]) for c in CATEGORICAL]), "",
        "Kategori baru berarti nilai (termasuk token missing jika belum pernah ada saat train) yang tidak terdapat pada encoder train. Jumlah baris dilaporkan per kolom dan dapat saling tumpang tindih.", "",
        "## Validasi hasil", "",
        "- Jumlah baris input, pengecualian, dan hasil telah direkonsiliasi.",
        "- Label hanya 0/1; SOURCE_ROW_NUMBER unik di masing-masing subset.",
        "- Semua matriks memiliki urutan fitur yang sama dan tidak mengandung NaN/infinity.",
        "- Metadata, CANCELLED, dan DEP_DELAY tidak menjadi masukan model. Tidak ada resampling atau fitur historis pada versi ini.", "",
        "## Berkas keluaran", "",
        "- datasets/prepared/{train,validation,test}.csv: data setelah cleaning/FE, sebelum transformasi yang dipelajari; untuk audit dan model dengan kebutuhan preprocessing lain.",
        "- datasets/model-ready/X_{train,validation,test}.npz: matriks fitur sparse CSR float32 setelah seluruh preprocessing.",
        "- datasets/model-ready/y_{train,validation,test}.npy: label int8. Baris X dan y sejajar dengan CSV prepared subset yang sama.",
        "- artifacts/preprocessor.joblib: transformer hasil fit train, menerima fitur engineered sesuai feature_contract.json.",
        "- outputs/feature_names.csv: nama dan urutan kolom matriks X; feature_contract.json: definisi fitur dan target.",
        "- outputs/train_transformed_preview.csv: sepuluh baris pertama hasil transformasi train untuk inspeksi.",
        "- outputs/preprocessing_summary.json dan e1-data-preparation-report.md: ringkasan lengkap.", "",
        "Matriks menggunakan format sparse agar one-hot encoding tidak menghasilkan CSV padat berukuran besar. Nilai nol tersirat di dalam sparse bukan missing. X dibaca dengan scipy.sparse.load_npz; y dengan numpy.load.", "",
        "## Sumber dan pemakaian ulang", "",
        f"Raw: {summary['input']['path']}",
        f"SHA-256: {summary['input']['sha256']}",
        f"Waktu proses: {summary['elapsed_seconds']:.1f} detik; chunk: {summary['chunk_size']:,} baris.", "",
        "Raw tidak disalin. Pembacaan awal bertahap, tetapi pengurutan dan preprocessing memuat satu subset ke RAM pada satu waktu. File sementara serta hasil lama dapat memerlukan ruang disk bersamaan saat menjalankan ulang.", "",
        "Langkah berikutnya adalah training baseline dan evaluasi. Jangan fit ulang preprocessing pada validation/test. Jika nanti menggunakan cross-validation di dalam train, preprocessing harus di-fit ulang di setiap fold dari CSV prepared, bukan memakai X yang telah di-fit pada seluruh train.", "",
        render_head_tail(previews),
    ])


def run(source=SOURCE, base=ROOT, chunk_size=CHUNK_SIZE):
    started = perf_counter()
    source, base = Path(source).resolve(), Path(base).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset raw tidak ditemukan: {source}")
    base.mkdir(parents=True, exist_ok=True)
    initial = source.stat()
    print(f"Membaca raw langsung dari: {source}", flush=True)
    print("Menghitung identitas sumber...", flush=True)
    digest = fingerprint(source)
    columns = list(pd.read_csv(source, nrows=0).columns)
    if set(INPUT_COLUMNS) - set(columns):
        raise ValueError("Kolom wajib tidak tersedia: " + ", ".join(sorted(set(INPUT_COLUMNS) - set(columns))))
    decisions = feature_decisions(columns)
    counts = Counter()
    preprocessor = make_preprocessor()
    splits = {}
    previews = {}
    with TemporaryDirectory(prefix=".e1-work-", dir=base) as temporary:
        stage = Path(temporary)
        prepared, ready = stage / "datasets/prepared", stage / "datasets/model-ready"
        outputs, artifacts = stage / "outputs", stage / "artifacts"
        for directory in (prepared, ready, outputs, artifacts):
            directory.mkdir(parents=True)
        for name in SPLITS:
            pd.DataFrame(columns=PREPARED_COLUMNS).to_csv(prepared / f"{name}.csv", index=False)
        for frame in pd.read_csv(source, usecols=INPUT_COLUMNS, dtype="string", chunksize=chunk_size):
            cleaned, stats = clean_chunk(frame, counts["input_rows"] + 1)
            counts.update(stats)
            engineered = engineer(cleaned)
            for name, (first, last) in SPLITS.items():
                part = engineered.loc[engineered.FL_DATE.between(first, last)]
                part.to_csv(prepared / f"{name}.csv", mode="a", index=False, header=False)
            print(f"  Dibaca {counts['input_rows']:,}; dipertahankan {counts['retained_rows']:,}", flush=True)
        if not counts["input_rows"]:
            raise ValueError("Dataset kosong.")
        del frame, cleaned, engineered, part
        for name in SPLITS:
            print(f"Memproses {name}: urutkan, {'fit dan transform' if name == 'train' else 'transform saja'}...", flush=True)
            path = prepared / f"{name}.csv"
            frame = pd.read_csv(path, dtype={c: "string" for c in ["FL_DATE"] + CATEGORICAL})
            if frame.empty:
                raise ValueError(f"Subset {name} kosong; hasil lama tidak diperbarui.")
            frame = frame.sort_values(["FL_DATE", "SCHEDULED_DEP_MINUTES", "SOURCE_ROW_NUMBER"], kind="stable", na_position="last")
            if frame.SOURCE_ROW_NUMBER.duplicated().any() or not frame[TARGET].isin([0, 1]).all():
                raise ValueError(f"Identitas record/label tidak valid pada {name}.")
            frame.to_csv(path, index=False)
            x = model_input(frame)
            if name == "train":
                if any(x[c].isna().all() for c in CONTINUOUS):
                    raise ValueError("Ada fitur kontinu yang seluruhnya missing pada train; median tidak dapat dipelajari.")
                transformed = preprocessor.fit_transform(x)
                feature_names = preprocessor.get_feature_names_out().tolist()
                pd.DataFrame({"feature_index": range(len(feature_names)), "feature_name": feature_names}).to_csv(outputs / "feature_names.csv", index=False)
                joblib.dump(preprocessor, artifacts / "preprocessor.joblib", compress=3)
                num_pipe = preprocessor.named_transformers_["continuous"]
                encoder = preprocessor.named_transformers_["category"].named_steps["encoder"]
                fitted = {
                    "numeric": {c: {"median": float(num_pipe.named_steps["imputer"].statistics_[i]),
                                    "mean": float(num_pipe.named_steps["scaler"].mean_[i]),
                                    "scale": float(num_pipe.named_steps["scaler"].scale_[i])}
                                for i, c in enumerate(CONTINUOUS)},
                    "category_counts": {c: len(encoder.categories_[i]) for i, c in enumerate(CATEGORICAL)},
                }
            else:
                transformed = preprocessor.transform(x)
            matrix = sparse.csr_matrix(transformed, dtype=np.float32)
            del transformed
            labels = frame[TARGET].to_numpy(dtype=np.int8)
            if matrix.shape != (len(frame), len(feature_names)) or not np.isfinite(matrix.data).all():
                raise ValueError(f"Matriks fitur tidak valid pada {name}.")
            sparse.save_npz(ready / f"X_{name}.npz", matrix, compressed=True)
            np.save(ready / f"y_{name}.npy", labels, allow_pickle=False)
            positions = preview_positions(len(frame))
            previews[name] = head_tail_preview(matrix, labels, feature_names, frame.iloc[positions], positions)
            previews[name].to_csv(outputs / f"{name}_head_tail.csv", index=False)
            if name == "train":
                preview = pd.DataFrame(matrix[:10].toarray(), columns=feature_names)
                preview.insert(0, "SOURCE_ROW_NUMBER", frame.SOURCE_ROW_NUMBER.iloc[:10].to_numpy())
                preview[TARGET] = labels[:10]
                preview.to_csv(outputs / "train_transformed_preview.csv", index=False)
            positives = int(labels.sum())
            unknown = {c: int((~x[c].fillna("__MISSING__").isin(encoder.categories_[i])).sum())
                       for i, c in enumerate(CATEGORICAL)}
            splits[name] = {
                "rows": len(frame), "date_min": str(frame.FL_DATE.min()), "date_max": str(frame.FL_DATE.max()),
                "label_0": len(frame) - positives, "label_1": positives, "delay_rate_pct": 100 * positives / len(frame),
                "missing_before": {c: int(frame[c].isna().sum()) for c in FEATURES},
                "unknown_category_rows": unknown, "matrix_shape": list(matrix.shape),
                "matrix_nnz": int(matrix.nnz), "matrix_dtype": str(matrix.dtype),
            }
            print(f"  {name}: {matrix.shape[0]:,} baris x {matrix.shape[1]:,} fitur; label 1 = {positives:,}", flush=True)
            del frame, x, matrix, labels
        excluded = sum(v for k, v in counts.items() if k.startswith("excluded_"))
        if counts["input_rows"] != counts["retained_rows"] + excluded or sum(s["rows"] for s in splits.values()) != counts["retained_rows"]:
            raise ValueError("Rekonsiliasi baris gagal.")
        final = source.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise ValueError("Sumber berubah selama pemrosesan; hasil tidak diterbitkan.")
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(source), "size_bytes": initial.st_size, "sha256": digest},
            "counts": dict(counts), "splits": splits, "fitted_on_train": fitted,
            "elapsed_seconds": round(perf_counter() - started, 2), "chunk_size": chunk_size,
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__},
        }
        contract = {
            "experiment": "E1", "target": TARGET, "target_rule": "DEP_DELAY >= 15",
            "population": "CANCELLED == 0 and finite DEP_DELAY", "split_boundaries": SPLITS,
            "model_input_features": MODEL_FEATURES, "categorical_features": CATEGORICAL,
            "standardized_features": CONTINUOUS, "cyclic_features": CYCLIC_FEATURES, "missing_flags": MISSING_FLAGS,
            "metadata_not_features": METADATA, "audit_only_base_features": list(CYCLES),
            "output_feature_names": feature_names, "preprocessing_fitted_on": "train",
            "model_trained": False, "row_alignment": "X and y match prepared CSV record order for the same split",
        }
        for filename, content in [("preprocessing_summary.json", summary), ("feature_contract.json", contract)]:
            (outputs / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        pd.DataFrame(decisions).to_csv(outputs / "feature_selection.csv", index=False, encoding="utf-8-sig")
        (outputs / "e1-data-preparation-report.md").write_text(render_report(summary, decisions, previews), encoding="utf-8")
        # Semua penghitungan/validasi selesai sebelum mengganti hasil sebelumnya.
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                destination = base / path.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
    print(f"\nSelesai. Kirim isi laporan:\n{base / 'outputs/e1-data-preparation-report.md'}")
    print("Preprocessing dan FE selesai; matriks siap untuk training. Raw tidak disalin.")
    return summary


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("\nDibatalkan. Raw tidak diubah.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"\nPersiapan E1 gagal: {exc}")
