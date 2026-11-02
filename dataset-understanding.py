"""Bagian 6: profil dataset tanpa mengubah CSV sumber.

Jalankan: python dataset-understanding.py
Dependensi: python -m pip install pandas numpy
Semua path dihitung dari lokasi skrip, bukan direktori terminal.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

try:
    import numpy as np
    import pandas as pd
except ImportError:
    raise SystemExit(
        "Dependensi belum tersedia. Jalankan: python -m pip install pandas numpy"
    )


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "flights_sample_3m.csv"
DICTIONARY = ROOT / "flights_sample_3m_features_dictionary.csv"
OUTPUT = ROOT / "outputs" / "01_dataset_understanding"
CHUNK_SIZE = 100_000
TIME_COLUMNS = ["CRS_DEP_TIME", "DEP_TIME", "WHEELS_OFF", "WHEELS_ON", "CRS_ARR_TIME", "ARR_TIME"]
KEY = ["FL_DATE", "AIRLINE_CODE", "FL_NUMBER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
TEXT_COLUMNS = {
    "FL_DATE", "AIRLINE", "AIRLINE_DOT", "AIRLINE_CODE", "ORIGIN",
    "ORIGIN_CITY", "DEST", "DEST_CITY", "CANCELLATION_CODE",
}
NUMERIC_COLUMNS = {
    "DOT_CODE", "FL_NUMBER", "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "TAXI_OUT",
    "WHEELS_OFF", "WHEELS_ON", "TAXI_IN", "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
    "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME", "ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "DELAY_DUE_CARRIER", "DELAY_DUE_WEATHER", "DELAY_DUE_NAS",
    "DELAY_DUE_SECURITY", "DELAY_DUE_LATE_AIRCRAFT",
}
PREDEPARTURE = {
    "FL_DATE", "AIRLINE_CODE", "DOT_CODE", "FL_NUMBER", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE",
}
CATEGORIES = ["AIRLINE_CODE", "ORIGIN", "DEST", "CANCELLED", "DIVERTED", "CANCELLATION_CODE"]


def percentage(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def file_info(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def update_numeric(stats, series):
    """Gabungkan mean dan M2 per chunk untuk standar deviasi sampel."""
    values = series.dropna().to_numpy(dtype=float)
    if not len(values):
        return
    n = len(values)
    mean = float(values.mean())
    m2 = float(np.sum((values - mean) ** 2))
    previous = stats.get("count", 0)
    total = previous + n
    delta = mean - stats.get("mean", 0.0)
    stats["mean"] = stats.get("mean", 0.0) + delta * n / total
    stats["m2"] = stats.get("m2", 0.0) + m2 + delta * delta * previous * n / total
    stats["count"] = total
    stats["min"] = min(stats.get("min", float("inf")), float(values.min()))
    stats["max"] = max(stats.get("max", float("-inf")), float(values.max()))


def feature_metadata(column):
    if column == "DEP_DELAY":
        role = "Target E1/E2; dilarang sebagai fitur pra-keberangkatan"
    elif column == "CANCELLED":
        role = "Target E3; filter populasi E1/E2; bukan fitur"
    elif column in PREDEPARTURE:
        role = "Kandidat fitur pra-keberangkatan (ketersediaan jadwal perlu asumsi)"
    elif column in {"AIRLINE", "AIRLINE_DOT", "ORIGIN_CITY", "DEST_CITY"}:
        role = "Label/identitas untuk pelaporan; periksa redundansi"
    elif column in NUMERIC_COLUMNS or column == "CANCELLATION_CODE":
        role = "Diagnostik setelah kejadian; dilarang sebagai fitur pra-keberangkatan"
    else:
        role = "Belum dipetakan; perlu verifikasi"
    if column in TIME_COLUMNS:
        unit = "HHMM waktu lokal"
    elif column == "DISTANCE":
        unit = "mil"
    elif column == "FL_DATE":
        unit = "tanggal YYYY-MM-DD"
    elif column in {"CANCELLED", "DIVERTED"}:
        unit = "indikator 0/1"
    elif column in {"DOT_CODE", "FL_NUMBER"} or column in TEXT_COLUMNS:
        unit = "identitas/kategori"
    elif column in NUMERIC_COLUMNS:
        unit = "menit"
    else:
        unit = "perlu verifikasi"
    return role, unit


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(display(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


SUMMARY_EXPLANATIONS = """### Penjelasan istilah pada ringkasan

- **Hari kalender tanpa data secara global** adalah jumlah tanggal di antara tanggal awal dan akhir dataset yang tidak memiliki satu pun baris penerbangan. Kata “global” berarti seluruh dataset diperiksa bersama, tanpa dipisahkan per bandara, maskapai, atau rute. Nilai 0 berarti setiap tanggal memiliki setidaknya satu catatan; bukan berarti setiap bandara memiliki data setiap hari atau seluruh penerbangan tercatat.

- **Cancelled** adalah jumlah baris penerbangan yang ditandai dibatalkan (`CANCELLED == 1`). Baris ini tetap diperlukan untuk penelitian pembatalan (E3), tetapi dikeluarkan dari populasi E1/E2 yang digunakan dalam laporan.

- **Cancellation rate (% seluruh baris)** adalah persentase penerbangan yang ditandai dibatalkan terhadap seluruh baris dataset: `jumlah CANCELLED == 1 / jumlah seluruh baris × 100%`. Ini adalah tingkat pembatalan dalam sampel, bukan probabilitas hasil prediksi model. Jika status pembatalan ada yang kosong atau tidak valid, baris tersebut tetap masuk penyebut sesuai definisi “seluruh baris”; jumlahnya dapat diperiksa pada bagian kualitas data.

- **Diverted (termasuk jika juga cancelled)** adalah jumlah baris yang ditandai mengalami pengalihan penerbangan (`DIVERTED == 1`), misalnya dialihkan ke bandara lain dari tujuan yang dijadwalkan. Penghitungan ini tidak memfilter `CANCELLED`, sehingga baris dengan kedua indikator bernilai 1 akan tetap dihitung jika ada. Frasa dalam kurung menjelaskan aturan hitung, bukan menyatakan bahwa kedua status tersebut pasti muncul bersamaan.

- **Populasi E1/E2: non-cancelled, DEP_DELAY tersedia** adalah kumpulan baris yang memenuhi `CANCELLED == 0` dan nilai `DEP_DELAY` tidak kosong. E1 adalah eksperimen klasifikasi apakah keterlambatan keberangkatan mencapai 15 menit; E2 adalah eksperimen prediksi jumlah menit keterlambatan. `DEP_DELAY` menyimpan selisih menit antara keberangkatan aktual dan terjadwal. Populasi ini mencakup penerbangan lebih awal, tepat waktu, dan terlambat; penerbangan diverted tetap masuk jika memenuhi kedua syarat. Kata “populasi” di sini merujuk pada data yang memenuhi syarat eksperimen di dalam sampel, sebelum pembagian train/validation/test."""


QUANTILE_EXPLANATIONS = """### Arti kuantil eksak

**Kuantil** adalah nilai batas pada posisi tertentu setelah seluruh nilai `DEP_DELAY` yang memenuhi syarat diurutkan dari yang terkecil hingga terbesar. Kuantil 0,95 sama dengan persentil ke-95 (P95): secara umum, sekitar 95% nilai berada pada atau di bawah batas tersebut. Persentase aktual dapat berbeda karena banyak penerbangan memiliki nilai delay yang sama.

**Eksak (exact)** berarti perhitungan menggunakan seluruh nilai `DEP_DELAY` dari penerbangan non-cancelled yang nilainya tersedia dalam dataset ini, tanpa mengambil sampel tambahan atau memakai algoritme kuantil perkiraan. Dataset asal tetap merupakan sampel penerbangan; kata “eksak” tidak berarti hasilnya mewakili seluruh penerbangan di dunia nyata.

Cara membaca kuantil pada tabel:

- **0,0:** nilai minimum, yaitu keberangkatan paling awal relatif terhadap jadwal.
- **0,25 (P25):** batas seperempat bagian bawah data.
- **0,5 (P50/median):** nilai tengah data. Jika nilainya -2 menit, titik tengah distribusi berada pada keberangkatan 2 menit lebih awal dari jadwal.
- **0,95 (P95):** jika nilainya 72 menit, sekitar 95% penerbangan dalam kelompok ini memiliki `DEP_DELAY` paling tinggi 72 menit; sisanya berada di ekor keterlambatan yang lebih panjang.
- **0,99 (P99):** jika nilainya 191 menit, batas untuk sekitar 99% data adalah 191 menit.
- **1,0:** nilai maksimum, yaitu keterlambatan keberangkatan terbesar yang tercatat.

Nilai negatif berarti berangkat lebih awal, nol berarti sesuai jadwal, dan positif berarti terlambat. Perhitungan menggunakan interpolasi linear bila posisi kuantil berada di antara dua nilai yang berurutan, sehingga hasil kuantil dapat berupa pecahan dan tidak harus sama dengan nilai satu penerbangan tertentu. Tampilan tabel dibulatkan hingga empat angka desimal."""


def render_report(profile, dictionary_rows):
    counts = profile["counts"]
    dates = profile["dates"]
    lines = [
        "# Dataset Understanding — Bagian 6", "",
        f"Dibuat (UTC): {profile['metadata']['generated_utc']}", "",
        "Seluruh baris dibaca. CSV sumber tidak dibersihkan atau diubah. Statistik menggambarkan sampel, bukan seluruh operasi penerbangan.", "",
        "## Ringkasan", "",
        table(["Aspek", "Hasil"], [
            ("Baris", counts["rows"]), ("Kolom", len(profile["columns"])),
            ("Ukuran CSV (MiB)", profile["metadata"]["input"]["size_bytes"] / 1024 ** 2),
            ("Tanggal awal", dates["min"]), ("Tanggal akhir", dates["max"]),
            ("Tanggal teramati", dates["observed_days"]),
            ("Hari kalender tanpa data secara global", dates["missing_calendar_days"]),
            ("Maskapai", profile["categories"]["AIRLINE_CODE"]["unique_nonmissing"]),
            ("Bandara asal", profile["categories"]["ORIGIN"]["unique_nonmissing"]),
            ("Bandara tujuan", profile["categories"]["DEST"]["unique_nonmissing"]),
            ("Rute berarah dengan asal/tujuan terisi", profile["routes"]["unique_nonmissing"]),
            ("Cancelled", counts["cancelled"]),
            ("Cancellation rate (% seluruh baris)", percentage(counts["cancelled"], counts["rows"])),
            ("Diverted (termasuk jika juga cancelled)", counts["diverted"]),
            ("Populasi E1/E2: non-cancelled, DEP_DELAY tersedia", counts["delay_eligible"]),
            ("Departure delay >=15 menit pada populasi E1", counts["delayed_15m"]),
            ("Delay rate (% populasi E1)", percentage(counts["delayed_15m"], counts["delay_eligible"])),
        ]), "", SUMMARY_EXPLANATIONS, "", "## Cakupan per tahun", "",
    ]
    for label in ("years", "months"):
        if label == "months":
            lines += ["", "## Cakupan per bulan", ""]
        lines.append(table(["Periode", "Baris", "Cancelled", "Eligible E1", "Delay >=15", "Delay rate (%)"], [
            (period, v["rows"], v["cancelled"], v["eligible"], v["delayed"], percentage(v["delayed"], v["eligible"]))
            for period, v in profile[label].items()
        ]))
    lines += ["", "## Profil kolom", "",
              "Tipe teramati adalah hasil inferensi pandas per chunk. Missing mengikuti parser CSV pandas; string kosong dan penanda NA bawaan dihitung missing.", "",
              table(["Kolom", "Tipe teramati", "Missing", "Missing (%)", "Minimum", "Maksimum", "Mean", "Std sampel"], [
                  (c, ", ".join(p["observed_dtypes"]), p["missing"], p["missing_pct"],
                   p.get("numeric", {}).get("min"), p.get("numeric", {}).get("max"),
                   p.get("numeric", {}).get("mean"), p.get("numeric", {}).get("std"))
                  for c, p in profile["columns"].items()
              ]), "", "Mean/std untuk kode identitas, indikator, dan HHMM bukan ukuran durasi operasional.", "",
              "## Distribusi departure delay", "",
              "Kuantil eksak pada non-cancelled dengan DEP_DELAY tersedia; nilai negatif tetap dipertahankan.", "",
              table(["Kuantil", "Menit"], profile["departure_delay_quantiles"].items()), "",
              QUANTILE_EXPLANATIONS, "",
              "## Status dan ketersediaan label", "",
              table(["Kelompok", "Baris", "DEP_DELAY missing", "ARR_DELAY missing"], [
                  (label, g["rows"], g["DEP_DELAY_missing"], g["ARR_DELAY_missing"])
                  for label, g in profile["status_groups"].items()
              ]), "", "## Pemeriksaan kualitas", "",
              table(["Pemeriksaan", "Jumlah"], [(k, v) for k, v in counts.items() if k not in {"rows", "cancelled", "diverted", "delay_eligible", "delayed_15m"}]), "",
              "Duplikat adalah kandidat berdasarkan hash 64-bit, bukan verifikasi kesamaan seluruh baris. Kunci: " + ", ".join(KEY) + ".", "",
              "Tanggal mundur dihitung antara baris bertetangga. Jam 2400 dicatat terpisah dari HHMM tidak valid; normalisasi dilakukan pada tahap cleaning.", "",
              "## Kategori dan rute teratas", ""]
    for c, info in {**profile["categories"], "ROUTE": profile["routes"]}.items():
        lines += [f"### {c}", "", table(["Nilai", "Baris"], [(v["value"], v["count"]) for v in info["top10"]]), ""]
    lines += ["## Tinjauan kamus fitur", "",
              table(["Kolom", "Peran usulan", "Catatan"], [(r["column"], r["proposed_role"], r["review_notes"]) for r in dictionary_rows]), "",
              "Kolom kamus yang tidak ditemukan di CSV: " + (", ".join(profile["dictionary_review"]["dictionary_not_in_dataset"]) or "tidak ada") + ".", "",
              "## Keputusan yang diperlukan sebelum cleaning/modeling", "",
              "- Tetapkan horizon prediksi dan asumsi ketersediaan jadwal; dataset tidak menyimpan riwayat revisi jadwal.",
              "- E1/E2 menggunakan CANCELLED == 0 dan DEP_DELAY tersedia. Jangan mengisi target missing dengan nol; diverted dengan label keberangkatan tersedia dapat tetap dianalisis.",
              "- E3 menggunakan label CANCELLED yang valid. Status dan waktu aktual serta DELAY_DUE_* tidak boleh menjadi prediktor sebelum keberangkatan.",
              "- Tentukan target E2: selisih bertanda atau durasi max(DEP_DELAY, 0). Nilai ekstrem perlu ditinjau sebelum dibuang.",
              "- Pisahkan data secara waktu. Statistik laporan ini untuk pemahaman data; parameter preprocessing dipelajari dari train dan test disimpan untuk evaluasi akhir.",
              "- Verifikasi sumber, metode sampling, dan pemetaan kode pembatalan. Laporan lokal tidak membuktikan representativitas atau sebab-akibat.", "",
              "## Metode dan keterbatasan", "",
              f"Chunk: {profile['metadata']['chunk_size']:,} baris. Waktu pemindaian dan peringkasan: {profile['metadata']['elapsed_seconds']:.1f} detik.", "",
              "CSV dibaca bertahap. Hash duplikasi dan nilai DEP_DELAY eligible disimpan sepanjang pemindaian, sehingga sebagian penggunaan RAM tetap bertambah dengan jumlah baris. Identitas input memakai path, ukuran, dan waktu modifikasi; bukan checksum kriptografis.", "",
              "Kirim laporan Markdown ini untuk analisis lanjutan. JSON menyimpan rincian numerik dan CSV menyimpan tinjauan kamus fitur.", ""]
    return "\n".join(lines)


def run(input_path=INPUT, dictionary_path=DICTIONARY, output_dir=OUTPUT, chunk_size=CHUNK_SIZE):
    started = perf_counter()
    input_path, dictionary_path, output_dir = map(Path, (input_path, dictionary_path, output_dir))
    before = file_info(input_path)
    dictionary_before = file_info(dictionary_path)
    dictionary = pd.read_csv(dictionary_path).fillna("")
    required_dictionary = {"Updated Header", "Source Header", "Data Type", "Description"}
    if not required_dictionary.issubset(dictionary.columns):
        raise ValueError("Format kamus fitur tidak sesuai: perlu Updated Header, Source Header, Data Type, Description.")
    if dictionary["Updated Header"].duplicated().any():
        raise ValueError("Kamus fitur memuat nama kolom duplikat; periksa sebelum audit.")
    columns = list(pd.read_csv(input_path, nrows=0).columns)
    required = set(KEY + TIME_COLUMNS + CATEGORIES + ["DEP_DELAY", "ARR_DELAY"])
    if required - set(columns):
        raise ValueError("Kolom wajib tidak tersedia: " + ", ".join(sorted(required - set(columns))))
    counts = Counter({k: 0 for k in [
        "rows", "cancelled", "diverted", "delay_eligible", "delayed_15m", "invalid_dates",
        "date_order_inversions", "cancelled_with_dep_delay", "noncancelled_missing_dep_delay",
        "negative_dep_delay", "invalid_CANCELLED", "invalid_DIVERTED",
    ]})
    missing, day_counts, routes = Counter(), Counter(), Counter()
    dtypes, numeric, categories = defaultdict(set), defaultdict(dict), defaultdict(Counter)
    periods = {"years": defaultdict(Counter), "months": defaultdict(Counter)}
    groups = defaultdict(Counter)
    row_hashes, key_hashes, delay_values = [], [], []
    previous_date = None
    print(f"Input: {input_path.resolve()}", flush=True)
    print("Memeriksa seluruh data; CSV sumber tidak diubah.", flush=True)
    for df in pd.read_csv(input_path, chunksize=chunk_size):
        counts["rows"] += len(df)
        missing.update({c: int(v) for c, v in df.isna().sum().items()})
        for c in columns:
            dtypes[c].add(str(df[c].dtype))
        # Normalisasi tipe numerik untuk statistik dan hash yang konsisten antar-chunk.
        canonical = df.copy()
        for c in columns:
            if c in NUMERIC_COLUMNS:
                s = pd.to_numeric(df[c], errors="raise").astype("float64")
                if np.isinf(s.to_numpy()).any():
                    raise ValueError(f"Kolom {c} mengandung infinity; periksa data sumber.")
                df[c] = s
                canonical[c] = s
                update_numeric(numeric[c], s)
            else:
                canonical[c] = df[c].astype("string").fillna("<MISSING>")
        dates = pd.to_datetime(df.FL_DATE, format="%Y-%m-%d", errors="coerce")
        counts["invalid_dates"] += int(dates.isna().sum())
        counts["date_order_inversions"] += int(dates.diff().lt(pd.Timedelta(0)).sum())
        if previous_date is not None and pd.notna(previous_date) and pd.notna(dates.iloc[0]):
            counts["date_order_inversions"] += int(dates.iloc[0] < previous_date)
        previous_date = dates.iloc[-1]
        day_counts.update({str(k.date()): int(v) for k, v in dates.value_counts().items()})
        cancelled, noncancelled, diverted = df.CANCELLED.eq(1), df.CANCELLED.eq(0), df.DIVERTED.eq(1)
        eligible = noncancelled & df.DEP_DELAY.notna()
        delayed = eligible & df.DEP_DELAY.ge(15)
        for name, mask in {
            "cancelled": cancelled, "diverted": diverted, "delay_eligible": eligible,
            "delayed_15m": delayed, "cancelled_with_dep_delay": cancelled & df.DEP_DELAY.notna(),
            "noncancelled_missing_dep_delay": noncancelled & df.DEP_DELAY.isna(),
            "negative_dep_delay": df.DEP_DELAY.lt(0),
            "invalid_CANCELLED": ~df.CANCELLED.isin([0, 1]),
            "invalid_DIVERTED": ~df.DIVERTED.isin([0, 1]),
        }.items():
            counts[name] += int(mask.sum())
        delay_values.append(df.loc[eligible, "DEP_DELAY"].to_numpy())
        masks = {"cancelled": cancelled, "noncancelled_diverted": noncancelled & diverted,
                 "noncancelled_not_diverted": noncancelled & df.DIVERTED.eq(0),
                 "unknown_status": ~df.CANCELLED.isin([0, 1]) | (noncancelled & ~df.DIVERTED.isin([0, 1]))}
        for label, mask in masks.items():
            groups[label]["rows"] += int(mask.sum())
            for c in ("DEP_DELAY", "ARR_DELAY"):
                groups[label][c + "_missing"] += int((mask & df[c].isna()).sum())
        for label, fmt in (("years", "%Y"), ("months", "%Y-%m")):
            block = pd.DataFrame({"period": dates.dt.strftime(fmt), "rows": 1,
                                  "cancelled": cancelled.astype(int), "eligible": eligible.astype(int),
                                  "delayed": delayed.astype(int)})
            for period, row in block.groupby("period").sum().iterrows():
                periods[label][period].update({k: int(v) for k, v in row.items()})
        for c in CATEGORIES:
            categories[c].update({str(k): int(v) for k, v in df[c].value_counts().items()})
        valid_routes = df.ORIGIN.notna() & df.DEST.notna()
        routes.update(zip(df.loc[valid_routes, "ORIGIN"], df.loc[valid_routes, "DEST"]))
        for c in TIME_COLUMNS:
            s = df[c].dropna()
            counts[c + "_2400"] += int(s.eq(2400).sum())
            counts[c + "_invalid_hhmm_except_2400"] += int(((s < 0) | (s > 2400) | (s % 100 >= 60) | (s % 1 != 0)).sum())
        for c in ("DISTANCE", "CRS_ELAPSED_TIME", "ELAPSED_TIME", "AIR_TIME"):
            if c in df:
                counts[c + "_nonpositive"] += int(df[c].le(0).sum())
        row_hashes.append(pd.util.hash_pandas_object(canonical, index=False).to_numpy())
        key_hashes.append(pd.util.hash_pandas_object(canonical[KEY], index=False).to_numpy())
        print(f"  {counts['rows']:,} baris diperiksa ({perf_counter() - started:.1f} detik)", flush=True)
    if counts["rows"] == 0:
        raise ValueError("CSV tidak memiliki baris data.")
    if before != file_info(input_path) or dictionary_before != file_info(dictionary_path):
        raise ValueError("File input berubah selama pemeriksaan. Jalankan ulang setelah file stabil.")
    print("Meringkas statistik dan kandidat duplikat...", flush=True)
    for name, blocks in (("full_row_hash_duplicate_candidates", row_hashes), ("flight_key_hash_duplicate_candidates", key_hashes)):
        hashes = np.concatenate(blocks)
        counts[name] = int(len(hashes) - len(np.unique(hashes)))
    date_keys = sorted(day_counts)
    date_summary = {"min": date_keys[0] if date_keys else None, "max": date_keys[-1] if date_keys else None,
                    "observed_days": len(date_keys), "missing_calendar_days": None}
    if date_keys:
        span = (pd.Timestamp(date_keys[-1]) - pd.Timestamp(date_keys[0])).days + 1
        date_summary["missing_calendar_days"] = span - len(date_keys)
    column_profiles = {}
    for c in columns:
        entry = {"observed_dtypes": sorted(dtypes[c]), "missing": missing[c],
                 "missing_pct": percentage(missing[c], counts["rows"])}
        if c in numeric and numeric[c]:
            stats = dict(numeric[c])
            m2 = stats.pop("m2")
            stats["std"] = float(np.sqrt(max(m2, 0) / (stats["count"] - 1))) if stats["count"] > 1 else None
            entry["numeric"] = stats
        column_profiles[c] = entry
    lookup = dictionary.set_index("Updated Header").to_dict("index")
    dictionary_rows = []
    for c in columns:
        original = lookup.get(c, {})
        role, unit = feature_metadata(c)
        notes = []
        if not original:
            notes.append("Belum tersedia dalam kamus sumber")
        if original and original["Data Type"] not in dtypes[c]:
            notes.append("Tipe deklarasi berbeda dari inferensi per chunk; tinjau efek missing")
        if c == "FL_DATE":
            notes.append("Format yang diperiksa YYYY-MM-DD; deskripsi kamus perlu diselaraskan")
        if c in {"DOT_CODE", "FL_NUMBER"}:
            notes.append("Pengenal kategorikal, bukan besaran kontinu")
        if c in TIME_COLUMNS:
            notes.append("HHMM waktu lokal; 2400 memerlukan aturan normalisasi")
        if c == "CANCELLATION_CODE":
            notes.append("Arti kode perlu verifikasi provenance; null dapat bersifat struktural")
        if c == "AIRLINE_DOT":
            notes.append("Usulan: label gabungan nama/kode maskapai; bukan nomor DOT; verifikasi sumber")
        if c == "AIRLINE":
            notes.append("Usulan: nama maskapai; verifikasi sumber")
        dictionary_rows.append({"column": c, "source_header": original.get("Source Header", ""),
            "source_description": original.get("Description", ""), "declared_dtype": original.get("Data Type", ""),
            "observed_dtypes": ", ".join(sorted(dtypes[c])), "unit": unit, "proposed_role": role,
            "missing_count": missing[c], "missing_pct": percentage(missing[c], counts["rows"]),
            "review_notes": "; ".join(notes) or "Peran adalah usulan; konfirmasi pada rancangan eksperimen"})
    values = np.concatenate(delay_values)
    quantiles = {str(q): float(v) for q, v in pd.Series(values).quantile([0, .01, .25, .5, .75, .9, .95, .99, .999, 1]).items()} if len(values) else {}
    profile = {
        "metadata": {"generated_utc": datetime.now(timezone.utc).isoformat(), "input": before,
                     "dictionary": dictionary_before, "chunk_size": chunk_size,
                     "elapsed_seconds": round(perf_counter() - started, 2),
                     "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
        "counts": dict(counts), "dates": date_summary, "daily_sample_rows": dict(sorted(day_counts.items())),
        "columns": column_profiles, "status_groups": dict(groups),
        **{label: dict(sorted(v.items())) for label, v in periods.items()},
        "categories": {c: {"unique_nonmissing": len(v), "top10": [{"value": k, "count": n} for k, n in v.most_common(10)]} for c, v in categories.items()},
        "routes": {"unique_nonmissing": len(routes), "top10": [{"value": f"{k[0]} -> {k[1]}", "count": n} for k, n in routes.most_common(10)]},
        "departure_delay_quantiles": quantiles,
        "dictionary_review": {"dataset_not_in_dictionary": sorted(set(columns) - set(lookup)),
                              "dictionary_not_in_dataset": sorted(set(lookup) - set(columns))},
    }
    report = render_report(profile, dictionary_rows)
    json_text = json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_profile.json").write_text(json_text, encoding="utf-8")
    pd.DataFrame(dictionary_rows).to_csv(output_dir / "data_dictionary_review.csv", index=False, encoding="utf-8-sig")
    report_path = output_dir / "dataset_understanding_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSelesai. Hasil tersimpan di: {output_dir.resolve()}")
    print(f"Kirim isi file ini ke chat: {report_path.resolve()}")
    print("Menjalankan ulang skrip akan memperbarui tiga file hasil pada folder tersebut.")
    return profile


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("\nPemeriksaan dibatalkan. CSV sumber tidak diubah.")
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"\nPemeriksaan gagal: {exc}")
