"""Baseline kelas mayoritas untuk Klasifikasi Keterlambatan Keberangkatan.

Jalankan: python majority-baseline.py
Input otomatis: ../datasets/model-ready/y_train.npy dan y_validation.npy.
Output otomatis: ./output/. Data test dan matriks fitur tidak dibaca.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    requirements = Path(__file__).resolve().with_name("requirements.txt")
    raise SystemExit(f'Dependensi belum tersedia. Jalankan: python -m pip install -r "{requirements}"')


ROOT = Path(__file__).resolve().parent
LABELS = ROOT.parent / "datasets" / "model-ready"
OUTPUT = ROOT / "output"
PREFIX = "majority-baseline"
CLASS_NAMES = ["0: delay <15 menit", "1: delay >=15 menit"]


def load_labels(path):
    """Label harus valid agar perhitungan baseline dapat dibandingkan antar-model."""
    labels = np.load(path, allow_pickle=False)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{path.name} harus berisi array label satu dimensi yang tidak kosong.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path.name} mengandung label selain 0/1 atau nilai kosong.")
    return labels.astype(np.int8, copy=False)


def evaluate(y_train, y_validation):
    """Kelas mayoritas dipilih hanya dari train, termasuk jika distribusi validation berbeda."""
    train_counts = np.bincount(y_train, minlength=2)
    validation_counts = np.bincount(y_validation, minlength=2)
    # np.argmax memilih indeks pertama: jika train seimbang, aturan tetap memilih 0.
    majority = int(np.argmax(train_counts))
    matrix = np.zeros((2, 2), dtype=np.int64)
    matrix[:, majority] = validation_counts
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision_defined = tp + fp > 0
    recall_defined = tp + fn > 0
    f1_defined = 2 * tp + fp + fn > 0
    metrics = {
        "accuracy": (tp + tn) / len(y_validation),
        "precision_class_1": tp / (tp + fp) if precision_defined else 0.0,
        "recall_class_1": tp / (tp + fn) if recall_defined else 0.0,
        "f1_class_1": 2 * tp / (2 * tp + fp + fn) if f1_defined else 0.0,
    }
    return {
        "strategy": "most_frequent", "majority_class_from_train": majority,
        "tie_break_rule": "Choose class 0 if train class counts are equal",
        "counts": {"train": train_counts.tolist(), "validation": validation_counts.tolist()},
        "metrics_validation": metrics,
        "metric_defined": {"accuracy": True, "precision_class_1": precision_defined,
                           "recall_class_1": recall_defined, "f1_class_1": f1_defined},
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "actual [0, 1]", "columns": "predicted [0, 1]"},
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_class": 1, "undefined_metric_policy": "display 0 and flag as undefined",
        "roc_auc": None, "pr_auc": None,
        "ranking_metrics_note": "Not evaluated: this baseline outputs one constant hard class, without a discriminating risk score",
    }


def write_csv(path, fieldnames, records):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_confusion_matrix(matrix, majority, path):
    fig, axis = plt.subplots(figsize=(7, 5.5))
    chart = axis.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(chart, ax=axis, label="Jumlah penerbangan")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES,
             yticklabels=CLASS_NAMES, xlabel="Prediksi baseline", ylabel="Label aktual",
             title=f"Confusion matrix — validation\nBaseline kelas mayoritas: prediksi selalu {majority}")
    tags = [["TN", "FP"], ["FN", "TP"]]
    threshold = float(matrix.max()) / 2
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            axis.text(column, row, f"{tags[row][column]}\n{value:,}", ha="center", va="center",
                      fontsize=13, color="white" if value > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_text(result):
    majority = result["majority_class_from_train"]
    counts = result["counts"]
    metrics = result["metrics_validation"]
    cm = result["confusion_counts"]
    lines = [
        "# Baseline kelas mayoritas — Klasifikasi Keterlambatan Keberangkatan", "",
        f"Dibuat (UTC): {result['created_utc']}", "",
        "## Tujuan dan metode", "",
        "Baseline ini mengukur hasil ketika setiap penerbangan mendapat prediksi kelas yang paling sering muncul pada train. Model berikutnya dibandingkan dengan hasil ini untuk menilai manfaat prediksinya.", "",
        "Label 0 berarti departure delay kurang dari 15 menit, termasuk keberangkatan lebih awal. Label 1 berarti departure delay minimal 15 menit. Populasi mengikuti preprocessing: non-cancelled dengan DEP_DELAY valid.", "",
        f"Kelas mayoritas yang dipilih dari train adalah **{majority}**. Semua baris validation diprediksi sebagai **{majority}**. Jika jumlah kedua kelas pada train sama, aturan tetap memilih kelas 0.", "",
        "Baseline ini tidak memerlukan fitur X. Tidak ada threshold yang dituning, probabilitas risiko yang dipelajari, atau resampling. Data test tidak dibaca. Perhitungan confusion matrix dari jumlah kelas setara dengan membuat prediksi konstan untuk setiap baris validation.", "",
        "## Distribusi label", "",
        "| Subset | Total | Label 0 | Label 1 | Label 1 (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, distribution in counts.items():
        total = sum(distribution)
        lines.append(f"| {name} | {total:,} | {distribution[0]:,} | {distribution[1]:,} | {100 * distribution[1] / total:.4f} |")
    lines += ["", "## Evaluasi pada validation", "",
              "Precision, recall, dan F1 berikut menggunakan kelas 1 (terlambat) sebagai kelas positif.", "",
              "| Metrik | Nilai (0–1) | Persen | Keterangan |", "| --- | ---: | ---: | --- |"]
    for name, value in metrics.items():
        note = "Terdefinisi" if result["metric_defined"][name] else "Tidak terdefinisi; ditampilkan sebagai 0"
        lines.append(f"| {name} | {value:.6f} | {100 * value:.4f}% | {note} |")
    lines += ["", "Precision tidak terdefinisi bila tidak ada prediksi positif; recall tidak terdefinisi bila tidak ada label positif aktual. F1 tidak terdefinisi bila TP, FP, dan FN semuanya nol. Nilai tersebut ditampilkan sebagai 0 sesuai aturan laporan, dengan status tetap dicatat dalam JSON/CSV.", "",
              "ROC-AUC dan PR-AUC tidak dievaluasi pada langkah ini. Prediksi kelas konstan tidak membedakan urutan risiko antar-penerbangan. Metrik berbasis skor akan digunakan pada model yang menghasilkan skor/probabilitas.", "",
              "## Confusion matrix", "",
              "Baris menunjukkan label aktual, kolom menunjukkan prediksi.", "",
              "| Aktual / Prediksi | 0: delay <15 menit | 1: delay >=15 menit |", "| --- | ---: | ---: |",
              f"| 0: delay <15 menit | {cm['tn']:,} (TN) | {cm['fp']:,} (FP) |",
              f"| 1: delay >=15 menit | {cm['fn']:,} (FN) | {cm['tp']:,} (TP) |", "",
              f"![Confusion matrix pada validation]({PREFIX}-confusion-matrix.png)", "",
              "TN: kelas 0 dikenali dengan benar. FP: peringatan keterlambatan pada kelas 0. FN: keterlambatan yang terlewat. TP: keterlambatan yang terdeteksi.", "",
              "## Interpretasi awal", ""]
    if majority == 0:
        lines.append(f"Accuracy {100 * metrics['accuracy']:.4f}% berasal dari keberhasilan menebak seluruh {cm['tn']:,} penerbangan kelas 0. Semua {cm['fn']:,} penerbangan terlambat terlewat karena tidak ada prediksi kelas 1. Jika validation memiliki label positif, recall kelas terlambat bernilai 0; accuracy saja tidak menunjukkan kemampuan mendeteksi keterlambatan.")
    else:
        lines.append(f"Semua penerbangan diprediksi terlambat. Baseline mendeteksi {cm['tp']:,} penerbangan terlambat, tetapi juga memberikan {cm['fp']:,} peringatan keliru pada penerbangan kelas 0. Jika validation memiliki label positif, recall bernilai 1; hal ini perlu dibaca bersama precision dan jumlah false positive.")
    lines += ["", "Hasil ini menjadi pembanding bagi Logistic Regression dan model berikutnya pada validation yang sama. Perbaikan dinilai dari kemampuan mendeteksi keterlambatan beserta jumlah peringatan keliru, bukan hanya kenaikan accuracy. Test disimpan untuk evaluasi akhir setelah pilihan model/parameter ditetapkan.", "",
              "## Input dan reproduksibilitas", ""]
    for name, info in result["input"].items():
        lines += [f"- {name}: `{info['path']}`", f"  SHA-256: `{info['sha256']}`"]
    lines += ["", f"Versi: Python {result['versions']['python']}, NumPy {result['versions']['numpy']}, matplotlib {result['versions']['matplotlib']}.",
              f"Waktu proses: {result['elapsed_seconds']:.2f} detik.", "",
              "Hasil perhitungan disimpan dalam JSON, tabel metrik/distribusi/confusion matrix dalam CSV, dan visualisasi dalam PNG. Menjalankan ulang skrip memperbarui file berawalan majority-baseline di folder output.", ""]
    return "\n".join(lines)


def run(labels_dir=LABELS, output_dir=OUTPUT):
    started = perf_counter()
    labels_dir, output_dir = Path(labels_dir).resolve(), Path(output_dir).resolve()
    print("Membaca label train dan validation...", flush=True)
    paths = {name: labels_dir / f"y_{name}.npy" for name in ("train", "validation")}
    initial = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in paths.items()}
    inputs = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()}
    y_train, y_validation = (load_labels(paths[name]) for name in ("train", "validation"))
    result = evaluate(y_train, y_validation)
    print(f"Train: {len(y_train):,} baris; validation: {len(y_validation):,} baris.", flush=True)
    print(f"Kelas mayoritas train: {result['majority_class_from_train']}. Menghitung evaluasi dan membuat laporan...", flush=True)
    for name, path in paths.items():
        if initial[name] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError(f"{path.name} berubah selama pembacaan. Jalankan ulang setelah file stabil.")
    result.update({"created_utc": datetime.now(timezone.utc).isoformat(), "input": inputs,
                   "evaluation_split": "validation", "test_used": False,
                   "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__}})
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".baseline-", dir=output_dir) as temporary:
        stage = Path(temporary)
        write_csv(stage / f"{PREFIX}-metrics.csv", ["split", "metric", "value", "defined"],
                  [{"split": "validation", "metric": name, "value": value, "defined": result["metric_defined"][name]}
                   for name, value in result["metrics_validation"].items()])
        distribution_rows = []
        for name, counts in result["counts"].items():
            for label, count in enumerate(counts):
                distribution_rows.append({"split": name, "label": label, "count": count, "percentage": 100 * count / sum(counts)})
        write_csv(stage / f"{PREFIX}-class-distribution.csv", ["split", "label", "count", "percentage"], distribution_rows)
        write_csv(stage / f"{PREFIX}-confusion-matrix.csv", ["actual_label", "predicted_0", "predicted_1"],
                  [{"actual_label": label, "predicted_0": values[0], "predicted_1": values[1]}
                   for label, values in enumerate(result["confusion_matrix"])])
        plot_confusion_matrix(np.asarray(result["confusion_matrix"]), result["majority_class_from_train"],
                              stage / f"{PREFIX}-confusion-matrix.png")
        result["elapsed_seconds"] = round(perf_counter() - started, 3)
        (stage / f"{PREFIX}-results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (stage / f"{PREFIX}-report.md").write_text(report_text(result), encoding="utf-8")
        for path in stage.iterdir():
            path.replace(output_dir / path.name)
    print(f"\nSelesai. Accuracy validation: {100 * result['metrics_validation']['accuracy']:.4f}%")
    print(f"Kirim isi laporan ini ke chat:\n{output_dir / (PREFIX + '-report.md')}")
    return result


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("\nProses dibatalkan.")
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"\nBaseline gagal: {exc}\nPastikan hasil preprocessing train dan validation sudah tersedia.")
