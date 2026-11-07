"""Latih baseline Logistic Regression dan evaluasi pada validation.

Jalankan: python logistic-regression-train.py
Input: classification/datasets/model-ready dan outputs/feature_contract.json.
Output: classification/modelling/output/#2-LR-train.
Preprocessing tidak di-fit ulang; data test tidak dibaca.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import warnings

try:
    import joblib
    import numpy as np
    import scipy
    from scipy import sparse
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                                 confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    requirements = Path(__file__).resolve().with_name("requirements.txt")
    raise SystemExit(f'Dependensi belum tersedia. Jalankan: python -m pip install -r "{requirements}"')


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "datasets" / "model-ready"
CONTRACT = ROOT.parent / "outputs" / "feature_contract.json"
OUTPUT = ROOT / "output" / "#2-LR-train"
THRESHOLD = 0.5
# Percobaan awal: regularisasi L2 bawaan, tanpa tuning atau class weighting.
MODEL_PARAMETERS = {"solver": "lbfgs", "C": 1.0, "class_weight": None,
                    "max_iter": 1000, "tol": 1e-4, "random_state": 42}
METRIC_NAMES = {
    "accuracy": "Accuracy", "precision_class_1": "Precision kelas 1",
    "recall_class_1": "Recall kelas 1", "f1_class_1": "F1 kelas 1",
    "roc_auc": "ROC-AUC", "pr_auc_trapezoid": "PR-AUC (trapesium)",
    "average_precision": "Average Precision (AP)",
}


def file_info(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def load_split(data_dir, name, feature_count):
    x = sparse.load_npz(data_dir / f"X_{name}.npz").tocsr()
    y = np.load(data_dir / f"y_{name}.npy", allow_pickle=False)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Label {name} harus array satu dimensi, tidak kosong, dan hanya berisi 0/1.")
    if x.shape != (len(y), feature_count):
        raise ValueError(f"Dimensi {name} tidak cocok: X={x.shape}, y={len(y)}, fitur kontrak={feature_count}.")
    if not np.isfinite(x.data).all():
        raise ValueError(f"Matriks {name} mengandung NaN atau infinity.")
    if np.unique(y).size != 2:
        raise ValueError(f"Subset {name} harus memiliki kedua kelas untuk training/evaluasi ROC-AUC ini.")
    return x, y.astype(np.int8, copy=False)


def class_metrics(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_class_1": tp / (tp + fp) if tp + fp else 0.0,
        "recall_class_1": tp / (tp + fn) if tp + fn else 0.0,
        "f1_class_1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": None, "pr_auc_trapezoid": None, "average_precision": None,
    }, {"precision_defined": bool(tp + fp), "recall_defined": bool(tp + fn)}, matrix


def save_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(folder, matrix, fpr, tpr, precision, recall, metrics, prevalence):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    chart = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(chart, ax=ax, label="Jumlah penerbangan")
    labels = ["0: delay <15 menit", "1: delay >=15 menit"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Prediksi", ylabel="Aktual", title="Logistic Regression — validation\nThreshold = 0,5")
    tags = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            ax.text(column, row, f"{tags[row][column]}\n{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(folder / "logistic-regression-confusion-matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Acuan tanpa pemeringkatan")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC — validation")
    axes[1].plot(recall, precision, label=f"AP = {metrics['average_precision']:.4f}")
    axes[1].axhline(prevalence, linestyle="--", color="gray", label=f"Proporsi kelas 1 = {prevalence:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall — validation")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="best")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / "logistic-regression-curves.png", dpi=180)
    plt.close(fig)


def render_report(result):
    metrics, baseline = result["metrics_validation"], result["majority_baseline"]["metrics_validation"]
    matrix = result["confusion_matrix"]
    lines = ["# Logistic Regression — Klasifikasi Keterlambatan Keberangkatan", "",
             f"Dibuat (UTC): {result['created_utc']}", "",
             "## Rancangan percobaan", "",
             "Model dilatih pada matriks train hasil preprocessing dan dievaluasi pada validation. Label positif adalah DELAYED_15=1: departure delay minimal 15 menit, pada populasi non-cancelled dengan label tersedia.", "",
             "Threshold awal ditetapkan 0,5 sebelum evaluasi: probabilitas kelas 1 >=0,5 menjadi prediksi 1. Belum ada tuning, class weighting, resampling, atau perubahan fitur. Test tidak dibaca. Transformer sebelumnya tidak di-fit ulang.", "",
             "| Parameter | Nilai |", "| --- | --- |",
             *[f"| {name} | {value} |" for name, value in result["model_parameters"].items()],
             "| Regularisasi | L2 (bawaan LogisticRegression) |", "",
             "C=1 adalah pengaturan awal kekuatan regularisasi. Solver lbfgs digunakan untuk baseline ini. Penetapan parameter tersebut belum membuktikan bahwa nilainya optimal.", "",
             "## Data", "", "| Subset | Baris | Fitur | Label 0 | Label 1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, info in result["data"].items():
        lines.append(f"| {name} | {info['rows']:,} | {info['features']:,} | {info['label_0']:,} | {info['label_1']:,} |")
    lines += ["", "## Status training", "",
              f"- Waktu fit: {result['fit_seconds']:.2f} detik.",
              f"- Iterasi solver: {result['iterations']} dari batas {MODEL_PARAMETERS['max_iter']}.",
              f"- Konvergensi solver: {'tercapai menurut status solver' if result['converged'] else 'BELUM tercapai; hasil perlu ditinjau sebelum perbandingan final'}.", ""]
    if result["warnings"]:
        lines += ["Pesan selama fit:", "", *[f"- {message}" for message in result["warnings"]], ""]
    lines += ["## Perbandingan pada validation", "",
              "Baseline mayoritas dihitung ulang dari y_train dan y_validation yang sama, sehingga perbandingan tetap berlaku walaupun folder hasil baseline sebelumnya dipindah. Kelas mayoritasnya adalah " + str(result["majority_baseline"]["majority_class_from_train"]) + ".", "",
              "| Metrik (0–1) | Mayoritas | Logistic Regression | Selisih LR − mayoritas |",
              "| --- | ---: | ---: | ---: |"]
    for name, title in METRIC_NAMES.items():
        previous = baseline[name]
        previous_text = "—" if previous is None else f"{previous:.6f}"
        difference = "—" if previous is None else f"{metrics[name] - previous:+.6f}"
        lines.append(f"| {title} | {previous_text} | {metrics[name]:.6f} | {difference} |")
    lines += ["", "Precision, recall, dan F1 mengacu pada kelas terlambat. Precision dengan nol prediksi positif ditampilkan sebagai 0, dengan status tidak terdefinisi dicatat dalam JSON/CSV. Metrik pemeringkatan baseline kelas keras tidak dievaluasi.", "",
              "PR-AUC (trapesium) mengintegrasikan kurva precision–recall secara linear. Average Precision (AP) merangkum precision dengan bobot kenaikan recall tanpa interpolasi linear; kedua angka dapat berbeda. ROC-AUC, PR-AUC, dan AP dihitung dari probabilitas, sehingga tidak bergantung pada threshold 0,5.", "",
              "## Confusion matrix", "",
              "| Aktual / Prediksi | 0 | 1 |", "| --- | ---: | ---: |",
              f"| 0 | {matrix[0][0]:,} (TN) | {matrix[0][1]:,} (FP) |",
              f"| 1 | {matrix[1][0]:,} (FN) | {matrix[1][1]:,} (TP) |", "",
              "![Confusion matrix](logistic-regression-confusion-matrix.png)", "",
              "## Kurva evaluasi", "", "![Kurva ROC dan PR](logistic-regression-curves.png)", "",
              "## Interpretasi hasil awal", "",
              f"Pada threshold 0,5, model mendeteksi {matrix[1][1]:,} penerbangan terlambat (TP), melewatkan {matrix[1][0]:,} (FN), dan menghasilkan {matrix[0][1]:,} peringatan keliru (FP).",
              f"Accuracy = {100 * metrics['accuracy']:.4f}%; recall kelas terlambat = {100 * metrics['recall_class_1']:.4f}%; precision kelas terlambat = {100 * metrics['precision_class_1']:.4f}%.", "",
              "Nilai metrik perlu dibaca bersama trade-off deteksi keterlambatan dan peringatan keliru. Konvergensi solver bukan bukti kualitas prediksi. Jika model belum konvergen, tinjau training terlebih dahulu; jika sudah konvergen, hasil ini menjadi acuan untuk eksperimen threshold atau class weighting pada validation. Test tetap untuk evaluasi akhir.", "",
              "## Artefak", "",
              "- logistic-regression-model.joblib: model terlatih, menerima matriks dengan urutan fitur pada feature_contract.json yang dipakai saat training.",
              "- logistic-regression-results.json: metrik, parameter, status konvergensi, versi, dan identitas input.",
              "- logistic-regression-metrics.csv: perbandingan metrik LR dan mayoritas, termasuk status precision.",
              "- logistic-regression-confusion-matrix.csv dan PNG: tabel serta visualisasi kesalahan.",
              "- logistic-regression-curves.png: kurva ROC dan precision–recall.",
              "- logistic-regression-validation-predictions.npz: dataset_row (mulai 1), y_true, y_pred, probability_class_1. Urutan sama dengan matriks validation dan CSV prepared; dapat dipakai untuk analisis threshold berikutnya.",
              "- logistic-regression-coefficients.csv: koefisien sesuai urutan fitur. Koefisien mengacu pada skala fitur setelah preprocessing; bukan bukti sebab-akibat.", "",
              "## Reproduksibilitas", ""]
    for name, info in result["input"].items():
        lines += [f"- {name}: `{info['path']}`", f"  SHA-256: `{info['sha256']}`"]
    lines += ["", "Versi: " + ", ".join(f"{key} {value}" for key, value in result["versions"].items()) + ".", "",
              "Model memakai matriks yang sudah ditransformasi. Untuk penerbangan baru, gunakan aturan FE dan transformer preprocessing yang sama sebelum model ini. Jika melakukan cross-validation, fit ulang preprocessing di setiap fold dari CSV prepared.", ""]
    return "\n".join(lines)


def run(data_dir=DATA, contract_path=CONTRACT, output_dir=OUTPUT):
    data_dir, contract_path, output_dir = map(lambda path: Path(path).resolve(), (data_dir, contract_path, output_dir))
    started = perf_counter()
    print("[1/5] Memeriksa kontrak fitur dan input train/validation...", flush=True)
    paths = {"feature_contract": contract_path,
             **{f"{kind}_{name}": data_dir / f"{kind}_{name}.{extension}"
                for name in ("train", "validation") for kind, extension in (("X", "npz"), ("y", "npy"))}}
    inputs = {name: file_info(path) for name, path in paths.items()}
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    features = contract["output_feature_names"]
    if contract.get("preprocessing_fitted_on") != "train" or contract.get("target") != "DELAYED_15":
        raise ValueError("Kontrak harus menetapkan preprocessing fit pada train dan target DELAYED_15.")
    if not features or len(features) != len(set(features)):
        raise ValueError("Nama fitur keluaran kosong atau duplikat.")
    x_train, y_train = load_split(data_dir, "train", len(features))
    x_valid, y_valid = load_split(data_dir, "validation", len(features))
    print(f"Train: {x_train.shape[0]:,} x {x_train.shape[1]}; validation: {x_valid.shape[0]:,} x {x_valid.shape[1]}.", flush=True)
    model = LogisticRegression(**MODEL_PARAMETERS)
    print("[2/5] Melatih Logistic Regression (lbfgs, C=1, tanpa class weighting). Proses dapat memerlukan beberapa menit...", flush=True)
    fit_started = perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    fit_seconds = perf_counter() - fit_started
    converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in captured)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Koefisien model tidak finite; hasil tidak disimpan.")
    print(f"Fit selesai dalam {fit_seconds:.1f} detik; iterasi={int(model.n_iter_.max())}; konvergen={converged}.", flush=True)
    print("[3/5] Menghitung probabilitas dan metrik validation pada threshold 0,5...", flush=True)
    probability = model.predict_proba(x_valid)[:, list(model.classes_).index(1)]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilitas validation tidak valid.")
    prediction = (probability >= THRESHOLD).astype(np.int8)
    metrics, metric_status, matrix = class_metrics(y_valid, prediction)
    precision, recall, _ = precision_recall_curve(y_valid, probability)
    fpr, tpr, _ = roc_curve(y_valid, probability)
    metrics.update({"roc_auc": float(roc_auc_score(y_valid, probability)),
                    "pr_auc_trapezoid": float(auc(recall, precision)),
                    "average_precision": float(average_precision_score(y_valid, probability))})
    majority = int(np.argmax(np.bincount(y_train, minlength=2)))
    baseline, baseline_status, _ = class_metrics(y_valid, np.full(len(y_valid), majority, dtype=np.int8))
    for name, path in paths.items():
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (inputs[name]["size_bytes"], inputs[name]["mtime_ns"]):
            raise ValueError(f"Input berubah saat training: {path.name}; hasil tidak diterbitkan.")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": "LogisticRegression",
        "model_parameters": MODEL_PARAMETERS, "threshold": THRESHOLD, "regularization": "L2",
        "evaluation_split": "validation", "test_used": False, "preprocessing_refitted": False,
        "converged": converged, "iterations": int(model.n_iter_.max()), "fit_seconds": round(fit_seconds, 3),
        "warnings": [f"{warning.category.__name__}: {warning.message}" for warning in captured],
        "data": {name: {"rows": len(y), "features": len(features), "label_0": int((y == 0).sum()), "label_1": int((y == 1).sum())}
                 for name, y in (("train", y_train), ("validation", y_valid))},
        "metrics_validation": metrics, "metric_status": metric_status, "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "actual [0,1]", "columns": "predicted [0,1]"},
        "majority_baseline": {"majority_class_from_train": majority, "metrics_validation": baseline, "metric_status": baseline_status},
        "intercept": float(model.intercept_[0]), "input": inputs,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
    }
    print("[4/5] Menyimpan model, prediksi, tabel metrik, dan grafik...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lr-", dir=output_dir) as temporary:
        stage = Path(temporary)
        joblib.dump(model, stage / "logistic-regression-model.joblib", compress=3)
        np.savez_compressed(stage / "logistic-regression-validation-predictions.npz",
                            dataset_row=np.arange(1, len(y_valid) + 1, dtype=np.int64),
                            y_true=y_valid, y_pred=prediction, probability_class_1=probability)
        rows = []
        for name, values, status in (("majority_baseline", baseline, baseline_status), ("logistic_regression", metrics, metric_status)):
            for key, value in values.items():
                defined = value is not None and (key != "precision_class_1" or status["precision_defined"])
                rows.append({"model": name, "split": "validation", "metric": key, "value": value, "defined": defined})
        save_csv(stage / "logistic-regression-metrics.csv", ["model", "split", "metric", "value", "defined"], rows)
        save_csv(stage / "logistic-regression-confusion-matrix.csv", ["actual_label", "predicted_0", "predicted_1"],
                 [{"actual_label": i, "predicted_0": int(row[0]), "predicted_1": int(row[1])} for i, row in enumerate(matrix)])
        save_csv(stage / "logistic-regression-coefficients.csv", ["feature_index", "feature_name", "coefficient"],
                 [{"feature_index": i, "feature_name": name, "coefficient": float(model.coef_[0, i])} for i, name in enumerate(features)])
        save_plots(stage, matrix, fpr, tpr, precision, recall, metrics, float(y_valid.mean()))
        result["elapsed_seconds"] = round(perf_counter() - started, 3)
        (stage / "logistic-regression-results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (stage / "logistic-regression-report.md").write_text(render_report(result), encoding="utf-8")
        for path in stage.iterdir():
            path.replace(output_dir / path.name)
    print(f"[5/5] Selesai. Accuracy={metrics['accuracy']:.4f}; recall={metrics['recall_class_1']:.4f}; AP={metrics['average_precision']:.4f}.")
    if not converged:
        print("PERHATIAN: solver belum konvergen. Status dan pesan fit tercantum dalam laporan.")
    print(f"Kirim isi laporan ini ke chat:\n{output_dir / 'logistic-regression-report.md'}")
    return result


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("\nTraining dibatalkan.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"\nTraining gagal: {exc}")
