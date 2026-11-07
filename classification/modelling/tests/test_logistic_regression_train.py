"""Uji data sintetis; tidak membaca dataset proyek atau memakai test split."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import joblib
import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / "logistic-regression-train.py"), run_name="lr_test_module")


def fixture(folder):
    data = folder / "data"
    data.mkdir()
    x, y = make_classification(n_samples=240, n_features=8, n_informative=5,
                               weights=[.75, .25], random_state=7)
    for name, selection in (("train", slice(0, 160)), ("validation", slice(160, None))):
        sparse.save_npz(data / f"X_{name}.npz", sparse.csr_matrix(x[selection], dtype=np.float32))
        np.save(data / f"y_{name}.npy", y[selection].astype(np.int8))
    contract = folder / "contract.json"
    contract.write_text(json.dumps({"output_feature_names": [f"f{i}" for i in range(8)],
                                    "target": "DELAYED_15", "preprocessing_fitted_on": "train"}), encoding="utf-8")
    return data, contract


class LogisticRegressionTests(unittest.TestCase):
    def test_fit_evaluation_saved_model_and_input_immutability(self):
        with tempfile.TemporaryDirectory(prefix=".lr-test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            data, contract = fixture(folder)
            before = {path: path.read_bytes() for path in list(data.iterdir()) + [contract]}
            output = folder / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                result = M["run"](data, contract, output)
            self.assertTrue(result["converged"])
            self.assertFalse(result["test_used"])
            self.assertFalse(result["preprocessing_refitted"])
            self.assertEqual(result["threshold"], .5)
            self.assertEqual(len(list(output.iterdir())), 9)
            model = joblib.load(output / "logistic-regression-model.joblib")
            x = sparse.load_npz(data / "X_validation.npz")
            y = np.load(data / "y_validation.npy")
            probability = model.predict_proba(x)[:, 1]
            prediction = (probability >= .5).astype(np.int8)
            self.assertAlmostEqual(result["metrics_validation"]["accuracy"], accuracy_score(y, prediction))
            self.assertAlmostEqual(result["metrics_validation"]["roc_auc"], roc_auc_score(y, probability))
            self.assertAlmostEqual(result["metrics_validation"]["average_precision"], average_precision_score(y, probability))
            np.testing.assert_array_equal(result["confusion_matrix"], confusion_matrix(y, prediction, labels=[0, 1]))
            with np.load(output / "logistic-regression-validation-predictions.npz") as saved:
                np.testing.assert_array_equal(saved["dataset_row"], np.arange(1, 81))
                np.testing.assert_array_equal(saved["y_true"], y)
                np.testing.assert_array_equal(saved["y_pred"], prediction)
                np.testing.assert_allclose(saved["probability_class_1"], probability)
            majority = int(np.argmax(np.bincount(np.load(data / "y_train.npy"))))
            self.assertEqual(result["majority_baseline"]["majority_class_from_train"], majority)
            self.assertAlmostEqual(result["majority_baseline"]["metrics_validation"]["accuracy"], float((y == majority).mean()))
            self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
            for name in ("logistic-regression-curves.png", "logistic-regression-confusion-matrix.png"):
                self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_invalid_inputs_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix=".lr-test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            data, contract = fixture(folder)
            with self.assertRaisesRegex(ValueError, "Dimensi"):
                M["load_split"](data, "train", 9)
            np.save(data / "y_validation.npy", np.zeros(80, dtype=np.int8))
            with self.assertRaisesRegex(ValueError, "kedua kelas"):
                M["load_split"](data, "validation", 8)
            np.save(data / "y_validation.npy", np.full(80, 2, dtype=np.int8))
            with self.assertRaisesRegex(ValueError, "0/1"):
                M["load_split"](data, "validation", 8)
            metadata = json.loads(contract.read_text())
            metadata["preprocessing_fitted_on"] = "all"
            contract.write_text(json.dumps(metadata))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "fit pada train"):
                M["run"](data, contract, folder / "output")
            self.assertFalse((folder / "output").exists())

    def test_nonconvergence_is_reported(self):
        with tempfile.TemporaryDirectory(prefix=".lr-test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            data, contract = fixture(folder)
            parameters = M["MODEL_PARAMETERS"]
            previous = parameters["max_iter"]
            try:
                parameters["max_iter"] = 1
                with contextlib.redirect_stdout(io.StringIO()):
                    result = M["run"](data, contract, folder / "output")
                self.assertFalse(result["converged"])
                self.assertTrue(any("ConvergenceWarning" in item for item in result["warnings"]))
                report = (folder / "output/logistic-regression-report.md").read_text(encoding="utf-8")
                self.assertIn("BELUM tercapai", report)
            finally:
                parameters["max_iter"] = previous


if __name__ == "__main__":
    unittest.main()
