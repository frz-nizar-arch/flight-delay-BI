"""Uji fixture kecil: cleaning, batas waktu, train-only fit, dan artefak."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / "e1-data-preparation.py"), run_name="test_module")


def row(**changes):
    result = dict(FL_DATE="2020-01-01", AIRLINE_CODE=" ua ", ORIGIN="jfk", DEST="LAX",
                  CRS_DEP_TIME="1530", CRS_ARR_TIME="2400", CRS_ELAPSED_TIME="90",
                  DISTANCE="100", CANCELLED="0", DEP_DELAY="15")
    result.update(changes)
    return result


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_and_cyclic_missing(self):
        frame = pd.DataFrame([
            row(DEP_DELAY="14", CRS_DEP_TIME="2359"),
            row(DEP_DELAY="15", CRS_DEP_TIME="0001"),
            row(DEP_DELAY="-90", CRS_DEP_TIME=None, CRS_ARR_TIME="1265", DISTANCE="bad"),
            row(DEP_DELAY="2966", CRS_DEP_TIME="2400"),
            row(CANCELLED="1"), row(CANCELLED=None), row(DEP_DELAY="inf"),
            row(FL_DATE="invalid"), row(FL_DATE="2023-09-01"),
        ], dtype="string")
        clean, counts = M["clean_chunk"](frame, 1)
        features = M["engineer"](clean)
        self.assertEqual(clean.DELAYED_15.tolist(), [0, 1, 0, 1])
        self.assertTrue(pd.isna(clean.iloc[2].SCHEDULED_DEP_MINUTES))
        self.assertEqual(counts["DISTANCE_invalid_to_missing"], 1)
        self.assertEqual(features.iloc[2].SCHEDULED_DEP_MINUTES_MISSING, 1)
        self.assertEqual(features.iloc[2].SCHEDULED_DEP_MINUTES_SIN, 0)
        self.assertEqual(features.iloc[2].SCHEDULED_DEP_MINUTES_COS, 0)
        self.assertEqual(features.iloc[3].SCHEDULED_DEP_MINUTES_COS, 1)
        self.assertLess(abs(features.iloc[0].SCHEDULED_DEP_MINUTES_SIN - features.iloc[1].SCHEDULED_DEP_MINUTES_SIN), .01)
        self.assertEqual(clean.iloc[0].AIRLINE_CODE, "UA")
        self.assertEqual(sum(v for k, v in counts.items() if k.startswith("excluded_")) + len(clean), len(frame))

    def test_train_only_fit_alignment_reload_and_chunk_invariance(self):
        records = [
            row(FL_DATE="2023-08-31", AIRLINE_CODE="ZZ", DISTANCE="100000"),
            row(FL_DATE="2019-01-01", DISTANCE="100", CRS_ELAPSED_TIME="60", DEP_DELAY="-5"),
            row(FL_DATE="2022-12-31", AIRLINE_CODE="ZZ", DISTANCE="99999", CRS_ELAPSED_TIME=None),
            row(FL_DATE="2021-12-31", DISTANCE=None, AIRLINE_CODE=None),
            row(FL_DATE="2020-01-01", AIRLINE_CODE="DL", DISTANCE="300", CRS_ELAPSED_TIME="120"),
            row(FL_DATE="2022-01-01", DISTANCE=None),
            row(FL_DATE="2023-01-01", DISTANCE=None, CRS_DEP_TIME=None),
            row(CANCELLED="1"), row(FL_DATE="2018-12-31"), row(DEP_DELAY=None),
        ]
        with tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as temporary:
            folder = Path(temporary)
            source = folder / "source.csv"
            pd.DataFrame(records).to_csv(source, index=False)
            original = source.read_bytes()
            runs = []
            for chunk_size in (2, 5):
                destination = folder / f"run-{chunk_size}"
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = M["run"](source, destination, chunk_size=chunk_size)
                self.assertEqual(summary["counts"]["retained_rows"], 7)
                self.assertEqual(summary["fitted_on_train"]["numeric"]["DISTANCE"]["median"], 200)
                self.assertEqual(summary["fitted_on_train"]["numeric"]["DISTANCE"]["mean"], 200)
                self.assertAlmostEqual(summary["fitted_on_train"]["numeric"]["DISTANCE"]["scale"], np.sqrt(20000 / 3))
                self.assertEqual(summary["splits"]["validation"]["unknown_category_rows"]["AIRLINE_CODE"], 1)
                self.assertFalse((destination / "datasets/raw").exists())
                self.assertEqual(source.read_bytes(), original)
                transformer = joblib.load(destination / "artifacts/preprocessor.joblib")
                contract = json.loads((destination / "outputs/feature_contract.json").read_text(encoding="utf-8"))
                self.assertTrue(set(contract["model_input_features"]).isdisjoint(["FL_DATE", "SOURCE_ROW_NUMBER", "DEP_DELAY", "CANCELLED", "DELAYED_15"]))
                for name, expected_count in (("train", 3), ("validation", 2), ("test", 2)):
                    prepared = pd.read_csv(destination / "datasets/prepared" / f"{name}.csv")
                    x = sparse.load_npz(destination / "datasets/model-ready" / f"X_{name}.npz")
                    y = np.load(destination / "datasets/model-ready" / f"y_{name}.npy", allow_pickle=False)
                    self.assertEqual(x.shape, (expected_count, len(contract["output_feature_names"])))
                    self.assertEqual(x.dtype, np.float32)
                    self.assertTrue(np.isfinite(x.data).all())
                    self.assertTrue(prepared.FL_DATE.is_monotonic_increasing)
                    np.testing.assert_array_equal(y, prepared.DELAYED_15.to_numpy())
                    for record_id, target in zip(prepared.SOURCE_ROW_NUMBER, y):
                        self.assertEqual(target, int(float(records[record_id - 1]["DEP_DELAY"]) >= 15))
                    restored = sparse.csr_matrix(transformer.transform(M["model_input"](prepared)), dtype=np.float32)
                    np.testing.assert_allclose(x.toarray(), restored.toarray(), atol=1e-7)
                runs.append(destination)
            for name in M["SPLITS"]:
                left = sparse.load_npz(runs[0] / "datasets/model-ready" / f"X_{name}.npz")
                right = sparse.load_npz(runs[1] / "datasets/model-ready" / f"X_{name}.npz")
                np.testing.assert_array_equal(left.toarray(), right.toarray())

    def test_failed_run_preserves_previous_outputs(self):
        for missing_train in (False, True):
            with self.subTest(missing_train=missing_train), tempfile.TemporaryDirectory(prefix=".test-", dir=ROOT) as temporary:
                folder = Path(temporary)
                source = folder / "source.csv"
                pd.DataFrame([row(DISTANCE=None if missing_train else "100")]).to_csv(source, index=False)
                report = folder / "run/outputs/e1-data-preparation-report.md"
                report.parent.mkdir(parents=True)
                report.write_text("previous result", encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    M["run"](source, folder / "run", chunk_size=2)
                self.assertEqual(report.read_text(encoding="utf-8"), "previous result")
                self.assertFalse((folder / "run/datasets/model-ready/X_train.npz").exists())


if __name__ == "__main__":
    unittest.main()
