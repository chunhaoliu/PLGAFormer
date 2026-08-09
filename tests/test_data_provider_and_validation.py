import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.trajectory_protocol import TRAJECTORY_LEVEL_PROTOCOL, LEGACY_WINDOW_PROTOCOL


def _write_npz(path: Path, *, protocol=TRAJECTORY_LEVEL_PROTOCOL, overlap=False, bad_height=False):
    seq_len = 4
    pred_len = 3
    x_train = np.zeros((2, seq_len, 6), dtype=np.float32)
    x_val = np.zeros((1, seq_len, 6), dtype=np.float32)
    x_test = np.zeros((2, seq_len, 6), dtype=np.float32)
    y_train = np.zeros((2, pred_len, 3), dtype=np.float32)
    y_val = np.zeros((1, pred_len, 3), dtype=np.float32)
    y_test = np.zeros((2, pred_len, 3), dtype=np.float32)

    earth_radius = 6_378_000.0
    for arr in [x_train, x_val, x_test]:
        arr[:, :, 0] = earth_radius + 70_000.0
        arr[:, :, 3] = 6_000.0
    for arr in [y_train, y_val, y_test]:
        arr[:, :, 0] = earth_radius + 65_000.0
    if bad_height:
        y_test[0, :, 0] = earth_radius - 100.0

    train_ids = np.asarray([10, 11], dtype=np.int64)
    val_ids = np.asarray([20], dtype=np.int64)
    test_ids = np.asarray([11, 31] if overlap else [30, 31], dtype=np.int64)

    np.savez(
        path,
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        dataset_protocol=np.asarray(protocol),
        seq_len=np.asarray(seq_len),
        pred_len=np.asarray(pred_len),
        sampling_interval_s=np.asarray(1.0),
        points_per_trajectory=np.asarray(1000),
        trajectory_duration_s=np.asarray(999.0),
        input_duration_s=np.asarray(float(seq_len)),
        prediction_duration_s=np.asarray(float(pred_len)),
        maneuver_taxonomy=np.asarray(["longitudinal", "turning", "weaving"]),
        trajectory_labels=np.asarray(["longitudinal", "turning", "weaving", "turning", "weaving"]),
        maneuver_labels_train=np.asarray(["longitudinal", "turning"]),
        maneuver_labels_val=np.asarray(["weaving"]),
        maneuver_labels_test=np.asarray(["turning", "weaving"]),
        trajectory_ids_train=train_ids,
        trajectory_ids_val=val_ids,
        trajectory_ids_test=test_ids,
        window_starts_train=np.asarray([0, 1], dtype=np.int64),
        window_starts_val=np.asarray([0], dtype=np.int64),
        window_starts_test=np.asarray([0, 1], dtype=np.int64),
    )


class DataProviderAndValidationTests(unittest.TestCase):
    def test_strict_loader_rejects_legacy_window_protocol(self):
        from data_provider.hgv_data import load_hgv_dataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hgv_trajectory_dataset.npz"
            _write_npz(path, protocol=LEGACY_WINDOW_PROTOCOL)

            with self.assertRaisesRegex(ValueError, "trajectory_level_v1"):
                load_hgv_dataset(path, require_trajectory_level=True)

    def test_strict_loader_returns_split_metadata_for_valid_protocol(self):
        from data_provider.hgv_data import load_hgv_dataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hgv_trajectory_dataset.npz"
            _write_npz(path)

            bundle = load_hgv_dataset(path, require_trajectory_level=True)

            self.assertEqual(bundle.protocol, TRAJECTORY_LEVEL_PROTOCOL)
            self.assertEqual(bundle.time_metadata["sampling_interval_s"], 1.0)
            self.assertEqual(bundle.time_metadata["prediction_duration_s"], 3.0)
            self.assertTrue(bundle.split_report["is_disjoint"])
            self.assertEqual(bundle.splits["train"].trajectory_ids.tolist(), [10, 11])
            self.assertEqual(bundle.splits["test"].maneuver_labels.tolist(), ["turning", "weaving"])

    def test_dataset_quality_report_detects_overlap_and_physical_violation(self):
        from data_provider.validation import validate_hgv_dataset

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hgv_trajectory_dataset.npz"
            _write_npz(path, overlap=True, bad_height=True)

            report = validate_hgv_dataset(path)

            self.assertFalse(report["passed"])
            self.assertFalse(report["split"]["is_disjoint"])
            self.assertTrue(report["taxonomy"]["uses_expected_three_class_taxonomy"])
            self.assertGreater(report["physical"]["test"]["height_below_min"], 0)


if __name__ == "__main__":
    unittest.main()
