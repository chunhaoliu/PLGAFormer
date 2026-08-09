import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TrajectoryProtocolTests(unittest.TestCase):
    def test_split_trajectory_ids_keeps_splits_disjoint(self):
        from utils.trajectory_protocol import split_trajectory_ids, validate_disjoint_trajectory_splits

        labels = np.array(
            [
                "longitudinal",
                "longitudinal",
                "longitudinal",
                "turning",
                "turning",
                "turning",
                "weaving",
                "weaving",
                "weaving",
            ]
        )

        splits = split_trajectory_ids(labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7)
        report = validate_disjoint_trajectory_splits(
            splits["train"],
            splits["val"],
            splits["test"],
        )

        all_ids = np.concatenate([splits["train"], splits["val"], splits["test"]])
        self.assertEqual(sorted(all_ids.tolist()), list(range(len(labels))))
        self.assertTrue(report["is_disjoint"])
        self.assertEqual(report["train_val_overlap"], [])
        self.assertEqual(report["train_test_overlap"], [])
        self.assertEqual(report["val_test_overlap"], [])

    def test_build_windows_preserves_source_trajectory_metadata(self):
        from utils.trajectory_protocol import build_windows_for_trajectories

        trajectories = [
            np.arange(12 * 6, dtype=np.float32).reshape(12, 6),
            np.arange(12 * 6, dtype=np.float32).reshape(12, 6) + 1000.0,
        ]
        labels = np.array(["turning", "weaving"])
        trajectory_ids = np.array([10, 25])

        windows = build_windows_for_trajectories(
            trajectories=trajectories,
            labels=labels,
            trajectory_ids=trajectory_ids,
            seq_len=3,
            pred_len=2,
            stride=2,
        )

        self.assertEqual(windows["X"].shape, (8, 3, 6))
        self.assertEqual(windows["y"].shape, (8, 2, 3))
        self.assertEqual(windows["trajectory_ids"].tolist(), [10, 10, 10, 10, 25, 25, 25, 25])
        self.assertEqual(windows["window_starts"].tolist(), [0, 2, 4, 6, 0, 2, 4, 6])
        self.assertEqual(windows["maneuver_labels"].tolist(), ["turning"] * 4 + ["weaving"] * 4)
        np.testing.assert_array_equal(windows["X"][0], trajectories[0][:3])
        np.testing.assert_array_equal(windows["y"][0], trajectories[0][3:5, :3])

    def test_time_metadata_converts_steps_to_physical_seconds(self):
        from utils.trajectory_protocol import dataset_time_metadata, horizon_label, steps_to_seconds

        data = {
            "sampling_interval_s": np.asarray(2.0),
            "seq_len": np.asarray(64),
            "pred_len": np.asarray(256),
            "window_stride": np.asarray(5),
            "points_per_trajectory": np.asarray(1000),
            "trajectory_duration_s": np.asarray(1998.0),
        }

        meta = dataset_time_metadata(data)

        self.assertEqual(steps_to_seconds(256, 2.0), 512.0)
        self.assertEqual(horizon_label(256, 2.0), "256 steps (512 s)")
        self.assertEqual(meta["input_duration_s"], 128.0)
        self.assertEqual(meta["prediction_duration_s"], 512.0)
        self.assertEqual(meta["points_per_trajectory"], 1000)

    def test_dataset_file_identity_records_protocol_hash_and_timing(self):
        from utils.trajectory_protocol import dataset_file_identity, sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.npz"
            np.savez(
                path,
                dataset_protocol=np.asarray("pit_aligned_radar_v1"),
                sampling_interval_s=np.asarray(2.0),
                seq_len=np.asarray(256),
                pred_len=np.asarray(256),
                window_stride=np.asarray(5),
            )

            identity = dataset_file_identity(path)

            self.assertEqual(identity["dataset_protocol"], "pit_aligned_radar_v1")
            self.assertEqual(identity["dataset_sha256"], sha256_file(path))
            self.assertEqual(identity["sampling_interval_s"], 2.0)
            self.assertEqual(identity["prediction_duration_s"], 512.0)

    def test_formal_scaler_paths_override_stale_cross_protocol_environment(self):
        from data_generation.data_paths import configure_protocol_scaler_paths

        with tempfile.TemporaryDirectory() as tmp:
            identity = {
                "dataset_path": str(Path(tmp) / "pit.npz"),
                "dataset_protocol": "pit_aligned_radar_v1",
                "dataset_sha256": "abcdef1234567890",
            }
            with patch.dict(
                "os.environ",
                {
                    "HGV_INPUT_SCALER_PATH": "legacy_input.joblib",
                    "HGV_OUTPUT_SCALER_PATH": "legacy_output.joblib",
                    "HGV_SCALER_SIGNATURE_PATH": "legacy_signature.json",
                },
            ):
                paths = configure_protocol_scaler_paths(
                    identity, force=True, namespace="exp1_sota"
                )

                self.assertIn("pit_aligned_radar_v1", str(paths["HGV_INPUT_SCALER_PATH"]))
                self.assertIn("abcdef1234567890", str(paths["HGV_OUTPUT_SCALER_PATH"]))
                self.assertIn("exp1_sota", str(paths["HGV_OUTPUT_SCALER_PATH"]))
                self.assertTrue(paths["HGV_INPUT_SCALER_PATH"].parent.is_dir())


if __name__ == "__main__":
    unittest.main()
