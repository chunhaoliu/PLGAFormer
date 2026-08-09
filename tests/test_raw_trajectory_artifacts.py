import tempfile
import unittest
from pathlib import Path

import numpy as np


def _sample_trajectory(label: str, offset: float = 0.0) -> dict:
    t = np.arange(5, dtype=np.float64)
    return {
        "r": 6_378_000.0 + 60_000.0 + t + offset,
        "λ": 0.01 * t + offset,
        "φ": -0.02 * t + offset,
        "V": 6_000.0 - 2.0 * t,
        "γ": -0.01 * np.ones_like(t),
        "ψ": 0.02 * np.ones_like(t),
        "time": t,
        "alpha": 0.1 * np.ones_like(t),
        "bank": 0.2 * np.ones_like(t),
        "velocity": 6_000.0 - 2.0 * t,
        "maneuver_type": label,
    }


class RawTrajectoryArtifactTests(unittest.TestCase):
    def test_save_raw_trajectories_preserves_complete_sequences_and_metadata(self):
        from data_generation.raw_trajectory_io import (
            RAW_TRAJECTORY_PROTOCOL,
            load_raw_trajectory_arrays,
            save_raw_trajectories,
        )

        trajectories = [
            _sample_trajectory("longitudinal"),
            _sample_trajectory("turning", offset=0.1),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_hgv_trajectories.npz"
            save_raw_trajectories(
                trajectories,
                path,
                sampling_interval_s=1.0,
                points_per_trajectory=5,
                trajectory_duration_s=4.0,
                generation_seed=42,
            )

            raw = load_raw_trajectory_arrays(path)

        self.assertEqual(raw["raw_protocol"].item(), RAW_TRAJECTORY_PROTOCOL)
        self.assertEqual(raw["trajectories"].shape, (2, 5, 6))
        self.assertEqual(raw["time"].shape, (2, 5))
        self.assertEqual(raw["alpha"].shape, (2, 5))
        self.assertEqual(raw["bank"].shape, (2, 5))
        self.assertEqual(raw["initial_states"].shape, (2, 6))
        self.assertEqual(raw["trajectory_labels"].tolist(), ["longitudinal", "turning"])
        self.assertEqual(raw["state_feature_names"].tolist(), ["r", "lambda", "phi", "V", "gamma", "psi"])
        self.assertEqual(float(raw["sampling_interval_s"]), 1.0)
        self.assertEqual(int(raw["points_per_trajectory"]), 5)
        self.assertEqual(int(raw["generation_seed"]), 42)
        np.testing.assert_allclose(raw["initial_states"][0], raw["trajectories"][0, 0])

    def test_load_raw_trajectories_as_dicts_round_trips_for_existing_plotters(self):
        from data_generation.raw_trajectory_io import (
            load_raw_trajectories_as_dicts,
            save_raw_trajectories,
        )

        trajectories = [
            _sample_trajectory("weaving"),
            _sample_trajectory("turning", offset=0.2),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw_hgv_trajectories.npz"
            save_raw_trajectories(
                trajectories,
                path,
                sampling_interval_s=1.0,
                points_per_trajectory=5,
                trajectory_duration_s=4.0,
            )

            loaded = load_raw_trajectories_as_dicts(path)

        self.assertEqual([item["maneuver_type"] for item in loaded], ["weaving", "turning"])
        self.assertEqual(loaded[0]["r"].shape, (5,))
        self.assertEqual(loaded[0]["λ"].shape, (5,))
        self.assertEqual(loaded[0]["φ"].shape, (5,))
        self.assertEqual(loaded[0]["V"].shape, (5,))
        self.assertEqual(loaded[0]["γ"].shape, (5,))
        self.assertEqual(loaded[0]["ψ"].shape, (5,))
        np.testing.assert_allclose(loaded[0]["r"], trajectories[0]["r"])
        np.testing.assert_allclose(loaded[1]["bank"], trajectories[1]["bank"])


if __name__ == "__main__":
    unittest.main()
