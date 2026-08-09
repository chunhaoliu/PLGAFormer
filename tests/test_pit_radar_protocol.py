import unittest
import tempfile
from pathlib import Path

import numpy as np


class PITRadarProtocolTests(unittest.TestCase):
    def test_generation_checkpoint_restores_rng_and_arrays(self):
        from scripts.generate_pit_radar_dataset import (
            _load_checkpoint,
            _save_checkpoint,
        )

        rng = np.random.default_rng(23)
        first_draw = rng.normal(size=4)
        rows = [np.arange(30, dtype=np.float64).reshape(5, 6)]
        radar_rows = [np.arange(15, dtype=np.float64).reshape(5, 3)]
        config = {"formal": True, "resolved_count": 9}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            _save_checkpoint(
                path,
                clean_rows=rows,
                tracked_rows=[rows[0] + 1.0],
                radar_rows=radar_rows,
                labels=["longitudinal"],
                initial_states=[rows[0][0]],
                attempts=2,
                cumulative_elapsed_s=12.5,
                rng=rng,
                config=config,
            )
            expected_next_draw = rng.normal(size=4)
            resumed_rng = np.random.default_rng(999)
            restored = _load_checkpoint(
                path,
                rng=resumed_rng,
                expected_config=config,
            )

        np.testing.assert_array_equal(restored[0][0], rows[0])
        np.testing.assert_array_equal(restored[1][0], rows[0] + 1.0)
        self.assertEqual(restored[3], ["longitudinal"])
        self.assertEqual(restored[5], 2)
        self.assertEqual(restored[6], 12.5)
        np.testing.assert_array_equal(resumed_rng.normal(size=4), expected_next_draw)
        self.assertEqual(first_draw.shape, (4,))

    def test_zero_noise_radar_round_trip(self):
        from data_generation.pit_radar_protocol import (
            RadarConfig,
            ecef_to_radar_razel,
            radar_razel_to_ecef,
            spherical_to_ecef,
        )

        states = np.asarray(
            [
                [6_438_000.0, 0.00, 0.00],
                [6_448_000.0, 0.05, 0.02],
                [6_428_000.0, -0.04, 0.03],
            ]
        )
        radar = RadarConfig(
            range_std_m=0.0,
            azimuth_std_rad=0.0,
            elevation_std_rad=0.0,
        )
        truth = spherical_to_ecef(states)
        measurements = ecef_to_radar_razel(truth, radar)
        reconstructed = radar_razel_to_ecef(measurements, radar)
        np.testing.assert_allclose(reconstructed, truth, atol=1e-6)

    def test_radar_noise_is_seed_deterministic(self):
        from data_generation.pit_radar_protocol import simulate_radar_observations

        time = np.arange(20, dtype=np.float64)
        states = np.column_stack(
            [
                6_438_000.0 - 10.0 * time,
                0.001 * time,
                0.0002 * time,
                np.full_like(time, 6_000.0),
                np.full_like(time, -0.02),
                np.full_like(time, 1.7),
            ]
        )
        first = simulate_radar_observations(
            states,
            rng=np.random.default_rng(17),
        )
        second = simulate_radar_observations(
            states,
            rng=np.random.default_rng(17),
        )
        np.testing.assert_array_equal(first["noisy_razel"], second["noisy_razel"])

    def test_filter_is_strictly_causal(self):
        from data_generation.pit_radar_protocol import causal_constant_velocity_filter

        time = np.arange(12, dtype=np.float64)
        measurements = np.column_stack(
            [1000.0 + 20.0 * time, 2000.0 - 5.0 * time, 3000.0 + time]
        )
        covariances = np.repeat(np.eye(3)[None, :, :] * 100.0, len(time), axis=0)
        changed = measurements.copy()
        changed[8:] += 1_000_000.0
        original_position, original_velocity = causal_constant_velocity_filter(
            measurements,
            covariances,
            sampling_interval_s=2.0,
        )
        changed_position, changed_velocity = causal_constant_velocity_filter(
            changed,
            covariances,
            sampling_interval_s=2.0,
        )
        np.testing.assert_array_equal(original_position[:8], changed_position[:8])
        np.testing.assert_array_equal(original_velocity[:8], changed_velocity[:8])

    def test_paired_windows_use_tracked_input_and_clean_future_target(self):
        from data_generation.pit_radar_protocol import build_paired_windows

        clean = np.arange(2 * 12 * 6, dtype=np.float64).reshape(2, 12, 6)
        tracked = clean + 10_000.0
        windows = build_paired_windows(
            tracked,
            clean,
            ["longitudinal", "turning"],
            [4, 9],
            seq_len=3,
            pred_len=2,
            starts=np.asarray([1, 5]),
        )
        np.testing.assert_array_equal(windows["X"][0], tracked[0, 1:4])
        np.testing.assert_array_equal(windows["y"][0], clean[0, 4:6, :3])
        self.assertEqual(windows["trajectory_ids"].tolist(), [4, 4, 9, 9])
        self.assertEqual(windows["window_starts"].tolist(), [1, 5, 1, 5])

    def test_assembled_dataset_is_split_before_windowing(self):
        from data_generation.pit_radar_protocol import (
            PIT_RADAR_PROTOCOL,
            assemble_pit_radar_dataset,
        )
        from utils.trajectory_protocol import validate_npz_trajectory_splits

        rng = np.random.default_rng(3)
        clean = rng.normal(size=(9, 20, 6))
        clean[:, :, 0] += 6_438_000.0
        tracked = clean + rng.normal(scale=0.01, size=clean.shape)
        radar = rng.normal(size=(9, 20, 3))
        labels = np.asarray(["longitudinal", "turning", "weaving"] * 3)
        payload = assemble_pit_radar_dataset(
            clean,
            tracked,
            radar,
            labels,
            sampling_interval_s=2.0,
            seq_len=4,
            pred_len=4,
            tracking_burn_in_steps=2,
            train_origins_per_trajectory=3,
            eval_window_stride=2,
        )

        self.assertEqual(payload["dataset_protocol"].item(), PIT_RADAR_PROTOCOL)
        self.assertTrue(validate_npz_trajectory_splits(payload)["is_disjoint"])
        self.assertEqual(payload["X_train"].shape, (9, 4, 6))
        self.assertEqual(payload["y_train"].shape, (9, 4, 3))
        self.assertEqual(set(payload["physical_horizon_s"].tolist()), {64, 128, 256, 512})
        for split in ("train", "val", "test"):
            complete_ids = set(payload[f"complete_trajectory_ids_{split}"].tolist())
            window_ids = set(payload[f"trajectory_ids_{split}"].tolist())
            self.assertEqual(complete_ids, window_ids)


if __name__ == "__main__":
    unittest.main()
