import unittest

import numpy as np


class PITRadarDatasetAuditTests(unittest.TestCase):
    def test_paired_window_integrity_detects_target_corruption(self):
        from data_generation.pit_radar_protocol import assemble_pit_radar_dataset
        from scripts.validate_pit_radar_dataset import _paired_window_integrity

        rng = np.random.default_rng(13)
        clean = rng.normal(size=(9, 20, 6))
        clean[:, :, 0] += 6_438_000.0
        tracked = clean + 0.1
        radar = rng.normal(size=(9, 20, 3))
        labels = np.asarray(["longitudinal", "turning", "weaving"] * 3)
        payload = assemble_pit_radar_dataset(
            clean,
            tracked,
            radar,
            labels,
            seq_len=4,
            pred_len=4,
            tracking_burn_in_steps=2,
            train_origins_per_trajectory=3,
            eval_window_stride=2,
        )
        self.assertTrue(_paired_window_integrity(payload)["passed"])
        payload["y_test"][0, 0, 0] += 1.0
        self.assertFalse(_paired_window_integrity(payload)["passed"])


if __name__ == "__main__":
    unittest.main()
