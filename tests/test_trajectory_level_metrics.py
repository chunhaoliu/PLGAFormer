import unittest

import numpy as np


class TrajectoryLevelMetricsTests(unittest.TestCase):
    def test_bootstrap_ci_and_paired_test_use_trajectory_pairs(self):
        from utils.trajectory_metrics import bootstrap_mean_ci, paired_permutation_test

        baseline = np.asarray([10.0, 12.0, 14.0, 16.0])
        candidate = baseline - 2.0
        ci = bootstrap_mean_ci(candidate, n_resamples=1000, seed=3)
        comparison = paired_permutation_test(candidate, baseline, n_resamples=2000, seed=4)

        self.assertEqual(ci["n"], 4)
        self.assertLessEqual(ci["ci_low"], ci["mean"])
        self.assertGreaterEqual(ci["ci_high"], ci["mean"])
        self.assertEqual(comparison["n_pairs"], 4)
        self.assertAlmostEqual(comparison["mean_difference"], -2.0)
        self.assertAlmostEqual(comparison["relative_improvement_percent"], 100.0 * 2.0 / 13.0)

    def test_holm_adjustment_controls_familywise_error(self):
        from utils.trajectory_metrics import holm_adjust

        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)

    def test_aggregate_window_metrics_by_trajectory(self):
        from utils.trajectory_metrics import aggregate_window_metrics_by_trajectory

        y_true = np.zeros((3, 2, 3), dtype=np.float32)
        y_pred = np.zeros_like(y_true)
        y_pred[0, :, 0] = 1.0
        y_pred[1, :, 0] = 3.0
        y_pred[2, :, 0] = 2.0
        trajectory_ids = np.asarray([5, 5, 9], dtype=np.int64)

        report = aggregate_window_metrics_by_trajectory(y_pred, y_true, trajectory_ids)

        self.assertEqual(report["trajectory_count"], 2)
        self.assertAlmostEqual(report["trajectories"][5]["ade"], 2.0)
        self.assertAlmostEqual(report["trajectories"][5]["fde"], 2.0)
        self.assertAlmostEqual(report["mean_ade"], 2.0)
        self.assertAlmostEqual(report["mean_fde"], 2.0)
        self.assertEqual(report["ade_ci"]["n"], 2)

    def test_aggregate_requires_matching_lengths(self):
        from utils.trajectory_metrics import aggregate_window_metrics_by_trajectory

        with self.assertRaisesRegex(ValueError, "trajectory_ids"):
            aggregate_window_metrics_by_trajectory(
                np.zeros((2, 3, 3), dtype=np.float32),
                np.zeros((2, 3, 3), dtype=np.float32),
                np.asarray([1], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
