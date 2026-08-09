import importlib.util
import unittest
from pathlib import Path

import torch


class Exp4PhysicsMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        module_path = project_root / "experiments" / "exp4_physics_consistency" / "physics_consistency.py"
        spec = importlib.util.spec_from_file_location("exp4_physics_consistency", module_path)
        cls.assertIsNotNone = staticmethod(unittest.TestCase.assertIsNotNone)
        unittest.TestCase().assertIsNotNone(spec)
        unittest.TestCase().assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_trajectory_smoothness_reports_distinct_position_and_velocity_terms(self):
        radius = float(self.module.EARTH_RADIUS + 80_000.0)
        true_lon = torch.tensor([0.0, 0.0010, 0.0021, 0.0033, 0.0046, 0.0060])
        pred_lon = torch.tensor([0.0, 0.0011, 0.0023, 0.0038, 0.0052, 0.0068])
        lat = torch.zeros_like(true_lon)
        true = torch.stack([torch.full_like(true_lon, radius), true_lon, lat], dim=-1).unsqueeze(0)
        pred = torch.stack([torch.full_like(pred_lon, radius), pred_lon, lat], dim=-1).unsqueeze(0)

        smoothness = self.module.compute_trajectory_smoothness(pred, true)

        self.assertNotAlmostEqual(
            smoothness["position_smoothness_ratio"],
            smoothness["velocity_smoothness_ratio"],
            places=6,
        )
        self.assertGreater(smoothness["position_smoothness_pred"], 0.0)
        self.assertGreater(smoothness["velocity_smoothness_pred"], 0.0)

    def test_constraint_violations_include_zero_dynamic_relative_rmse_for_exact_match(self):
        radius = float(self.module.EARTH_RADIUS + 80_000.0)
        lon = torch.tensor([0.0, 0.0010, 0.0021, 0.0033, 0.0046, 0.0060])
        lat = torch.zeros_like(lon)
        traj = torch.stack([torch.full_like(lon, radius), lon, lat], dim=-1).unsqueeze(0)

        violations = self.module.compute_constraint_violations(traj, traj)

        self.assertAlmostEqual(violations["velocity_relative_rmse"], 0.0)
        self.assertAlmostEqual(violations["acceleration_relative_rmse"], 0.0)
        self.assertAlmostEqual(violations["velocity_violation_rate"], 0.0)
        self.assertAlmostEqual(violations["acceleration_violation_rate"], 0.0)

    def test_acceleration_relative_rmse_is_invariant_to_time_unit_scaling(self):
        radius = float(self.module.EARTH_RADIUS + 80_000.0)
        steps = torch.arange(8, dtype=torch.float32)
        true_lon = 0.0001 * steps ** 2
        pred_lon = 0.00011 * steps ** 2
        lat = torch.zeros_like(steps)
        true = torch.stack([torch.full_like(steps, radius), true_lon, lat], dim=-1).unsqueeze(0)
        pred = torch.stack([torch.full_like(steps, radius), pred_lon, lat], dim=-1).unsqueeze(0)

        metric_1s = self.module.compute_constraint_violations(pred, true, sampling_interval_s=1.0)
        metric_2s = self.module.compute_constraint_violations(pred, true, sampling_interval_s=2.0)

        self.assertAlmostEqual(
            metric_1s["acceleration_relative_rmse"],
            metric_2s["acceleration_relative_rmse"],
            places=5,
        )

    def test_format_physics_metric_prints_rates_as_percent_only(self):
        self.assertEqual(self.module.format_physics_metric("velocity_violation_rate", 0.125), "12.50%")
        self.assertEqual(self.module.format_physics_metric("velocity_relative_rmse", 1.23456), "1.2346")


if __name__ == "__main__":
    unittest.main()
