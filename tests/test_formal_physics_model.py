import math
import unittest

import numpy as np
import torch


class GeometryAwareLossTests(unittest.TestCase):
    def test_ecef_trajectory_loss_is_zero_for_identity_and_has_finite_gradients(self):
        from models.plgaformer import ECEFTrajectoryLoss

        criterion = ECEFTrajectoryLoss(
            scaler_mean=[6_430_000.0, 0.02, 0.4],
            scaler_scale=[6_000.0, 0.06, 0.12],
        )
        target = torch.randn(2, 8, 3)
        identical = target.clone().requires_grad_(True)
        zero_loss = criterion(identical, target)
        self.assertAlmostEqual(float(zero_loss.detach()), 0.0, places=8)

        pred = (target + 0.01).requires_grad_(True)
        loss = criterion(pred, target)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())


class FormalSimulatorTests(unittest.TestCase):
    def setUp(self):
        from data_generation.data_generator import DCBNN_HGV_Simulator

        self.simulator = DCBNN_HGV_Simulator()

    def test_standard_atmosphere_reference_values_are_plausible(self):
        rho0, a0 = self.simulator.atmospheric_model(0.0)
        rho50, a50 = self.simulator.atmospheric_model(50_000.0)

        self.assertAlmostEqual(rho0, 1.225, places=3)
        self.assertGreater(a0, 330.0)
        self.assertLess(rho50, 0.002)
        self.assertGreater(a50, 250.0)

    def test_cav_h_polynomial_is_used_without_bank_force_amplification(self):
        alpha = math.radians(15.0)
        cl_zero_bank, cd_zero_bank, cy_zero_bank = self.simulator.aerodynamic_coefficients(
            alpha, 0.0, 15.0
        )
        cl_banked, cd_banked, cy_banked = self.simulator.aerodynamic_coefficients(
            alpha, math.radians(25.0), 15.0
        )

        self.assertAlmostEqual(cl_zero_bank, cl_banked, places=12)
        self.assertAlmostEqual(cd_zero_bank, cd_banked, places=12)
        self.assertEqual(cy_zero_bank, 0.0)
        self.assertEqual(cy_banked, 0.0)

    def test_formal_dynamics_do_not_apply_legacy_radial_rate_clip(self):
        state = np.asarray(
            [self.simulator.R_earth + 70_000.0, 0.1, 0.2, 6_000.0, math.radians(10.0), 0.4]
        )
        derivative = self.simulator.hgv_dynamics(100.0, state, "turning")

        self.assertGreater(abs(derivative[0]), 200.0)
        self.assertAlmostEqual(derivative[0], state[3] * math.sin(state[4]), places=8)
        self.assertTrue(np.all(np.isfinite(derivative)))

    def test_earth_rotation_changes_velocity_attitude_dynamics(self):
        state = np.asarray(
            [self.simulator.R_earth + 65_000.0, 0.2, 0.6, 5_800.0, -0.03, 1.0]
        )
        rotating = np.asarray(self.simulator.hgv_dynamics(200.0, state, "weaving"))
        omega = self.simulator.earth_rotation_rate
        self.simulator.earth_rotation_rate = 0.0
        nonrotating = np.asarray(self.simulator.hgv_dynamics(200.0, state, "weaving"))
        self.simulator.earth_rotation_rate = omega

        self.assertGreater(np.linalg.norm(rotating[3:] - nonrotating[3:]), 0.0)


class ExplicitPhysicsPriorTests(unittest.TestCase):
    def test_parameter_free_spherical_kinematic_baseline(self):
        from models.baseline_models import SphericalKinematicBaseline

        model = SphericalKinematicBaseline(
            input_scaler_mean=np.zeros(6, dtype=np.float32),
            input_scaler_scale=np.ones(6, dtype=np.float32),
            output_scaler_mean=np.zeros(3, dtype=np.float32),
            output_scaler_scale=np.ones(3, dtype=np.float32),
            sampling_interval_s=2.0,
        )
        source = torch.tensor(
            [[[6_448_000.0, 0.2, 0.3, 6_000.0, 0.05, 0.4]]], dtype=torch.float32
        )
        prediction = model(source, target_length=3)
        radial_rate = source[0, -1, 3] * torch.sin(source[0, -1, 4])

        self.assertEqual(tuple(prediction.shape), (1, 3, 3))
        self.assertAlmostEqual(
            float(prediction[0, 0, 0]),
            float(source[0, -1, 0] + 2.0 * radial_rate),
            delta=1.0,
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_attention_uses_explicit_physical_priors(self):
        from models.plgaformer import PhysicsAwareAttention

        attention = PhysicsAwareAttention(d_model=16, nhead=4, dropout=0.0)
        embedded = torch.randn(2, 5, 16)
        physical = torch.zeros(2, 5, 6)
        physical[..., 0] = 6_378_000.0 + torch.linspace(70_000.0, 60_000.0, 5)
        physical[..., 1] = torch.linspace(0.0, 0.02, 5)
        physical[..., 2] = torch.linspace(0.1, 0.11, 5)
        physical[..., 3] = torch.linspace(6_000.0, 5_500.0, 5)
        physical[..., 4] = -0.03
        physical[..., 5] = 0.4

        output = attention(embedded, raw_input=physical)

        self.assertEqual(output.shape, embedded.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(attention.last_bias_diagnostics)
        self.assertFalse(hasattr(attention, "phase_query"))
        self.assertFalse(hasattr(attention, "geo_proj"))

    def test_attention_prior_mask_is_auditable(self):
        from models.plgaformer import PhysicsAwareAttention

        attention = PhysicsAwareAttention(
            d_model=16,
            nhead=4,
            dropout=0.0,
            prior_mask=(True, False, False),
        )
        embedded = torch.randn(1, 4, 16)
        physical = torch.zeros(1, 4, 6)
        physical[..., 0] = 6_448_000.0
        physical[..., 3] = 6_000.0

        attention(embedded, raw_input=physical)
        diagnostics = attention.last_bias_diagnostics

        self.assertGreater(float(diagnostics["temporal_gate"]), 0.0)
        self.assertEqual(float(diagnostics["phase_gate"]), 0.0)
        self.assertEqual(float(diagnostics["geometry_gate"]), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the AMP regression test")
    def test_physics_attention_amp_backward_is_finite(self):
        """Large Earth constants must not overflow the FP16 autocast path."""
        from models.plgaformer import PhysicsAwareAttention

        attention = PhysicsAwareAttention(d_model=32, nhead=4, dropout=0.0).cuda()
        embedded = torch.randn(2, 16, 32, device="cuda", requires_grad=True)
        physical = torch.zeros(2, 16, 6, device="cuda")
        physical[..., 0] = 6_448_000.0
        physical[..., 1] = torch.linspace(0.0, 0.03, 16, device="cuda")
        physical[..., 2] = 0.2
        physical[..., 3] = 6_000.0
        physical[..., 4] = -0.03
        physical[..., 5] = 0.4

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = attention(embedded, raw_input=physical)
            loss = output.square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(embedded.grad).all())
        for name, parameter in attention.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_frozen_state_prior_uses_spherical_3dof_kinematics(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
            output_scaler_mean=np.zeros(3, dtype=np.float32),
            output_scaler_scale=np.ones(3, dtype=np.float32),
            sampling_interval_s=2.0,
        )
        source = torch.tensor(
            [[[6_448_000.0, 0.2, 0.3, 6_000.0, 0.05, 0.4]]], dtype=torch.float32
        )
        target = torch.zeros(1, 4, 3)
        target[:, 0, :] = source[:, -1, :3]

        prior, future_mask = model._build_kinematic_prior(target, source)
        expected_first_radius = source[0, -1, 0] + 2.0 * source[0, -1, 3] * torch.sin(source[0, -1, 4])

        self.assertAlmostEqual(float(prior[0, 1, 0]), float(expected_first_radius), delta=1.0)
        self.assertEqual(float(future_mask[0, 0, 0]), 0.0)
        self.assertEqual(float(future_mask[0, 1, 0]), 1.0)

    def test_physics_loss_is_dimensionless_and_zero_for_exact_target(self):
        from models.plgaformer import HGVPhysicsLoss

        mean = np.asarray([6_438_000.0, 0.0, 0.0], dtype=np.float32)
        scale = np.asarray([10_000.0, 0.1, 0.1], dtype=np.float32)
        criterion = HGVPhysicsLoss(alpha=1.0, beta=1.0, scaler_mean=mean, scaler_std=scale)
        target = torch.zeros(2, 5, 3)
        target[..., 0] = torch.linspace(0.0, -0.2, 5)

        loss = criterion(target, target, dt=1.0)

        self.assertAlmostEqual(float(loss), 0.0, places=7)
        self.assertTrue(torch.isfinite(loss))

    def test_physics_loss_is_robust_to_untrained_prediction_outliers(self):
        from models.plgaformer import HGVPhysicsLoss

        mean = np.asarray([6_438_000.0, 0.0, 0.0], dtype=np.float32)
        scale = np.asarray([10_000.0, 0.1, 0.1], dtype=np.float32)
        criterion = HGVPhysicsLoss(
            alpha=1e-4,
            beta=5e-4,
            scaler_mean=mean,
            scaler_std=scale,
        )
        target = torch.zeros(1, 32, 3)
        target[..., 0] = torch.linspace(0.0, -0.5, 32)
        prediction = torch.randn_like(target) * 3.0

        loss = criterion(prediction, target, dt=1.0, include_mse=False)

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 10.0)
        self.assertIn("acceleration_cauchy", criterion.last_components)


if __name__ == "__main__":
    unittest.main()
