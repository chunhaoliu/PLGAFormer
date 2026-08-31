import unittest

import torch


class PLGAFormerInnovationTests(unittest.TestCase):
    def test_explicit_paper_lock_is_inference_only_and_exact(self):
        from models.plgaformer import PLGAFormerTransformer
        from utils.final_plgaformer import apply_paper_inference_policy

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_multi_head_output=True,
        )
        self.assertEqual(model.physics_prior_lock_steps, 0)
        self.assertEqual(model.physics_prior_decay_power, 2.0)

        policy = apply_paper_inference_policy(model)
        self.assertEqual(policy["lock_steps"], 64)
        self.assertEqual(model.physics_prior_lock_steps, 64)
        self.assertEqual(model.physics_prior_time_constant_s, 450.0)
        self.assertEqual(model.physics_prior_decay_power, 2.5)
        snapshot = model.physics_gate_snapshot()
        self.assertEqual(snapshot["physics_prior_lock_steps"], 64)
        self.assertEqual(snapshot["physics_prior_time_constant_s"], 450.0)
        self.assertEqual(snapshot["physics_prior_decay_power"], 2.5)

        future_mask = torch.ones(1, 66, 1)
        prior = torch.zeros(1, 66, 3)
        output = torch.ones_like(prior)
        decoder = torch.zeros(1, 66, model.d_model)
        weight = model._scheduled_prior_weight(
            prior, output, decoder, future_mask
        )

        self.assertTrue(torch.equal(weight[:, :64], torch.ones_like(weight[:, :64])))
        self.assertTrue(torch.all(weight[:, 64:] < 1.0))

    def test_prior_type_selects_spherical_control(self):
        from models.baseline_models import SphericalKinematicBaseline
        from models.plgaformer import PLGAFormerTransformer

        scaler_mean = torch.zeros(6)
        scaler_scale = torch.ones(6)
        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_multi_head_output=True,
            prior_type='spherical_kinematic',
            input_scaler_mean=scaler_mean,
            input_scaler_scale=scaler_scale,
            output_scaler_mean=scaler_mean[:3],
            output_scaler_scale=scaler_scale[:3],
        )

        self.assertIsInstance(model.physics_prior, SphericalKinematicBaseline)
        self.assertEqual(
            model.physics_gate_snapshot()["prior_type"],
            "spherical_kinematic",
        )

    def test_schedule_only_fusion_ignores_decoder_and_disagreement(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_multi_head_output=True,
            prior_blend_mode='schedule_only',
        )
        model.physics_prior_time_constant_s = 10.0
        future_mask = torch.ones(1, 8, 1)
        prior = torch.zeros(1, 8, 3)
        low_disagreement = torch.zeros_like(prior)
        high_disagreement = torch.full_like(prior, 4.0)
        decoder_a = torch.zeros(1, 8, model.d_model)
        decoder_b = torch.randn(1, 8, model.d_model)

        weight_a = model._scheduled_prior_weight(
            prior, low_disagreement, decoder_a, future_mask
        )
        weight_b = model._scheduled_prior_weight(
            prior, high_disagreement, decoder_b, future_mask
        )

        self.assertTrue(torch.equal(weight_a, weight_b))
        self.assertGreater(float(weight_a[0, 0, 0]), float(weight_a[0, -1, 0]))

    def test_direct_attention_without_priors_matches_backbone(self):
        from models.plgaformer import PLGAFormerTransformer

        common = dict(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
        torch.manual_seed(17)
        backbone = PLGAFormerTransformer(
            **common,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=False,
        ).eval()
        torch.manual_seed(17)
        direct_attention = PLGAFormerTransformer(
            **common,
            use_sparse_attention=True,
            use_physics_corrector=False,
            use_multi_head_output=False,
            attention_prior_mask=(False, False, False),
        ).eval()
        src = torch.randn(2, 6, 6)
        tgt = torch.randn(2, 5, 3)

        with torch.no_grad():
            expected = backbone(src, tgt)
            actual = direct_attention(src, tgt)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        snapshot = direct_attention.physics_gate_snapshot()
        self.assertEqual(snapshot["attention_residual_scale"], 1.0)
        self.assertEqual(snapshot["mean_temporal_gate"], 0.0)
        self.assertEqual(snapshot["mean_phase_gate"], 0.0)
        self.assertEqual(snapshot["mean_geometry_gate"], 0.0)

    def test_multi_head_decoder_preserves_base_projection_when_delta_is_zero(self):
        from models.plgaformer import PLGAFormerTransformer

        torch.manual_seed(7)
        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )
        model.eval()
        with torch.no_grad():
            model.trajectory_delta_scale.zero_()
            model.physics_fusion_gate[2].weight.zero_()
            model.physics_fusion_gate[2].bias.fill_(-100.0)

        self.assertTrue(hasattr(model, "trajectory_delta_head"))
        self.assertTrue(hasattr(model, "trajectory_delta_scale"))
        self.assertTrue(hasattr(model, "physics_fusion_gate"))

        for param in model.trajectory_delta_head.parameters():
            torch.nn.init.zeros_(param)
        with torch.no_grad():
            model.multihead_fusion_gate[-2].weight.zero_()
            model.multihead_fusion_gate[-2].bias.fill_(12.0)

        src = torch.randn(2, 5, 6)
        tgt = torch.randn(2, 4, 3)

        src_embedded = model.pos_encoder(model.input_embedding(src))
        memory = model.transformer_encoder(src_embedded)
        tgt_embedded = model.pos_encoder(model.output_embedding(tgt))
        decoded = model.transformer_decoder(tgt_embedded, memory)
        expected_base = model.output_projection(decoded)

        actual = model(src, tgt)

        self.assertTrue(torch.allclose(actual, expected_base, atol=1e-6))

    def test_kinematic_prior_extrapolates_future_zero_decoder_slots(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )

        tgt = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [2.0, 3.0, 4.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )

        prior, future_mask = model._build_kinematic_prior(tgt)

        expected_future = torch.tensor(
            [
                [2.5, 3.5, 4.5],
                [3.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(prior[0, :3], tgt[0, :3]))
        self.assertTrue(torch.allclose(prior[0, 3:], expected_future))
        self.assertTrue(torch.allclose(future_mask[0, :3], torch.zeros(3, 1)))
        self.assertTrue(torch.allclose(future_mask[0, 3:], torch.ones(2, 1)))

    def test_physics_corrector_is_identity_at_initialization(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )
        model.eval()
        output = torch.randn(2, 5, 3)
        decoder_output = torch.randn(2, 5, 32)
        prior = torch.randn(2, 5, 3)
        future_mask = torch.ones(2, 5, 1)

        corrected = model._apply_physics_correction(
            output, decoder_output, prior, future_mask
        )

        self.assertTrue(torch.equal(corrected, output))

    def test_physics_corrector_moves_prediction_toward_prior_when_enabled(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )
        model.eval()
        with torch.no_grad():
            model.physics_correction_scale.fill_(0.5)
            model.smoothness_gate[2].weight.zero_()
            model.smoothness_gate[2].bias.fill_(12.0)

        output = torch.zeros(1, 3, 3)
        prior = torch.full_like(output, 0.25)
        decoder_output = torch.zeros(1, 3, 32)
        future_mask = torch.ones(1, 3, 1)
        corrected = model._apply_physics_correction(
            output, decoder_output, prior, future_mask
        )

        self.assertTrue(torch.all(corrected > output))
        self.assertTrue(torch.all(corrected < prior))

    def test_physics_corrector_operates_in_prediction_space(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )

        output = torch.zeros(1, 4, model.output_dim)
        corrected = model._apply_physics_correction(
            output,
            torch.zeros(1, 4, model.d_model),
            torch.ones_like(output),
            torch.ones(1, 4, 1),
        )
        self.assertEqual(tuple(corrected.shape), tuple(output.shape))

    def test_physics_prior_pull_decays_when_prior_disagrees(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )

        future_mask = torch.ones(1, 1, 1)
        small_pull = model._confidence_weighted_prior_pull(
            torch.zeros(1, 1, 3),
            torch.full((1, 1, 3), 0.1),
            future_mask,
        )
        large_pull = model._confidence_weighted_prior_pull(
            torch.zeros(1, 1, 3),
            torch.full((1, 1, 3), 2.0),
            future_mask,
        )

        self.assertLess(float(large_pull.abs().mean()), float(small_pull.abs().mean()))

    def test_physics_corrector_changes_only_future_slots(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )

        with torch.no_grad():
            model.physics_correction_scale.fill_(0.5)
            model.smoothness_gate[2].weight.zero_()
            model.smoothness_gate[2].bias.fill_(12.0)
        output = torch.zeros(1, 3, 3)
        prior = torch.full_like(output, 0.25)
        future_mask = torch.tensor([[[0.0], [1.0], [1.0]]])
        corrected = model._apply_physics_correction(
            output,
            torch.zeros(1, 3, model.d_model),
            prior,
            future_mask,
        )

        self.assertTrue(torch.equal(corrected[:, :1], output[:, :1]))
        self.assertTrue(torch.all(corrected[:, 1:] > output[:, 1:]))

    def test_physics_correction_is_mode_consistent(self):
        from models.plgaformer import PLGAFormerTransformer

        torch.manual_seed(19)
        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=False,
        )
        with torch.no_grad():
            model.physics_correction_scale.fill_(0.5)

        output = torch.zeros(1, 3, 3)
        prior = torch.full_like(output, 0.25)
        decoder_output = torch.zeros(1, 3, model.d_model)
        future_mask = torch.ones(1, 3, 1)
        model.train()
        train_out = model._apply_physics_correction(
            output, decoder_output, prior, future_mask
        )
        model.eval()
        eval_out = model._apply_physics_correction(
            output, decoder_output, prior, future_mask
        )

        self.assertFalse(torch.allclose(train_out, output))
        self.assertTrue(torch.allclose(train_out, eval_out))

    def test_residual_innovation_scales_are_bounded(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=True,
        )

        self.assertLessEqual(float(model.physics_corrector_max_scale), 0.05)
        self.assertLessEqual(float(model.trajectory_delta_max_scale), 0.1)
        self.assertLessEqual(float(model.physics_prior_max_scale), 1.0)
        self.assertLessEqual(float(model.physics_context_max_scale), 0.2)

    def test_physics_prior_fusion_decays_with_forecast_horizon_at_initialization(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )

        snapshot = model.physics_gate_snapshot()
        self.assertGreater(float(snapshot["nominal_prior_weight_32"]), 0.98)
        self.assertGreater(
            float(snapshot["nominal_prior_weight_32"]),
            float(snapshot["nominal_prior_weight_128"]),
        )
        self.assertGreater(
            float(snapshot["nominal_prior_weight_128"]),
            float(snapshot["nominal_prior_weight_256"]),
        )
        self.assertGreater(float(snapshot["nominal_prior_weight_256"]), 0.70)
        self.assertLess(float(snapshot["nominal_prior_weight_256"]), 0.75)
        self.assertLess(float(snapshot["trajectory_delta_scale"]), 0.01)

    def test_prior_and_channel_residual_ablation_switches_are_independent(self):
        from models.plgaformer import PLGAFormerTransformer

        common = dict(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )
        torch.manual_seed(31)
        full = PLGAFormerTransformer(
            **common,
            use_prior_fusion=True,
            use_channel_residual=True,
        ).eval()
        variants = {}
        for name, prior, residual in (
            ("prior_only", True, False),
            ("residual_only", False, True),
            ("neither", False, False),
        ):
            model = PLGAFormerTransformer(
                **common,
                use_prior_fusion=prior,
                use_channel_residual=residual,
            ).eval()
            model.load_state_dict(full.state_dict())
            variants[name] = model

        with torch.no_grad():
            full.trajectory_delta_scale.fill_(1.0)
            for model in variants.values():
                model.trajectory_delta_scale.fill_(1.0)
            src = torch.randn(2, 8, 6)
            tgt = torch.cat((torch.randn(2, 4, 3), torch.zeros(2, 6, 3)), dim=1)
            full_output = full(src, tgt)
            prior_output = variants["prior_only"](src, tgt)
            residual_output = variants["residual_only"](src, tgt)
            neither_output = variants["neither"](src, tgt)

        self.assertFalse(torch.allclose(full_output, prior_output))
        self.assertFalse(torch.allclose(full_output, residual_output))
        self.assertFalse(torch.allclose(prior_output, neither_output))
        self.assertFalse(torch.allclose(residual_output, neither_output))
        self.assertTrue(full.physics_gate_snapshot()["use_prior_fusion"])
        self.assertTrue(full.physics_gate_snapshot()["use_channel_residual"])

    def test_default_component_switches_match_explicit_full_model(self):
        from models.plgaformer import PLGAFormerTransformer

        common = dict(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_multi_head_output=True,
        )
        torch.manual_seed(37)
        default_model = PLGAFormerTransformer(**common).eval()
        torch.manual_seed(37)
        explicit_model = PLGAFormerTransformer(
            **common,
            use_prior_fusion=True,
            use_channel_residual=True,
        ).eval()
        src = torch.randn(2, 8, 6)
        tgt = torch.cat((torch.randn(2, 4, 3), torch.zeros(2, 6, 3)), dim=1)

        with torch.no_grad():
            default_output = default_model(src, tgt)
            explicit_output = explicit_model(src, tgt)

        self.assertTrue(torch.equal(default_output, explicit_output))

    def test_scheduled_prior_weight_is_bounded_and_disagreement_aware(self):
        from models.plgaformer import PLGAFormerTransformer

        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )
        decoder = torch.zeros(1, 64, model.d_model)
        future_mask = torch.ones(1, 64, 1)
        base = torch.zeros(1, 64, 3)
        close_prior = torch.full_like(base, 0.05)
        far_prior = torch.full_like(base, 2.0)

        close_weight = model._scheduled_prior_weight(
            close_prior, base, decoder, future_mask
        )
        far_weight = model._scheduled_prior_weight(
            far_prior, base, decoder, future_mask
        )

        self.assertTrue(torch.all((close_weight >= 0.0) & (close_weight <= 1.0)))
        self.assertTrue(torch.all((far_weight >= 0.0) & (far_weight <= 1.0)))
        self.assertLess(float(far_weight.mean()), float(close_weight.mean()))
        self.assertGreater(float(close_weight[:, 0].mean()), float(close_weight[:, -1].mean()))

    def test_direct_physics_attention_can_reduce_to_standard_encoder(self):
        from models.plgaformer import PLGAFormerTransformer

        common = dict(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_physics_corrector=False,
            use_multi_head_output=False,
        )
        torch.manual_seed(13)
        baseline = PLGAFormerTransformer(
            **common,
            use_sparse_attention=False,
        ).eval()
        torch.manual_seed(13)
        model = PLGAFormerTransformer(
            **common,
            use_sparse_attention=True,
            attention_prior_mask=(False, False, False),
        ).eval()

        src = torch.randn(2, 6, 6)
        tgt = torch.randn(2, 5, 3)

        with torch.no_grad():
            expected = baseline(src, tgt)
            actual = model(src, tgt)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_direct_physics_encoder_has_negligible_parameter_overhead(self):
        from models.plgaformer import PLGAFormerTransformer

        torch.manual_seed(17)
        model = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=True,
            use_physics_corrector=False,
            use_multi_head_output=False,
        )

        torch.manual_seed(17)
        backbone = PLGAFormerTransformer(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            use_sparse_attention=False,
            use_physics_corrector=False,
            use_multi_head_output=False,
        )
        proposed_parameters = sum(parameter.numel() for parameter in model.parameters())
        backbone_parameters = sum(parameter.numel() for parameter in backbone.parameters())

        self.assertLess(proposed_parameters, 1.01 * backbone_parameters)

    def test_shared_backbone_initialization_is_stable_across_ablation_flags(self):
        from models.plgaformer import PLGAFormerTransformer

        kwargs = dict(
            input_dim=6,
            d_model=32,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )

        torch.manual_seed(23)
        full = PLGAFormerTransformer(
            **kwargs,
            use_sparse_attention=True,
            use_physics_corrector=True,
            use_multi_head_output=True,
        )
        torch.manual_seed(23)
        without_b = PLGAFormerTransformer(
            **kwargs,
            use_sparse_attention=True,
            use_physics_corrector=False,
            use_multi_head_output=True,
        )
        torch.manual_seed(23)
        without_a = PLGAFormerTransformer(
            **kwargs,
            use_sparse_attention=False,
            use_physics_corrector=True,
            use_multi_head_output=True,
        )

        self.assertTrue(torch.allclose(
            full.physics_encoder.layers[0].self_attn.q_proj.weight,
            without_b.physics_encoder.layers[0].self_attn.q_proj.weight,
        ))
        self.assertTrue(torch.allclose(
            full.transformer_decoder.layers[0].self_attn.in_proj_weight,
            without_a.transformer_decoder.layers[0].self_attn.in_proj_weight,
        ))
        self.assertTrue(torch.allclose(
            full.transformer_decoder.layers[0].self_attn.in_proj_weight,
            without_b.transformer_decoder.layers[0].self_attn.in_proj_weight,
        ))
        self.assertTrue(torch.allclose(full.output_projection.weight, without_b.output_projection.weight))

    def test_positional_encoding_uses_standard_transformer_magnitude(self):
        from models.plgaformer import PositionalEncoding

        pos = PositionalEncoding(d_model=32, dropout=0.0)

        self.assertGreater(float(pos.pe.abs().max()), 0.9)


if __name__ == "__main__":
    unittest.main()
