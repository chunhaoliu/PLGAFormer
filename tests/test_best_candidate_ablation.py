import unittest


class BestCandidateAblationTests(unittest.TestCase):
    def test_ablation_checkpoint_tokens_are_unique_for_variants(self):
        from experiments.exp2_ablation.ablation_study import _safe_filename_token

        names = [
            "PLGAFormer (A+B+C)",
            "PLGAFormer w/o A",
            "PLGAFormer w/o B",
            "PLGAFormer w/o C",
        ]
        tokens = [_safe_filename_token(name) for name in names]

        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(all("/" not in token and "\\" not in token for token in tokens))

    def test_ablation_run_seed_is_shared_within_phase(self):
        from experiments.exp2_ablation.ablation_study import ablation_run_seed

        seed_a = ablation_run_seed(42, "best_candidate_ablation")
        seed_b = ablation_run_seed(42, "best_candidate_ablation")
        other_phase = ablation_run_seed(42, "other_phase")

        self.assertEqual(seed_a, seed_b)
        self.assertEqual(seed_a, other_phase)

    def test_parse_prediction_horizons_sorts_unique_values(self):
        from scripts.run_best_candidate_ablation import parse_prediction_horizons

        self.assertEqual(parse_prediction_horizons("128,32,64,64"), [32, 64, 128])
        self.assertEqual(parse_prediction_horizons("", fallback=96), [96])

    def test_build_ablation_model_configs_uses_candidate_hyperparameters(self):
        from scripts.run_best_candidate_ablation import build_ablation_model_configs

        candidate = {
            "learning_rate": 0.0005,
            "dropout": 0.15,
            "alpha": 0.0005,
            "warmup_epochs": 2,
            "gradient_clip_norm": 0.7,
        }

        configs = build_ablation_model_configs(candidate)

        self.assertEqual(list(configs.keys()), [
            "Transformer (baseline)",
            "PLGAFormer (A+B+C)",
            "PLGAFormer w/o A",
            "PLGAFormer w/o B",
            "PLGAFormer w/o C",
        ])
        self.assertEqual(configs["PLGAFormer (A+B+C)"]["dropout"], 0.15)
        self.assertEqual(configs["PLGAFormer (A+B+C)"]["physics_loss_weight"], 0.0005)
        self.assertFalse(configs["PLGAFormer w/o A"]["innovations"]["use_sparse_attention"])
        self.assertFalse(configs["PLGAFormer w/o B"]["innovations"]["use_physics_corrector"])
        self.assertEqual(configs["PLGAFormer w/o B"]["physics_loss_weight"], 0.0)
        self.assertFalse(configs["PLGAFormer w/o C"]["innovations"]["use_multi_head_output"])


if __name__ == "__main__":
    unittest.main()
