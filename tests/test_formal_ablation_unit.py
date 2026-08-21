import tempfile
import unittest
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


class FormalAblationUnitTests(unittest.TestCase):
    def test_select_model_configs_by_key_preserves_requested_order(self):
        from scripts.run_formal_ablation_unit import select_model_configs_by_key

        selected = select_model_configs_by_key(
            {
                "Transformer (baseline)": {"model_type": "baseline"},
                "PLGAFormer (A+B+C)": {"model_type": "plgaformer"},
                "PLGAFormer w/o C": {"model_type": "plgaformer"},
            },
            "full,baseline",
        )

        self.assertEqual(list(selected.keys()), ["PLGAFormer (A+B+C)", "Transformer (baseline)"])

    def test_current_structural_aliases_select_locked_variants(self):
        from scripts.run_formal_ablation_unit import select_model_configs_by_key

        selected = select_model_configs_by_key(
            {
                "Transformer (baseline)": {"model_type": "baseline"},
                "PLGAFormer prior fusion only": {"model_type": "plgaformer"},
                "PLGAFormer (proposed)": {"model_type": "plgaformer"},
            },
            "proposed,prior_only,baseline",
        )

        self.assertEqual(
            list(selected),
            [
                "PLGAFormer (proposed)",
                "PLGAFormer prior fusion only",
                "Transformer (baseline)",
            ],
        )

    def test_final_mechanism_aliases_select_controls(self):
        from experiments.exp2_ablation.ablation_study import (
            ABLATION_FINAL_MECHANISM_CONTROLS,
        )
        from scripts.run_formal_ablation_unit import select_model_configs_by_key

        selected = select_model_configs_by_key(
            ABLATION_FINAL_MECHANISM_CONTROLS,
            "schedule_only,spherical_prior",
        )

        self.assertEqual(
            list(selected),
            [
                "PLGAFormer rotating-prior schedule only",
                "PLGAFormer spherical-prior fusion",
            ],
        )

    def test_append_metric_rows_writes_header_once(self):
        from scripts.run_formal_ablation_unit import append_metric_rows

        rows = [
            {
                "run_id": "r1",
                "seed": 42,
                "model_key": "full",
                "model": "PLGAFormer (A+B+C)",
                "horizon": 64,
                "metric": "mse",
                "value": 0.1,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formal_runs.csv"
            append_metric_rows(path, rows)
            append_metric_rows(path, rows)

            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0].split(",")[:3], ["run_id", "seed", "model_key"])

    def test_nested_trajectory_metrics_are_not_scalar_metrics(self):
        from scripts.run_formal_ablation_unit import _is_scalar_metric

        self.assertTrue(_is_scalar_metric(1.5))
        self.assertFalse(_is_scalar_metric({"1": {"ade": 10.0}}))
        self.assertFalse(_is_scalar_metric([1.0, 2.0]))

    def test_completed_units_require_all_horizon_ade_fde_rows(self):
        from scripts.run_formal_ablation_unit import append_metric_rows, completed_units

        args = Namespace(
            subset_ratio=1.0,
            epochs=10,
            batch_size=128,
            prediction_horizons="32,64,128,256",
            train_supervision_protocol="source_context_pred_window",
            eval_protocol="source_context_decoder",
            label_len=128,
            amp=False,
            amp_dtype="bfloat16",
            cache_physics_prior=True,
            physics_prior_cache_batch_size=512,
        )
        rows = []
        for horizon in (32, 64, 128, 256):
            for metric in ("ade", "fde"):
                rows.append(
                    {
                        "run_id": "x",
                        "seed": 42,
                        "model_key": "prior_only",
                        "model": "PLGAFormer prior fusion only",
                        "horizon": horizon,
                        "metric": metric,
                        "value": 1.0,
                        "timestamp": "20260725",
                        "subset_ratio": 1.0,
                        "epochs": 10,
                        "batch_size": 128,
                        "prediction_horizons": "32,64,128,256",
                        "train_supervision_protocol": "source_context_pred_window",
                        "eval_protocol": "source_context_decoder",
                        "label_len": 128,
                        "mixed_precision": False,
                        "mixed_precision_dtype": "float32",
                        "cache_physics_prior": True,
                        "physics_prior_cache_batch_size": 512,
                        "candidate_id": "locked_current_design",
                        "phase": "phase1_structural",
                        "checkpoint": "checkpoint.pth",
                        "source": "formal_ablation_training",
                        "source_run_signature": "",
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formal_runs.csv"
            append_metric_rows(path, rows)
            self.assertIn(
                ("phase1_structural", 42, "prior_only"),
                completed_units(path, args),
            )

    def test_formal_defaults_use_unified_fifty_epoch_fp32_protocol(self):
        from scripts.run_formal_ablation_unit import parse_args

        args = parse_args([])
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertTrue(args.cache_physics_prior)
        self.assertEqual(args.physics_prior_cache_batch_size, 512)
        self.assertFalse(args.reuse_exp1_controls)

    def test_pit_sampling_interval_updates_model_and_physics_dt(self):
        from scripts.run_formal_ablation_unit import _configure_exp2, parse_args

        exp2 = SimpleNamespace(
            BASE_RANDOM_SEEDS=[],
            NUM_RUNS=0,
            RANDOM_SEEDS=[],
            TRAIN_CONFIG={},
            train_config={},
            PHYSICS_DT=1.0,
        )
        horizons = _configure_exp2(
            exp2,
            parse_args([]),
            {"candidate_id": "test"},
            {"sampling_interval_s": 2.0},
        )

        self.assertEqual(horizons, [32, 64, 128, 256])
        self.assertEqual(exp2.TRAIN_CONFIG["sampling_interval_s"], 2.0)
        self.assertEqual(exp2.train_config["sampling_interval_s"], 2.0)
        self.assertEqual(exp2.PHYSICS_DT, 2.0)

    def test_main_plga_checkpoint_is_not_reused_as_residual_proposed(self):
        from scripts.run_formal_ablation_unit import _exp1_control_name

        self.assertEqual(
            _exp1_control_name("phase1_structural", "baseline"),
            "Transformer (baseline)",
        )
        self.assertIsNone(
            _exp1_control_name("phase1_structural", "proposed")
        )
        self.assertIsNone(
            _exp1_control_name("phase1_structural", "prior_only")
        )

    def test_frozen_ablation_is_deduplicated_by_exact_source(self):
        from scripts.evaluate_formal_ablation_selection import (
            convert_to_serializable,
            find_existing_frozen_ablation,
        )
        import numpy as np

        self.assertEqual(
            convert_to_serializable({"value": np.float32(1.5), "ids": (1, 2)}),
            {"value": 1.5, "ids": [1, 2]},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_ablation_phase_seed42_model_1.json"
            path.write_text(
                json.dumps(
                    {
                        "test_evaluation_performed": True,
                        "source_selection_record_sha256": "source",
                        "checkpoint_sha256": "checkpoint",
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(
                find_existing_frozen_ablation(
                    tmp,
                    source_record_sha256="source",
                    checkpoint_sha256="checkpoint",
                )
            )
            self.assertIsNone(
                find_existing_frozen_ablation(
                    tmp,
                    source_record_sha256="changed",
                    checkpoint_sha256="checkpoint",
                )
            )


    def test_learned_only_capacity_control_disables_physics_paths(self):
        from experiments.mechanism_analysis.ablation_study import (
            ABLATION_LEARNED_ONLY_CAPACITY_CONTROL,
        )
        from scripts.run_formal_ablation_unit import select_model_configs_by_key

        selected = select_model_configs_by_key(
            ABLATION_LEARNED_ONLY_CAPACITY_CONTROL,
            "learned_only",
        )
        self.assertEqual(list(selected), ["PLGAFormer learned-only backbone"])
        innovations = next(iter(selected.values()))["innovations"]
        self.assertFalse(innovations["use_sparse_attention"])
        self.assertFalse(innovations["use_physics_corrector"])
        self.assertFalse(innovations["use_multi_head_output"])
        self.assertFalse(innovations["use_prior_fusion"])
        self.assertFalse(innovations["use_channel_residual"])

    def test_custom_output_dirs_must_be_provided_as_a_pair(self):
        from scripts.run_formal_ablation_unit import resolve_custom_output_dirs

        with self.assertRaisesRegex(ValueError, "must be provided together"):
            resolve_custom_output_dirs(
                Namespace(output_dir="results", checkpoint_dir=None)
            )
        with tempfile.TemporaryDirectory() as tmp:
            result = resolve_custom_output_dirs(
                Namespace(
                    output_dir=str(Path(tmp) / "results"),
                    checkpoint_dir=str(Path(tmp) / "checkpoints"),
                )
            )
            self.assertIsNotNone(result)
            self.assertTrue(result[0].is_dir())
            self.assertTrue(result[1].is_dir())

if __name__ == "__main__":
    unittest.main()
