import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FormalSotaUnitTests(unittest.TestCase):
    def test_main_results_matrix_is_exact_paper_mainline(self):
        from experiments.overall_prediction import main_results as exp1

        self.assertEqual(
            [
                (name, config["model_type"])
                for name, config in exp1.COMPARISON_MODELS.items()
            ],
            [
                ("Transformer (baseline)", "transformer"),
                ("PLGAFormer (proposed)", "plgaformer"),
                ("Spherical kinematics", "kinematic"),
                ("Rotating-Earth 3-DOF", "rotating_3dof"),
                ("DLinear", "dlinear"),
                ("PatchTST", "patchtst"),
                ("iTransformer", "itransformer"),
            ],
        )

        for function in (
            exp1.perform_statistical_tests,
            exp1.perform_trajectory_level_tests,
            exp1.print_statistical_summary,
        ):
            self.assertEqual(
                inspect.signature(function).parameters["baseline_name"].default,
                "Transformer (baseline)",
            )

    def test_default_formal_trainable_set_is_exact(self):
        from scripts.run_formal_sota_unit import parse_args, select_model_configs_by_key

        args = parse_args([])
        self.assertEqual(
            args.models,
            "transformer,plgaformer,dlinear,patchtst,itransformer",
        )
        selected = select_model_configs_by_key(args.models)
        self.assertEqual(
            [config["model_type"] for config in selected.values()],
            ["transformer", "plgaformer", "dlinear", "patchtst", "itransformer"],
        )

    def test_analytical_records_remain_selectable(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        selected = select_model_configs_by_key("kinematic,rotating_3dof")
        self.assertEqual(
            [config["model_type"] for config in selected.values()],
            ["kinematic", "rotating_3dof"],
        )

    def test_inactive_formal_keys_are_rejected(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        for model_key in (
            "baseline",
            "pit",
            "kalman",
            "af_ciln",
            "informer",
            "autoformer",
            "fedformer",
            "timesnet",
        ):
            with self.subTest(model_key=model_key):
                with self.assertRaisesRegex(ValueError, "Unknown active model key"):
                    select_model_configs_by_key(model_key)

    def test_formal_cli_does_not_expose_candidate_or_af_ciln_sources(self):
        from scripts.run_formal_sota_unit import parse_args

        args = parse_args([])
        self.assertFalse(hasattr(args, "candidate_file"))
        self.assertFalse(hasattr(args, "af_ciln_root"))
        for option in ("--candidate-file", "--af-ciln-root"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit):
                    parse_args([option, "unused"])

    def test_frozen_evaluator_uses_active_parser_and_frozen_defaults(self):
        from scripts.evaluate_formal_sota_selection import parse_args
        from utils.mainline_contract import ACTIVE_CONFIG_PATH

        args = parse_args([])
        self.assertEqual(
            args.models,
            "transformer,plgaformer,dlinear,patchtst,itransformer",
        )
        self.assertEqual(Path(args.config).resolve(), ACTIVE_CONFIG_PATH)
        self.assertFalse(hasattr(args, "candidate_file"))
        self.assertFalse(hasattr(args, "af_ciln_root"))
        self.assertEqual(args.workers, 0)
        self.assertFalse(args.amp)
        self.assertTrue(args.cache_physics_prior)
        for option in ("--candidate-file", "--af-ciln-root"):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit):
                    parse_args([option, "unused"])

    def test_frozen_evaluator_rejects_inactive_selection(self):
        from scripts.evaluate_formal_sota_selection import parse_args
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        args = parse_args(["--models", "af_ciln"])
        with self.assertRaisesRegex(ValueError, "Unknown active model key"):
            select_model_configs_by_key(args.models)

    def test_frozen_evaluator_rejects_plgaformer_source_flag_drift(self):
        from scripts.evaluate_formal_sota_selection import (
            _validate_source_plgaformer_config,
        )
        from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS

        source_config = dict(ACTIVE_PLGAFORMER_FLAGS)
        source_config.update(model_type="plgaformer")
        _validate_source_plgaformer_config(source_config)

        source_config["use_prior_fusion"] = False
        with self.assertRaisesRegex(ValueError, "use_prior_fusion"):
            _validate_source_plgaformer_config(source_config)

    def test_selection_only_runs_are_isolated_from_test_evidence(self):
        from scripts.run_formal_sota_unit import resolve_evidence_tier

        self.assertEqual(resolve_evidence_tier(1.0, True), "convergence_pilot")
        self.assertEqual(resolve_evidence_tier(1.0, False), "final")
        with self.assertRaisesRegex(ValueError, "subset_ratio"):
            resolve_evidence_tier(0.2, False)

    def test_plgaformer_record_config_uses_actual_model_values(self):
        from scripts.run_formal_sota_unit import (
            _resolved_plgaformer_record_config,
        )
        from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS

        actual = dict(ACTIVE_PLGAFORMER_FLAGS)
        actual["use_prior_fusion"] = False
        model = SimpleNamespace(**actual, use_adaptive_fusion=False, dropout=0.25)

        recorded = _resolved_plgaformer_record_config(
            model,
            {"model_type": "plgaformer", "use_prior_fusion": True, "dropout": 0.1},
        )

        self.assertFalse(recorded["use_prior_fusion"])
        self.assertFalse(recorded["use_adaptive_fusion"])
        self.assertEqual(recorded["dropout"], 0.25)

    def test_final_plgaformer_contract_rejects_actual_flag_drift(self):
        from scripts.run_formal_sota_unit import validate_final_plgaformer_contract
        from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS

        actual = dict(ACTIVE_PLGAFORMER_FLAGS)
        actual["use_prior_fusion"] = False
        model = SimpleNamespace(**actual)

        with self.assertRaisesRegex(ValueError, "use_prior_fusion"):
            validate_final_plgaformer_contract(model)

    def test_parser_uses_frozen_formal_runtime_defaults(self):
        from scripts.run_formal_sota_unit import parse_args

        args = parse_args([])
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.prediction_length, 256)
        self.assertEqual(args.prediction_horizons, "32,64,128,256")
        self.assertEqual(
            args.train_supervision_protocol, "source_context_pred_window"
        )
        self.assertEqual(args.eval_protocol, "source_context_decoder")
        self.assertEqual(args.eval_ar_seed_mode, "zero")
        self.assertEqual(args.label_len, 128)
        self.assertEqual(args.seeds, "42,123,456")
        self.assertEqual(args.learning_rate, 0.001)
        self.assertEqual(args.weight_decay, 0.00005)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.patience, 15)
        self.assertEqual(args.gradient_clip_norm, 1.0)
        self.assertEqual(args.workers, 0)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertTrue(args.cache_physics_prior)
        self.assertEqual(args.physics_prior_cache_batch_size, 512)

    def test_runtime_drift_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name == "data_generation" or name.startswith("data_generation."):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            if name == "experiments" or name.startswith("experiments."):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, r"prediction_length.*256"):
                run_formal_sota_units(
                    parse_args(["--prediction-length", "128"])
                )

        self.assertEqual(training_imports, [])

    def test_reduced_final_subset_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, r"subset_ratio.*1\.0"):
                run_formal_sota_units(parse_args(["--subset-ratio", "0.2"]))
        self.assertEqual(training_imports, [])

    def test_custom_config_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "config"):
                run_formal_sota_units(parse_args(["--config", "other.json"]))
        self.assertEqual(training_imports, [])

    def test_default_config_is_the_canonical_contract(self):
        from scripts.run_formal_sota_unit import parse_args
        from utils.mainline_contract import ACTIVE_CONFIG_PATH

        self.assertEqual(Path(parse_args([]).config).resolve(), ACTIVE_CONFIG_PATH)

    def test_default_paper_selection_excludes_isolated_baselines(self):
        from scripts.run_formal_sota_unit import parse_args, select_model_configs_by_key

        args = parse_args([])
        default_keys = {item.strip() for item in args.models.split(",")}
        self.assertNotIn("pit", default_keys)
        self.assertNotIn("af_ciln", default_keys)

        selected = select_model_configs_by_key("")
        selected_types = {config["model_type"] for config in selected.values()}
        self.assertNotIn("pit", selected_types)
        self.assertNotIn("af_ciln", selected_types)

    def test_all_model_selection_uses_every_available_registry_entry(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        available = {
            "Transformer (baseline)": {"model_type": "transformer"},
            "PLGAFormer (proposed)": {"model_type": "plgaformer"},
        }

        selected = select_model_configs_by_key("all", available)

        self.assertEqual(list(selected), list(available))

    def test_protocol_identity_records_step_and_second_horizons(self):
        from scripts.run_formal_sota_unit import build_protocol_identity, parse_args

        args = parse_args([])
        identity = build_protocol_identity(
            args,
            [32, 64, 128, 256],
            "signature",
            {
                "dataset_protocol": "hgv_multiregime_state_v2_1",
                "dataset_sha256": "abc",
                "sampling_interval_s": 2.0,
            },
        )

        self.assertEqual(identity["dataset_protocol"], "hgv_multiregime_state_v2_1")
        self.assertEqual(identity["prediction_horizons_s"], [64.0, 128.0, 256.0, 512.0])
        self.assertEqual(identity["mixed_precision_dtype"], "float32")
        self.assertTrue(identity["cache_physics_prior"])

    def test_selection_record_requires_exact_protocol_and_checkpoint(self):
        from scripts.run_formal_sota_unit import (
            build_protocol_identity,
            parse_args,
            selection_record_matches,
            sha256_file,
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pth"
            checkpoint.write_bytes(b"weights")
            args = parse_args(["--selection-only", "--epochs", "50"])
            identity = build_protocol_identity(args, [32, 64, 128, 256], "signature")
            payload = {
                "seed": 42,
                "model_key": "full",
                "evidence_tier": "convergence_pilot",
                "test_evaluation_performed": False,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "training_history": {
                    "epochs_completed": 50,
                    "best_epoch": 45,
                    "best_val_loss": 0.1,
                    "fixed_parameter_model": False,
                },
                "protocol_identity": identity,
            }

            self.assertTrue(
                selection_record_matches(
                    payload,
                    seed=42,
                    model_key="full",
                    expected_identity=identity,
                )
            )
            changed = dict(identity, batch_size=256)
            self.assertFalse(
                selection_record_matches(
                    payload,
                    seed=42,
                    model_key="full",
                    expected_identity=changed,
                )
            )
            checkpoint.write_bytes(b"tampered")
            self.assertFalse(
                selection_record_matches(
                    payload,
                    seed=42,
                    model_key="full",
                    expected_identity=identity,
                )
            )

    def test_select_model_configs_by_key_preserves_order(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        selected = select_model_configs_by_key("plgaformer,patchtst,transformer")

        self.assertEqual(list(selected.keys()), [
            "PLGAFormer (proposed)",
            "PatchTST",
            "Transformer (baseline)",
        ])
        self.assertEqual(selected["PatchTST"]["model_type"], "patchtst")

    def test_seeded_loader_rebuild_is_repeatable_without_reloading_dataset(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from experiments.exp1_sota import SOTA_comparison as exp1
        from scripts.run_formal_sota_unit import _rebuild_train_loader_for_seed

        dataset = TensorDataset(torch.arange(32))
        base = DataLoader(dataset, batch_size=8, shuffle=False)
        first = _rebuild_train_loader_for_seed(base, exp1, 42)
        second = _rebuild_train_loader_for_seed(base, exp1, 42)

        first_order = torch.cat([batch[0] for batch in first])
        second_order = torch.cat([batch[0] for batch in second])
        self.assertIs(first.dataset, dataset)
        self.assertTrue(torch.equal(first_order, second_order))

    def test_formal_main_table_model_set_is_available(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        selected = select_model_configs_by_key("all")

        self.assertEqual(
            [config["model_type"] for config in selected.values()],
            [
                "transformer",
                "plgaformer",
                "kinematic",
                "rotating_3dof",
                "dlinear",
                "patchtst",
                "itransformer",
            ],
        )

    def test_append_metric_rows_writes_header_once(self):
        from scripts.run_formal_sota_unit import append_metric_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "formal_sota_runs.csv"
            row = {
                "run_id": "r1",
                "seed": 42,
                "model_key": "autoformer",
                "model": "Autoformer",
                "horizon": 32,
                "metric": "mse",
                "value": 0.1,
                "timestamp": "1",
            }

            append_metric_rows(path, [row])
            append_metric_rows(path, [dict(row, run_id="r2", timestamp="2")])

            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0][:4], ["run_id", "seed", "model_key", "model"])
            self.assertEqual(len(rows), 3)

    def test_runner_shaped_records_preserve_compatible_display_names_and_keys(self):
        from scripts.run_formal_sota_unit import (
            MODEL_KEY_TO_NAME,
            _flatten_metric_rows,
            parse_args,
        )

        args = parse_args([])
        for model_type, expected_key, expected_name in (
            ("transformer", "baseline", "Transformer (baseline)"),
            ("plgaformer", "full", "PLGAFormer (proposed)"),
        ):
            rows = _flatten_metric_rows(
                run_id="run",
                timestamp="timestamp",
                seed=42,
                model_name=MODEL_KEY_TO_NAME[model_type],
                model_config={"model_type": model_type},
                eval_results={32: {"rmse": 1.0}},
                args=args,
                checkpoint=Path("model.pth"),
            )
            self.assertEqual(rows[0]["model"], expected_name)
            self.assertEqual(rows[0]["model_key"], expected_key)

    def test_all_selection_applies_analytical_anchor_policies(self):
        from scripts.run_formal_sota_unit import (
            _select_models_and_analytical_policies,
        )
        from utils.mainline_contract import load_mainline_config

        selected, policies = _select_models_and_analytical_policies(
            "all", load_mainline_config()
        )

        self.assertEqual(
            {config["model_type"] for config in selected.values()},
            {
                "transformer",
                "plgaformer",
                "kinematic",
                "rotating_3dof",
                "dlinear",
                "patchtst",
                "itransformer",
            },
        )
        self.assertEqual(set(policies), {"kinematic", "rotating_3dof"})

    def test_frozen_evaluation_is_deduplicated_by_source_and_checkpoint(self):
        from scripts.evaluate_formal_sota_selection import (
            find_existing_frozen_evaluation,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen_sota_seed42_full_1.json"
            path.write_text(
                json.dumps(
                    {
                        "run_id": "frozen",
                        "test_evaluation_performed": True,
                        "source_selection_record_sha256": "source",
                        "checkpoint_sha256": "checkpoint",
                    }
                ),
                encoding="utf-8",
            )

            found = find_existing_frozen_evaluation(
                tmp,
                source_record_sha256="source",
                checkpoint_sha256="checkpoint",
            )
            self.assertIsNotNone(found)
            self.assertEqual(found[0], path)
            self.assertIsNone(
                find_existing_frozen_evaluation(
                    tmp,
                    source_record_sha256="other",
                    checkpoint_sha256="checkpoint",
                )
            )

if __name__ == "__main__":
    unittest.main()
