import tempfile
import unittest
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _write_final_ablation_record(root, mutate=None):
    from scripts.run_formal_ablation_unit import sha256_file

    root = Path(root)
    checkpoint = root / "checkpoint.pth"
    checkpoint.write_bytes(b"formal weights")
    phase = "phase4_final_mechanism_controls"
    seed = 42
    model_key = "schedule_only"
    protocol_identity = {
        "run_signature": "exact",
        "prediction_length": 256,
        "strict_flag": True,
    }
    model_config = {
        "model_type": "plgaformer",
        "innovations": {"use_prior_fusion": True},
    }
    formal_config_sha256 = "a" * 64
    payload = {
        "schema_version": 2,
        "formal_config_sha256": formal_config_sha256,
        "evidence_tier": "final",
        "test_evaluation_performed": True,
        "phase": phase,
        "seed": seed,
        "model_key": model_key,
        "protocol_identity": deepcopy(protocol_identity),
        "model_config": deepcopy(model_config),
        "horizons": [32, 64, 128, 256],
        "eval_results": {
            str(horizon): {"ade": 1.0, "fde": 2.0, "rmse_cart_m": 3.0}
            for horizon in (32, 64, 128, 256)
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": {"epochs_completed": 50, "best_epoch": 12},
        "run_id": "formal-final",
    }
    if mutate is not None:
        mutate(payload, checkpoint)
    record = root / f"formal_{phase}_seed{seed}_{model_key}_1.json"
    record.write_text(json.dumps(payload), encoding="utf-8")
    return record, protocol_identity, model_config, formal_config_sha256


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

    def test_ablation_record_config_uses_actual_plgaformer_values(self):
        from scripts.run_formal_ablation_unit import (
            _resolved_ablation_record_config,
        )

        requested = {
            "model_type": "plgaformer",
            "dropout": 0.1,
            "innovations": {
                "use_prior_fusion": True,
                "prior_type": "rotating_3dof",
            },
        }
        model = SimpleNamespace(
            dropout=0.25,
            use_prior_fusion=False,
            prior_type="spherical_kinematic",
        )

        recorded = _resolved_ablation_record_config(model, requested)

        self.assertEqual(recorded["dropout"], 0.25)
        self.assertFalse(recorded["innovations"]["use_prior_fusion"])
        self.assertEqual(
            recorded["innovations"]["prior_type"], "spherical_kinematic"
        )
        self.assertTrue(requested["innovations"]["use_prior_fusion"])

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
        self.assertEqual(args.seeds, "42,123,456")
        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.prediction_length, 256)
        self.assertEqual(args.prediction_horizons, "32,64,128,256")
        self.assertEqual(args.label_len, 128)
        self.assertEqual(args.learning_rate, 0.001)
        self.assertEqual(args.weight_decay, 0.00005)
        self.assertEqual(args.patience, 15)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertTrue(args.cache_physics_prior)
        self.assertEqual(args.physics_prior_cache_batch_size, 512)
        self.assertFalse(args.reuse_exp1_controls)

    def test_runtime_drift_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_ablation_unit import (
            parse_args,
            run_formal_ablation_units,
        )

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
                run_formal_ablation_units(
                    parse_args(["--prediction-length", "128"])
                )

        self.assertEqual(training_imports, [])

    def test_reduced_final_subset_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_ablation_unit import (
            parse_args,
            run_formal_ablation_units,
        )

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, r"subset_ratio.*1\.0"):
                run_formal_ablation_units(parse_args(["--subset-ratio", "0.2"]))
        self.assertEqual(training_imports, [])

    def test_custom_config_is_rejected_before_training_imports(self):
        import builtins

        from scripts.run_formal_ablation_unit import (
            parse_args,
            run_formal_ablation_units,
        )

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "config"):
                run_formal_ablation_units(parse_args(["--config", "other.json"]))
        self.assertEqual(training_imports, [])

    def test_default_config_is_the_canonical_contract(self):
        from scripts.run_formal_ablation_unit import parse_args
        from utils.mainline_contract import ACTIVE_CONFIG_PATH

        self.assertEqual(Path(parse_args([]).config).resolve(), ACTIVE_CONFIG_PATH)

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

    def test_final_ablation_resume_accepts_only_exact_complete_json(self):
        from scripts.run_formal_ablation_unit import find_existing_final_ablation

        with tempfile.TemporaryDirectory() as tmp:
            (
                record,
                protocol_identity,
                model_config,
                formal_config_sha256,
            ) = _write_final_ablation_record(tmp)
            found = find_existing_final_ablation(
                tmp,
                phase="phase4_final_mechanism_controls",
                seed=42,
                model_key="schedule_only",
                protocol_identity=protocol_identity,
                model_config=model_config,
                horizons=[32, 64, 128, 256],
                formal_config_sha256=formal_config_sha256,
            )

            self.assertIsNotNone(found)
            self.assertEqual(found[0], record)

    def test_final_ablation_resume_rejects_missing_json(self):
        from scripts.run_formal_ablation_unit import find_existing_final_ablation

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                find_existing_final_ablation(
                    tmp,
                    phase="phase4_final_mechanism_controls",
                    seed=42,
                    model_key="schedule_only",
                    protocol_identity={"run_signature": "exact"},
                    model_config={"model_type": "plgaformer"},
                    horizons=[32, 64, 128, 256],
                    formal_config_sha256="a" * 64,
                )
            )

    def test_final_ablation_resume_rejects_malformed_json_records(self):
        from scripts.run_formal_ablation_unit import find_existing_final_ablation

        for content in ("{bad json", "[]"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                record = Path(tmp) / (
                    "formal_phase4_final_mechanism_controls_"
                    "seed42_schedule_only_1.json"
                )
                record.write_text(content, encoding="utf-8")
                self.assertIsNone(
                    find_existing_final_ablation(
                        tmp,
                        phase="phase4_final_mechanism_controls",
                        seed=42,
                        model_key="schedule_only",
                        protocol_identity={"run_signature": "exact"},
                        model_config={"model_type": "plgaformer"},
                        horizons=[32, 64, 128, 256],
                        formal_config_sha256="a" * 64,
                    )
                )

    def test_final_ablation_resume_rejects_inexact_records(self):
        from scripts.run_formal_ablation_unit import find_existing_final_ablation

        def checkpoint_missing(payload, checkpoint):
            payload["checkpoint"] = str(checkpoint.with_name("missing.pth"))

        def hash_missing(payload, _checkpoint):
            payload.pop("checkpoint_sha256")

        def hash_mismatch(payload, _checkpoint):
            payload["checkpoint_sha256"] = "0" * 64

        def formal_config_hash_missing(payload, _checkpoint):
            payload.pop("formal_config_sha256")

        def formal_config_hash_mismatch(payload, _checkpoint):
            payload["formal_config_sha256"] = "b" * 64

        def schema_version_missing(payload, _checkpoint):
            payload.pop("schema_version")

        def protocol_mismatch(payload, _checkpoint):
            payload["protocol_identity"] = {"run_signature": "changed"}

        def protocol_bool_int_mismatch(payload, _checkpoint):
            payload["protocol_identity"]["strict_flag"] = 1

        def config_mismatch(payload, _checkpoint):
            payload["model_config"] = {"model_type": "changed"}

        def config_bool_int_mismatch(payload, _checkpoint):
            payload["model_config"]["innovations"]["use_prior_fusion"] = 1

        def metrics_incomplete(payload, _checkpoint):
            payload["eval_results"]["256"].pop("rmse_cart_m")

        def metric_is_bool(payload, _checkpoint):
            payload["eval_results"]["256"]["ade"] = True

        def declared_horizons_incomplete(payload, _checkpoint):
            payload["horizons"] = [32, 64, 128]

        def history_incomplete(payload, _checkpoint):
            payload["history"].pop("best_epoch")

        def history_best_epoch_out_of_range(payload, _checkpoint):
            payload["history"]["best_epoch"] = 51

        def run_id_missing(payload, _checkpoint):
            payload.pop("run_id")

        def identity_mismatch(payload, _checkpoint):
            payload["seed"] = 123

        cases = {
            "checkpoint missing": checkpoint_missing,
            "hash missing": hash_missing,
            "hash mismatch": hash_mismatch,
            "formal config hash missing": formal_config_hash_missing,
            "formal config hash mismatch": formal_config_hash_mismatch,
            "schema version missing": schema_version_missing,
            "protocol mismatch": protocol_mismatch,
            "protocol bool-int mismatch": protocol_bool_int_mismatch,
            "config mismatch": config_mismatch,
            "config bool-int mismatch": config_bool_int_mismatch,
            "metrics incomplete": metrics_incomplete,
            "metric is bool": metric_is_bool,
            "declared horizons incomplete": declared_horizons_incomplete,
            "history incomplete": history_incomplete,
            "history best epoch out of range": history_best_epoch_out_of_range,
            "run id missing": run_id_missing,
            "identity mismatch": identity_mismatch,
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                (
                    _,
                    protocol_identity,
                    model_config,
                    formal_config_sha256,
                ) = _write_final_ablation_record(tmp, mutate)
                self.assertIsNone(
                    find_existing_final_ablation(
                        tmp,
                        phase="phase4_final_mechanism_controls",
                        seed=42,
                        model_key="schedule_only",
                        protocol_identity=protocol_identity,
                        model_config=model_config,
                        horizons=[32, 64, 128, 256],
                        formal_config_sha256=formal_config_sha256,
                    )
                )

    def test_all_exact_final_records_avoid_training_data_load(self):
        from scripts.run_formal_ablation_unit import (
            _requires_ablation_training,
        )

        models = {
            "PLGAFormer rotating-prior schedule only": {
                "model_type": "plgaformer"
            }
        }
        reusable = {
            ("phase4_final_mechanism_controls", 42, "schedule_only")
        }
        self.assertFalse(
            _requires_ablation_training(
                "phase4_final_mechanism_controls", [42], models, reusable
            )
        )

    def test_missing_final_control_requires_training_without_exp1_reuse(self):
        from scripts.run_formal_ablation_unit import _requires_ablation_training

        models = {"Transformer (baseline)": {"model_type": "baseline"}}

        self.assertTrue(
            _requires_ablation_training(
                "phase1_structural",
                [42],
                models,
                set(),
                reuse_exp1_controls=False,
            )
        )
        self.assertFalse(
            _requires_ablation_training(
                "phase1_structural",
                [42],
                models,
                set(),
                reuse_exp1_controls=True,
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
