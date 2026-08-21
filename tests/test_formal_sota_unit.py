import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FormalSotaUnitTests(unittest.TestCase):
    def test_selection_only_runs_are_isolated_from_test_evidence(self):
        from scripts.run_formal_sota_unit import resolve_evidence_tier

        self.assertEqual(resolve_evidence_tier(1.0, True), "convergence_pilot")
        self.assertEqual(resolve_evidence_tier(1.0, False), "final")
        with self.assertRaisesRegex(ValueError, "subset_ratio"):
            resolve_evidence_tier(0.2, False)

    def test_final_candidate_is_rejected_before_training_imports(self):
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

        args = parse_args(["--candidate-file", "candidate.json"])
        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "candidate-file"):
                run_formal_sota_units(args)

        self.assertEqual(training_imports, [])

    def test_selection_only_candidate_remains_an_explicit_diagnostic(self):
        from scripts.run_formal_sota_unit import (
            parse_args,
            validate_sota_authority,
        )

        validate_sota_authority(
            parse_args(["--selection-only", "--candidate-file", "candidate.json"])
        )

    def test_selection_candidate_cannot_resume_before_training_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        args = parse_args(
            [
                "--selection-only",
                "--candidate-file",
                "candidate.json",
                "--skip-existing",
            ]
        )
        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError, "skip-existing"):
                run_formal_sota_units(args)

        self.assertEqual(training_imports, [])

    def test_malformed_candidate_for_transformer_then_plgaformer_fails_before_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.json"
            candidate.write_text(
                json.dumps({"best_candidate": {"dropout": float("nan")}}),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--selection-only",
                    "--models",
                    "transformer,plgaformer",
                    "--candidate-file",
                    str(candidate),
                ]
            )
            with patch("builtins.__import__", side_effect=guarded_import):
                with self.assertRaisesRegex(ValueError, "candidate"):
                    run_formal_sota_units(args)

        self.assertEqual(training_imports, [])

    def test_transformer_only_candidate_is_rejected_before_imports(self):
        import builtins

        from scripts.run_formal_sota_unit import parse_args, run_formal_sota_units

        original_import = builtins.__import__
        training_imports = []

        def guarded_import(name, *args, **kwargs):
            if name.startswith(("data_generation", "experiments")):
                training_imports.append(name)
                raise AssertionError(f"training import reached: {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.json"
            candidate.write_text(json.dumps({"best_candidate": {}}), encoding="utf-8")
            args = parse_args(
                [
                    "--selection-only",
                    "--models",
                    "transformer",
                    "--candidate-file",
                    str(candidate),
                ]
            )
            with patch("builtins.__import__", side_effect=guarded_import):
                with self.assertRaisesRegex(ValueError, "PLGAFormer"):
                    run_formal_sota_units(args)

        self.assertEqual(training_imports, [])

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

    def test_af_ciln_reconstruction_contract_contains_scalers(self):
        import numpy as np
        from experiments.exp1_sota import SOTA_comparison as exp1

        class _Scaler:
            mean_ = np.asarray([1.0, 2.0, 3.0])
            scale_ = np.asarray([4.0, 5.0, 6.0])

        input_scaler = _Scaler()
        input_scaler.mean_ = np.asarray([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        input_scaler.scale_ = np.asarray([13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        kwargs = exp1._model_reconstruction_kwargs(
            "af_ciln", input_scaler, _Scaler()
        )

        self.assertIsNotNone(kwargs)
        self.assertEqual(kwargs["input_scaler_mean"].shape, (6,))
        self.assertEqual(kwargs["input_scaler_scale"].shape, (6,))
        self.assertEqual(kwargs["output_scaler_mean"].shape, (3,))
        self.assertEqual(kwargs["output_scaler_scale"].shape, (3,))

    def test_parser_records_explicit_af_ciln_checkout(self):
        from scripts.run_formal_sota_unit import parse_args

        args = parse_args(["--af-ciln-root", "tmp/AF-CILN"])
        self.assertEqual(args.af_ciln_root, "tmp/AF-CILN")
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

    def test_pit_identity_records_step_and_second_horizons(self):
        from scripts.run_formal_sota_unit import build_protocol_identity, parse_args

        args = parse_args([])
        identity = build_protocol_identity(
            args,
            [32, 64, 128, 256],
            "signature",
            {
                "dataset_protocol": "pit_aligned_radar_v1",
                "dataset_sha256": "abc",
                "sampling_interval_s": 2.0,
            },
        )

        self.assertEqual(identity["dataset_protocol"], "pit_aligned_radar_v1")
        self.assertEqual(identity["prediction_horizons_s"], [64.0, 128.0, 256.0, 512.0])
        self.assertEqual(identity["mixed_precision_dtype"], "float32")
        self.assertTrue(identity["cache_physics_prior"])

    def test_candidate_sha256_is_bound_into_selection_protocol(self):
        from scripts.run_formal_sota_unit import (
            build_protocol_identity,
            load_plgaformer_candidate_snapshot,
            parse_args,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps({"best_candidate": {}}), encoding="utf-8")
            snapshot = load_plgaformer_candidate_snapshot(path)
            identity = build_protocol_identity(
                parse_args(["--selection-only"]),
                [32, 64, 128, 256],
                "signature",
                {
                    "dataset_protocol": "hgv_multiregime_state_v2_1",
                    "dataset_sha256": "dataset",
                    "sampling_interval_s": 1.0,
                },
                candidate_snapshot=snapshot,
            )

        self.assertEqual(identity["candidate_file_sha256"], snapshot.sha256)

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

    def test_patchtst_uses_channel_independent_forecast_heads(self):
        import torch
        from models.sota_models import PatchTST

        model = PatchTST(
            input_dim=6, d_model=16, nhead=4, num_layers=1,
            dim_feedforward=32, dropout=0.0,
            seq_len=32, pred_len=16, patch_len=8, stride=4,
        )
        output = model(torch.randn(2, 32, 6), target_length=16)

        self.assertEqual(tuple(output.shape), (2, 16, 3))
        self.assertEqual(model.output_head.out_features, 16)
        self.assertFalse(torch.allclose(output[..., 0], output[..., 1]))

    def test_dlinear_forecast_shape(self):
        import torch
        from models.baseline_models import DLinear

        model = DLinear(seq_len=32, pred_len=16, moving_avg=5)
        output = model(torch.randn(2, 32, 6), target_length=16)

        self.assertEqual(tuple(output.shape), (2, 16, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_select_model_configs_by_key_preserves_order(self):
        from scripts.run_formal_sota_unit import select_model_configs_by_key

        selected = select_model_configs_by_key("plgaformer,autoformer,transformer")

        self.assertEqual(list(selected.keys()), [
            "PLGAFormer (proposed)",
            "Autoformer",
            "Transformer (baseline)",
        ])
        self.assertEqual(selected["Autoformer"]["model_type"], "autoformer")

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

        selected = select_model_configs_by_key(
            "transformer,plgaformer,pit,kinematic,rotating_3dof,"
            "dlinear,patchtst,itransformer,af_ciln"
        )

        self.assertEqual(
            [config["model_type"] for config in selected.values()],
            [
                "transformer",
                "plgaformer",
                "pit",
                "kinematic",
                "rotating_3dof",
                "dlinear",
                "patchtst",
                "itransformer",
                "af_ciln",
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

    def test_load_plgaformer_candidate_overrides_sota_config(self):
        from scripts.run_formal_sota_unit import (
            apply_plgaformer_candidate_snapshot,
            load_plgaformer_candidate_snapshot,
            parse_args,
            validate_candidate_model_selection,
            validate_sota_authority,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(
                json.dumps(
                    {
                        "best_candidate": {
                            "alpha": 0.0,
                            "dropout": 0.05,
                            "innovations": {
                                "use_sparse_attention": True,
                                "use_physics_corrector": True,
                                "use_multi_head_output": True,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = {"model_type": "plgaformer", "physics_loss_weight": 0.0001}

            args = parse_args(
                [
                    "--selection-only",
                    "--models",
                    "plgaformer",
                    "--candidate-file",
                    str(path),
                ]
            )
            validate_sota_authority(args)
            validate_candidate_model_selection(args)
            snapshot = load_plgaformer_candidate_snapshot(args.candidate_file)
            kwargs = apply_plgaformer_candidate_snapshot(config, snapshot)

            self.assertEqual(config["physics_loss_weight"], 0.0)
            self.assertEqual(kwargs["dropout"], 0.05)
            self.assertTrue(kwargs["use_sparse_attention"])

    def test_explicit_candidate_file_rejects_missing_path(self):
        from scripts.run_formal_sota_unit import load_plgaformer_candidate_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaisesRegex(FileNotFoundError, "candidate"):
                load_plgaformer_candidate_snapshot(missing)

    def test_explicit_candidate_file_rejects_malformed_structure_and_values(self):
        from scripts.run_formal_sota_unit import load_plgaformer_candidate_snapshot

        cases = {
            "invalid json": "{bad json",
            "root is not object": json.dumps([]),
            "best candidate is not object": json.dumps({"best_candidate": []}),
            "innovations is not object": json.dumps(
                {"best_candidate": {"innovations": []}}
            ),
            "dropout is bool": json.dumps({"best_candidate": {"dropout": True}}),
            "dropout is nonfinite": json.dumps(
                {"best_candidate": {"dropout": float("nan")}}
            ),
            "dropout is out of range": json.dumps(
                {"best_candidate": {"dropout": 1.1}}
            ),
            "alpha is bool": json.dumps({"best_candidate": {"alpha": False}}),
            "alpha is nonfinite": json.dumps(
                {"best_candidate": {"alpha": float("inf")}}
            ),
            "alpha is negative": json.dumps({"best_candidate": {"alpha": -0.1}}),
            "innovation flag is not bool": json.dumps(
                {
                    "best_candidate": {
                        "innovations": {"use_sparse_attention": 1}
                    }
                }
            ),
        }
        for name, content in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "candidate.json"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex((ValueError, json.JSONDecodeError), "candidate"):
                    load_plgaformer_candidate_snapshot(path)

    def test_candidate_file_is_read_once_and_frozen_snapshot_is_reused(self):
        from scripts.run_formal_sota_unit import (
            apply_plgaformer_candidate_snapshot,
            load_plgaformer_candidate_snapshot,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(
                json.dumps(
                    {
                        "best_candidate": {
                            "alpha": 0.0,
                            "dropout": 0.05,
                            "innovations": {"use_sparse_attention": True},
                        }
                    }
                ),
                encoding="utf-8",
            )
            original_read_bytes = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=original_read_bytes,
            ) as read_bytes:
                snapshot = load_plgaformer_candidate_snapshot(path)
                first_config = {"model_type": "plgaformer"}
                second_config = {"model_type": "plgaformer"}
                first = apply_plgaformer_candidate_snapshot(first_config, snapshot)
                first["dropout"] = 0.9
                second = apply_plgaformer_candidate_snapshot(second_config, snapshot)

            self.assertEqual(read_bytes.call_count, 1)
            self.assertEqual(first_config["physics_loss_weight"], 0.0)
            self.assertEqual(second_config["physics_loss_weight"], 0.0)
            self.assertEqual(second["dropout"], 0.05)
            self.assertTrue(second["use_sparse_attention"])
            self.assertEqual(len(snapshot.sha256), 64)


if __name__ == "__main__":
    unittest.main()
