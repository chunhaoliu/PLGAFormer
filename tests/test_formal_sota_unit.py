import csv
import json
import tempfile
import unittest
from pathlib import Path


class FormalSotaUnitTests(unittest.TestCase):
    def test_selection_only_runs_are_isolated_from_test_evidence(self):
        from scripts.run_formal_sota_unit import resolve_evidence_tier

        self.assertEqual(resolve_evidence_tier(1.0, True), "convergence_pilot")
        self.assertEqual(resolve_evidence_tier(1.0, False), "final")
        self.assertEqual(resolve_evidence_tier(0.2, False), "screening")

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
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.prediction_length, 256)
        self.assertEqual(args.prediction_horizons, "32,64,128,256")
        self.assertEqual(args.label_len, 128)
        self.assertEqual(args.seeds, "42,123,456")
        self.assertEqual(args.workers, 0)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertTrue(args.cache_physics_prior)
        self.assertEqual(args.physics_prior_cache_batch_size, 512)

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
        from scripts.run_formal_sota_unit import apply_plgaformer_candidate_overrides

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

            kwargs = apply_plgaformer_candidate_overrides(config, path)

            self.assertEqual(config["physics_loss_weight"], 0.0)
            self.assertEqual(kwargs["dropout"], 0.05)
            self.assertTrue(kwargs["use_sparse_attention"])


if __name__ == "__main__":
    unittest.main()
