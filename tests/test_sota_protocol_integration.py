import unittest
from unittest import mock

import numpy as np
import torch


class SotaProtocolIntegrationTests(unittest.TestCase):
    @staticmethod
    def _tiny_training_config(exp1):
        config = dict(exp1._base_train_config)
        config.update({
            "device": "cpu",
            "epochs": 1,
            "pred_len": 3,
            "label_len": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "early_stopping_patience": 2,
            "gradient_clip_norm": 1.0,
            "use_mixed_precision": False,
        })
        return config

    def test_formal_training_persists_complete_epoch_history_without_output_clipping(self):
        from torch.utils.data import DataLoader, TensorDataset
        from experiments.exp1_sota import SOTA_comparison as exp1

        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(10.0))

            def forward(self, src, tgt, tgt_mask=None):
                del src, tgt_mask
                return self.bias.expand(tgt.size(0), tgt.size(1), 3)

        src = torch.zeros(2, 4, 6)
        tgt = torch.zeros(2, 3, 3)
        loader = DataLoader(TensorDataset(src, tgt), batch_size=2, shuffle=False)
        config = self._tiny_training_config(exp1)

        with mock.patch.object(exp1, "_base_train_config", config):
            model, history = exp1.train_model(
                "constant",
                {"model_type": "transformer", "physics_loss_weight": 0.0},
                loader,
                loader,
                torch.zeros(3),
                torch.ones(3),
                model_override=ConstantModel(),
            )

        self.assertIsInstance(model, ConstantModel)
        self.assertEqual(history["epochs_completed"], 1)
        self.assertEqual(history["best_epoch"], 1)
        self.assertEqual(history["epoch_records"][0]["train_batches"], 1)
        self.assertEqual(history["epoch_records"][0]["val_batches"], 1)
        self.assertAlmostEqual(history["train_losses"][0], 10.0, places=4)

    def test_formal_training_fails_on_nonfinite_input(self):
        from torch.utils.data import DataLoader, TensorDataset
        from experiments.exp1_sota import SOTA_comparison as exp1

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3)

            def forward(self, src, tgt, tgt_mask=None):
                del src, tgt_mask
                return self.projection(tgt)

        src = torch.zeros(2, 4, 6)
        src[0, 0, 0] = float("nan")
        tgt = torch.zeros(2, 3, 3)
        loader = DataLoader(TensorDataset(src, tgt), batch_size=2, shuffle=False)
        config = self._tiny_training_config(exp1)

        with mock.patch.object(exp1, "_base_train_config", config):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite training input"):
                exp1.train_model(
                    "tiny",
                    {"model_type": "transformer", "physics_loss_weight": 0.0},
                    loader,
                    loader,
                    torch.zeros(3),
                    torch.ones(3),
                    model_override=TinyModel(),
                )

    def test_per_maneuver_evaluation_is_batched_and_exactly_weighted(self):
        from experiments.exp1_sota import SOTA_comparison as exp1

        model = torch.nn.Identity()
        X = torch.randn(7, 8, 6)
        y = torch.zeros(7, 4, 3)
        y[..., 0] = 6_448_000.0
        labels = np.asarray(["a", "a", "a", "a", "b", "b", "b"])
        observed_batch_sizes = []

        def exact_prediction(_model, x, _target_length, **kwargs):
            observed_batch_sizes.append(x.size(0))
            return kwargs["y_true_scaled"]

        old_batch_size = exp1._base_train_config["batch_size"]
        exp1._base_train_config["batch_size"] = 2
        try:
            with mock.patch.object(exp1, "unified_predict", side_effect=exact_prediction):
                result = exp1.evaluate_per_maneuver(
                    model,
                    (X, y),
                    labels,
                    torch.zeros(3),
                    torch.ones(3),
                    torch.device("cpu"),
                    horizon=4,
                )
        finally:
            exp1._base_train_config["batch_size"] = old_batch_size

        self.assertLessEqual(max(observed_batch_sizes), 2)
        self.assertEqual(result["a"]["num_samples"], 4)
        self.assertEqual(result["b"]["num_samples"], 3)
        for metrics in result.values():
            self.assertAlmostEqual(metrics["mse"], 0.0)
            self.assertAlmostEqual(metrics["mae"], 0.0)
            self.assertAlmostEqual(metrics["fde"], 0.0)
            self.assertAlmostEqual(metrics["ade"], 0.0)

    def test_plgaformer_reconstruction_uses_the_training_scalers(self):
        from experiments.exp1_sota import SOTA_comparison as exp1

        class Scaler:
            pass

        input_scaler = Scaler()
        input_scaler.mean_ = np.arange(6, dtype=np.float64)
        input_scaler.scale_ = np.arange(1, 7, dtype=np.float64)
        output_scaler = Scaler()
        output_scaler.mean_ = np.asarray([10.0, 11.0, 12.0])
        output_scaler.scale_ = np.asarray([2.0, 3.0, 4.0])

        kwargs = exp1._model_reconstruction_kwargs(
            "plgaformer", input_scaler, output_scaler
        )

        np.testing.assert_array_equal(kwargs["input_scaler_mean"], [10, 11, 12, 3, 4, 5])
        np.testing.assert_array_equal(kwargs["input_scaler_scale"], [2, 3, 4, 4, 5, 6])
        self.assertTrue(kwargs["require_physical_scaler"])
        self.assertIsNone(
            exp1._model_reconstruction_kwargs("transformer", input_scaler, output_scaler)
        )

    def test_exp1_source_context_training_window_uses_observed_source(self):
        from experiments.exp1_sota import SOTA_comparison as exp1

        src = torch.randn(2, 12, 6)
        tgt = torch.randn(2, 16, 3)

        decoder_in, target = exp1.build_supervision_windows(
            tgt,
            "source_context_pred_window",
            src=src,
        )

        label_len = min(int(exp1._base_train_config.get("label_len", 48)), src.size(1))
        pred_len = min(int(exp1._base_train_config.get("pred_len", tgt.size(1))), tgt.size(1))
        self.assertEqual(tuple(decoder_in.shape), (2, label_len + pred_len, 3))
        self.assertTrue(torch.allclose(decoder_in[:, :label_len], src[:, -label_len:, :3]))
        self.assertTrue(torch.allclose(target, tgt[:, :pred_len, :]))

    def test_exp1_source_context_eval_uses_observed_source_for_decoder(self):
        from experiments.exp1_sota import SOTA_comparison as exp1

        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_tgt = None

            def forward(self, src, tgt, tgt_mask=None):
                self.last_tgt = tgt
                return torch.zeros_like(tgt)

        model = CaptureModel()
        src = torch.randn(2, 12, 6)
        y = torch.randn(2, 16, 3)

        pred = exp1.unified_predict(
            model,
            src,
            target_length=8,
            y_true_scaled=y,
            device="cpu",
            eval_protocol="source_context_decoder",
        )

        label_len = min(int(exp1._base_train_config.get("label_len", 48)), src.size(1))
        self.assertEqual(tuple(pred.shape), (2, 8, 3))
        self.assertIsNotNone(model.last_tgt)
        self.assertTrue(torch.allclose(model.last_tgt[:, :label_len], src[:, -label_len:, :3]))

    def test_exp1_oneshot_source_context_training_uses_pred_len_and_decoder_context(self):
        from experiments.exp1_sota import SOTA_comparison as exp1

        class ActiveOneShotAdapter(torch.nn.Module):
            is_oneshot = True

            def __init__(self):
                super().__init__()
                self.target_length = None
                self.decoder_context = None

            def forward(self, src, target_length=None, decoder_context=None):
                self.target_length = target_length
                self.decoder_context = decoder_context
                return torch.zeros(src.size(0), target_length, 3)

        model = ActiveOneShotAdapter()
        src = torch.randn(2, 12, 6)
        decoder_input = torch.randn(2, 20, 3)

        out = exp1.training_forward(
            model,
            src,
            decoder_input,
            tgt_mask=None,
            device="cpu",
            supervision_protocol="source_context_pred_window",
        )

        self.assertEqual(model.target_length, int(exp1._base_train_config.get("pred_len", 128)))
        self.assertIs(model.decoder_context, decoder_input)
        self.assertEqual(tuple(out.shape), (2, model.target_length, 3))


if __name__ == "__main__":
    unittest.main()
