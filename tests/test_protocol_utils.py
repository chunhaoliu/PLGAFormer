import unittest

import torch

from utils.inference_protocol import predict_by_eval_protocol
from utils.seq2seq_protocol import align_source_position_scale, build_supervision_windows


class Informer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_decoder = None

    def forward(self, x, target_length=None, decoder_context=None):
        self.last_decoder = decoder_context
        return torch.zeros(x.size(0), target_length, 3, dtype=x.dtype, device=x.device)


class ARModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x, y_input, tgt_mask=None):
        self.last_input = y_input
        return torch.zeros_like(y_input)


class ProtocolUtilsTest(unittest.TestCase):
    def test_build_supervision_windows_pred_window(self):
        tgt = torch.randn(2, 16, 3)
        decoder_in, target = build_supervision_windows(tgt, "pred_window", pred_len=8, label_len=4)
        self.assertEqual(tuple(decoder_in.shape), (2, 12, 3))
        self.assertEqual(tuple(target.shape), (2, 8, 3))

    def test_build_supervision_windows_source_context_pred_window(self):
        src = torch.randn(2, 12, 6)
        tgt = torch.randn(2, 16, 3)
        decoder_in, target = build_supervision_windows(
            tgt,
            "source_context_pred_window",
            pred_len=8,
            label_len=4,
            src=src,
        )

        self.assertEqual(tuple(decoder_in.shape), (2, 12, 3))
        self.assertEqual(tuple(target.shape), (2, 8, 3))
        self.assertTrue(torch.allclose(decoder_in[:, :4], src[:, -4:, :3]))
        self.assertTrue(torch.allclose(decoder_in[:, 4:], torch.zeros_like(decoder_in[:, 4:])))

    def test_align_source_position_scale_converts_input_scaled_positions_to_target_scale(self):
        src = torch.tensor(
            [
                [
                    [0.0, 1.0, -1.0, 9.0],
                    [2.0, 0.0, 1.0, 8.0],
                ]
            ],
            dtype=torch.float32,
        )
        input_mean = torch.tensor([10.0, 20.0, 30.0, 0.0])
        input_scale = torch.tensor([2.0, 4.0, 5.0, 1.0])
        output_mean = torch.tensor([8.0, 18.0, 25.0])
        output_scale = torch.tensor([4.0, 2.0, 10.0])

        converted = align_source_position_scale(
            src,
            input_mean=input_mean,
            input_scale=input_scale,
            output_mean=output_mean,
            output_scale=output_scale,
            output_dim=3,
        )

        raw_positions = src[..., :3] * input_scale[:3] + input_mean[:3]
        expected = (raw_positions - output_mean) / output_scale
        self.assertTrue(torch.allclose(converted[..., :3], expected))
        self.assertTrue(torch.allclose(converted[..., 3:], src[..., 3:]))

    def test_inference_official_like_decoder_for_oneshot(self):
        model = Informer()
        x = torch.randn(2, 12, 6)
        y = torch.randn(2, 16, 3)
        pred = predict_by_eval_protocol(
            model=model,
            x=x,
            y_true_scaled=y,
            pred_length=8,
            device="cpu",
            eval_protocol="official_like_decoder",
            eval_ar_seed_mode="zero",
            label_len=4,
        )
        self.assertEqual(tuple(pred.shape), (2, 8, 3))
        self.assertIsNotNone(model.last_decoder)
        self.assertEqual(tuple(model.last_decoder.shape), (2, 12, 3))

    def test_inference_autoregressive(self):
        model = ARModel()
        x = torch.randn(2, 12, 6)
        y = torch.randn(2, 16, 3)
        pred = predict_by_eval_protocol(
            model=model,
            x=x,
            y_true_scaled=y,
            pred_length=8,
            device="cpu",
            eval_protocol="strict_autoregressive",
            eval_ar_seed_mode="zero",
            label_len=4,
            oneshot_models=[],
        )
        self.assertEqual(tuple(pred.shape), (2, 8, 3))

    def test_inference_source_context_decoder_has_no_future_leakage(self):
        model = ARModel()
        x = torch.randn(2, 12, 6)
        y = torch.randn(2, 16, 3)
        pred = predict_by_eval_protocol(
            model=model,
            x=x,
            y_true_scaled=y,
            pred_length=8,
            device="cpu",
            eval_protocol="source_context_decoder",
            eval_ar_seed_mode="zero",
            label_len=4,
            oneshot_models=[],
        )

        self.assertEqual(tuple(pred.shape), (2, 8, 3))
        self.assertIsNotNone(model.last_input)
        self.assertTrue(torch.allclose(model.last_input[:, :4], x[:, -4:, :3]))
        self.assertTrue(torch.allclose(model.last_input[:, 4:], torch.zeros_like(model.last_input[:, 4:])))


if __name__ == "__main__":
    unittest.main()
