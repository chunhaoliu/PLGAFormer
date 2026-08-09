import unittest

import torch


class AutoformerModelTests(unittest.TestCase):
    def test_autoformer_registered_model_outputs_requested_horizon(self):
        from models import create_registered_model, get_supported_model_types

        self.assertIn("autoformer", get_supported_model_types())

        model = create_registered_model("autoformer", input_dim=6, device="cpu")
        model.eval()
        x = torch.randn(2, 64, 6)
        with torch.no_grad():
            out = model(x, target_length=32)

        self.assertEqual(tuple(out.shape), (2, 32, 3))


if __name__ == "__main__":
    unittest.main()
