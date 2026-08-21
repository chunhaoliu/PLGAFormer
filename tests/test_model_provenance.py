import unittest

from utils.model_provenance import (
    build_comparison_model_provenance,
    get_model_provenance,
    validate_comparison_model_types,
)


class ModelProvenanceTests(unittest.TestCase):
    def test_get_model_provenance_known_and_unknown(self):
        known = get_model_provenance("patchtst")
        self.assertEqual(known["reference_family"], "patchtst")
        self.assertNotEqual(known["source_type"], "unknown")

        historical = get_model_provenance("autoformer")
        self.assertEqual(historical["reference_family"], "autoformer")
        self.assertEqual(historical["source_type"], "reimplemented")

        unknown = get_model_provenance("does_not_exist")
        self.assertEqual(unknown["source_type"], "unknown")
        self.assertEqual(unknown["reference_family"], "does_not_exist")

    def test_validate_comparison_model_types(self):
        comparison_models = {
            "Transformer": {"model_type": "transformer"},
            "UnknownModel": {"model_type": "abc_xyz"},
            "EmptyModel": {},
        }
        unknown = validate_comparison_model_types(comparison_models)
        self.assertIn("abc_xyz", unknown)
        self.assertIn("<empty>", unknown)

    def test_build_comparison_model_provenance(self):
        comparison_models = {
            "Transformer": {"model_type": "transformer"},
            "PLGAFormer": {"model_type": "plgaformer"},
            "PatchTST": {"model_type": "patchtst"},
        }
        provenance = build_comparison_model_provenance(comparison_models)
        self.assertEqual(set(provenance.keys()), {"Transformer", "PLGAFormer", "PatchTST"})
        self.assertEqual(provenance["Transformer"]["reference_family"], "vanilla_transformer")
        self.assertEqual(provenance["PLGAFormer"]["reference_family"], "plgaformer")
        self.assertEqual(provenance["PatchTST"]["reference_family"], "patchtst")


if __name__ == "__main__":
    unittest.main()
