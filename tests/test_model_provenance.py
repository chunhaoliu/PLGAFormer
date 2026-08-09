import unittest

from utils.model_provenance import (
    build_comparison_model_provenance,
    get_model_provenance,
    validate_comparison_model_types,
)


class ModelProvenanceTests(unittest.TestCase):
    def test_get_model_provenance_known_and_unknown(self):
        known = get_model_provenance("informer")
        self.assertEqual(known["reference_family"], "informer")
        self.assertNotEqual(known["source_type"], "unknown")

        autoformer = get_model_provenance("autoformer")
        self.assertEqual(autoformer["reference_family"], "autoformer")
        self.assertEqual(autoformer["source_type"], "reimplemented")

        unknown = get_model_provenance("does_not_exist")
        self.assertEqual(unknown["source_type"], "unknown")
        self.assertEqual(unknown["reference_family"], "does_not_exist")

    def test_validate_comparison_model_types(self):
        comparison_models = {
            "Informer": {"model_type": "informer"},
            "UnknownModel": {"model_type": "abc_xyz"},
            "EmptyModel": {},
        }
        unknown = validate_comparison_model_types(comparison_models)
        self.assertIn("abc_xyz", unknown)
        self.assertIn("<empty>", unknown)

    def test_build_comparison_model_provenance(self):
        comparison_models = {
            "Informer": {"model_type": "informer"},
            "Autoformer": {"model_type": "autoformer"},
            "PatchTST": {"model_type": "patchtst"},
        }
        provenance = build_comparison_model_provenance(comparison_models)
        self.assertEqual(set(provenance.keys()), {"Informer", "Autoformer", "PatchTST"})
        self.assertEqual(provenance["Informer"]["reference_family"], "informer")
        self.assertEqual(provenance["Autoformer"]["reference_family"], "autoformer")
        self.assertEqual(provenance["PatchTST"]["reference_family"], "patchtst")


if __name__ == "__main__":
    unittest.main()
