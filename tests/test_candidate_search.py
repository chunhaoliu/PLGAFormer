import tempfile
import unittest
from pathlib import Path


class CandidateSearchTests(unittest.TestCase):
    def test_candidate_grid_is_deterministic_and_limited(self):
        from scripts.run_plgaformer_candidate_search import build_candidate_grid

        candidates = build_candidate_grid(max_candidates=3)

        self.assertEqual([c.candidate_id for c in candidates], [
            "c01_lr0.001_do0.10_a0",
            "c02_lr0.001_do0.10_a0.0001",
            "c03_lr0.001_do0.10_a0.0005",
        ])
        self.assertTrue(all(c.innovations["use_sparse_attention"] for c in candidates))
        self.assertTrue(all(c.innovations["use_physics_corrector"] for c in candidates))
        self.assertTrue(all(c.innovations["use_multi_head_output"] for c in candidates))

    def test_rank_candidates_prefers_low_mse_then_higher_improvement(self):
        from scripts.run_plgaformer_candidate_search import rank_candidate_rows

        rows = [
            {"candidate_id": "a", "mse": 0.21, "baseline_improvement_pct": 4.0},
            {"candidate_id": "b", "mse": 0.18, "baseline_improvement_pct": 3.0},
            {"candidate_id": "c", "mse": 0.18, "baseline_improvement_pct": 9.0},
        ]

        ranked = rank_candidate_rows(rows)

        self.assertEqual([r["candidate_id"] for r in ranked], ["c", "b", "a"])
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3])

    def test_select_candidates_by_id_keeps_requested_order(self):
        from scripts.run_plgaformer_candidate_search import (
            build_candidate_grid,
            select_candidates_by_id,
        )

        candidates = build_candidate_grid(max_candidates=6)

        selected = select_candidates_by_id(candidates, ["c05_lr0.001_do0.05_a0.0001", "c02_lr0.001_do0.10_a0.0001"])

        self.assertEqual([c.candidate_id for c in selected], [
            "c05_lr0.001_do0.05_a0.0001",
            "c02_lr0.001_do0.10_a0.0001",
        ])

    def test_candidate_seed_is_stable_and_order_independent(self):
        from scripts.run_plgaformer_candidate_search import candidate_seed_for_id

        first = candidate_seed_for_id(42, "c17_lr0.0007_do0.15_a0.0001")
        second = candidate_seed_for_id(42, "c17_lr0.0007_do0.15_a0.0001")
        other = candidate_seed_for_id(42, "c27_lr0.0005_do0.15_a0.0005")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_model_seed_mode_can_share_seed_for_fair_ablation_search(self):
        from scripts.run_plgaformer_candidate_search import model_seed_for_id

        shared_a = model_seed_for_id(42, "baseline", seed_mode="shared")
        shared_b = model_seed_for_id(42, "c11_lr0.0007_do0.10_a0.0001", seed_mode="shared")
        candidate_a = model_seed_for_id(42, "baseline", seed_mode="candidate")
        candidate_b = model_seed_for_id(42, "c11_lr0.0007_do0.10_a0.0001", seed_mode="candidate")

        self.assertEqual(shared_a, 42)
        self.assertEqual(shared_a, shared_b)
        self.assertNotEqual(candidate_a, candidate_b)

    def test_write_candidate_outputs_creates_latest_and_best_files(self):
        from scripts.run_plgaformer_candidate_search import write_candidate_outputs

        rows = [
            {"candidate_id": "c01", "mse": 0.2, "baseline_improvement_pct": 5.0, "alpha": 0.0},
            {"candidate_id": "c02", "mse": 0.1, "baseline_improvement_pct": 15.0, "alpha": 0.0001},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_candidate_outputs(
                rows=rows,
                output_dir=Path(tmp),
                run_config={"epochs": 2, "subset_ratio": 0.05},
                timestamp="20260514_120000",
            )

            self.assertTrue(paths["latest_json"].exists())
            self.assertTrue(paths["latest_csv"].exists())
            self.assertTrue(paths["best_json"].exists())
            self.assertIn("candidate_search_20260514_120000.json", paths["timestamped_json"].name)


if __name__ == "__main__":
    unittest.main()
