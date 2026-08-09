import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory


class FormalAblationSummaryTests(unittest.TestCase):
    def test_select_latest_metric_rows_keeps_latest_duplicate(self):
        from scripts.summarize_formal_ablation import select_latest_metric_rows

        rows = [
            {"seed": "42", "model_key": "full", "horizon": "64", "metric": "mse", "value": "2.0", "timestamp": "1", "run_id": "a"},
            {"seed": "42", "model_key": "full", "horizon": "64", "metric": "mse", "value": "1.0", "timestamp": "2", "run_id": "b"},
        ]

        latest = select_latest_metric_rows(rows)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["value"], "1.0")

    def test_summarize_metric_rows_computes_full_delta(self):
        from scripts.summarize_formal_ablation import summarize_metric_rows

        rows = [
            {"seed": "42", "model_key": "baseline", "horizon": "64", "metric": "mse", "value": "10.0", "timestamp": "1", "run_id": "b"},
            {"seed": "42", "model_key": "full", "horizon": "64", "metric": "mse", "value": "4.0", "timestamp": "1", "run_id": "f"},
            {"seed": "42", "model_key": "wo_a", "horizon": "64", "metric": "mse", "value": "5.0", "timestamp": "1", "run_id": "a"},
        ]

        summary = summarize_metric_rows(rows)
        by_model = {row["model_key"]: row for row in summary}

        self.assertAlmostEqual(by_model["full"]["baseline_improvement_pct"], 60.0)
        self.assertAlmostEqual(by_model["wo_a"]["full_delta_pct"], 20.0)

    def test_phase_is_part_of_the_deduplication_key(self):
        from scripts.summarize_formal_ablation import select_latest_metric_rows

        rows = [
            {"phase": "phase1", "seed": "42", "model_key": "baseline", "horizon": "256", "metric": "ade", "value": "10", "timestamp": "1", "run_id": "a"},
            {"phase": "phase3", "seed": "42", "model_key": "baseline", "horizon": "256", "metric": "ade", "value": "12", "timestamp": "2", "run_id": "b"},
        ]

        self.assertEqual(len(select_latest_metric_rows(rows)), 2)

    def test_validation_selection_uses_mean_best_validation_objective(self):
        from scripts.summarize_formal_ablation import summarize_validation_selection

        rows = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            losses = {
                "prior_only": [0.20, 0.30, 0.25],
                "proposed": [0.22, 0.31, 0.27],
            }
            for model_key, model_losses in losses.items():
                for seed, loss in zip((42, 123, 456), model_losses):
                    run_id = f"run_{model_key}_{seed}"
                    rows.append(
                        {
                            "phase": "phase1_structural",
                            "seed": str(seed),
                            "model_key": model_key,
                            "timestamp": "1",
                            "run_id": run_id,
                        }
                    )
                    (root / f"{run_id}.json").write_text(
                        json.dumps({"history": {"best_val_loss": loss}}),
                        encoding="utf-8",
                    )

            summary = summarize_validation_selection(rows, root)

        self.assertEqual(summary["selected_model_key"], "prior_only")
        by_model = {row["model_key"]: row for row in summary["candidates"]}
        self.assertAlmostEqual(by_model["prior_only"]["mean_best_val_loss"], 0.25)
        self.assertEqual(by_model["prior_only"]["seeds"], "42,123,456")


if __name__ == "__main__":
    unittest.main()
