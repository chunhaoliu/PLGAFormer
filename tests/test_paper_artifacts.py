import csv
import tempfile
import unittest
from pathlib import Path


class PaperArtifactTests(unittest.TestCase):
    def test_extract_ablation_rows_computes_baseline_and_full_deltas(self):
        from scripts.generate_paper_artifacts import extract_ablation_rows

        payload = {
            "phases": {
                "best_candidate_ablation": {
                    "statistics": {
                        "Transformer (baseline)": {
                            "64": {
                                "mse": {"mean": 0.010, "std": 0.001, "n": 3},
                                "mae": {"mean": 0.080, "std": 0.002, "n": 3},
                                "rmse": {"mean": 0.100, "std": 0.003, "n": 3},
                            }
                        },
                        "PLGAFormer (A+B+C)": {
                            "64": {
                                "mse": {"mean": 0.006, "std": 0.0005, "n": 3},
                                "mae": {"mean": 0.050, "std": 0.001, "n": 3},
                                "rmse": {"mean": 0.077, "std": 0.002, "n": 3},
                            }
                        },
                        "PLGAFormer w/o A": {
                            "64": {
                                "mse": {"mean": 0.008, "std": 0.0004, "n": 3},
                                "mae": {"mean": 0.060, "std": 0.001, "n": 3},
                                "rmse": {"mean": 0.089, "std": 0.002, "n": 3},
                            }
                        },
                    }
                }
            }
        }

        rows = extract_ablation_rows(payload)
        by_model = {row["model"]: row for row in rows}

        self.assertEqual([row["model"] for row in rows][:3], [
            "Transformer (baseline)",
            "PLGAFormer (A+B+C)",
            "PLGAFormer w/o A",
        ])
        self.assertAlmostEqual(by_model["PLGAFormer (A+B+C)"]["mse_improvement_vs_baseline_pct"], 40.0)
        self.assertAlmostEqual(by_model["PLGAFormer w/o A"]["mse_delta_vs_full_pct"], 33.3333333333)
        self.assertEqual(by_model["Transformer (baseline)"]["n"], 3)

    def test_extract_ablation_rows_can_return_all_horizons(self):
        from scripts.generate_paper_artifacts import extract_ablation_rows

        payload = {
            "phases": {
                "best_candidate_ablation": {
                    "statistics": {
                        "Transformer (baseline)": {
                            "32": {"mse": {"mean": 0.010, "n": 1}},
                            "64": {"mse": {"mean": 0.020, "n": 1}},
                        },
                        "PLGAFormer (A+B+C)": {
                            "32": {"mse": {"mean": 0.005, "n": 1}},
                            "64": {"mse": {"mean": 0.012, "n": 1}},
                        },
                    }
                }
            }
        }

        rows = extract_ablation_rows(payload, horizon="all")

        self.assertEqual([(row["model"], row["horizon"]) for row in rows], [
            ("Transformer (baseline)", 32),
            ("PLGAFormer (A+B+C)", 32),
            ("Transformer (baseline)", 64),
            ("PLGAFormer (A+B+C)", 64),
        ])

    def test_extract_ablation_rows_accepts_formal_summary_payload(self):
        from scripts.generate_paper_artifacts import extract_ablation_rows

        payload = {
            "rows": [
                {
                    "model_key": "baseline",
                    "horizon": 32,
                    "metric": "mse",
                    "n": 3,
                    "mean": 0.010,
                    "std": 0.001,
                },
                {
                    "model_key": "baseline",
                    "horizon": 32,
                    "metric": "mae",
                    "n": 3,
                    "mean": 0.080,
                    "std": 0.002,
                },
                {
                    "model_key": "full",
                    "horizon": 32,
                    "metric": "mse",
                    "n": 3,
                    "mean": 0.006,
                    "std": 0.0005,
                },
                {
                    "model_key": "full",
                    "horizon": 32,
                    "metric": "mae",
                    "n": 3,
                    "mean": 0.050,
                    "std": 0.001,
                },
            ]
        }

        rows = extract_ablation_rows(payload, horizon="all")
        by_model = {row["model"]: row for row in rows}

        self.assertEqual(by_model["Transformer (baseline)"]["n"], 3)
        self.assertAlmostEqual(by_model["PLGAFormer (A+B+C)"]["mse"], 0.006)
        self.assertAlmostEqual(by_model["PLGAFormer (A+B+C)"]["mse_improvement_vs_baseline_pct"], 40.0)

    def test_extract_candidate_rows_keeps_baseline_and_ranked_candidates(self):
        from scripts.generate_paper_artifacts import extract_candidate_rows

        payload = {
            "rows": [
                {
                    "candidate_id": "baseline",
                    "model": "Transformer (baseline)",
                    "mse": 0.010,
                    "mae": 0.080,
                    "rmse": 0.100,
                    "baseline_improvement_pct": 0.0,
                }
            ],
            "ranked_candidates": [
                {
                    "rank": 1,
                    "candidate_id": "c02",
                    "learning_rate": 0.001,
                    "dropout": 0.05,
                    "alpha": 0.0005,
                    "mse": 0.006,
                    "mae": 0.050,
                    "rmse": 0.077,
                    "baseline_improvement_pct": 40.0,
                }
            ],
        }

        rows = extract_candidate_rows(payload)

        self.assertEqual(rows[0]["candidate_id"], "baseline")
        self.assertEqual(rows[1]["rank"], 1)
        self.assertEqual(rows[1]["candidate_id"], "c02")
        self.assertEqual(rows[1]["improvement_pct"], 40.0)

    def test_extract_sota_rows_accepts_formal_summary_payload(self):
        from scripts.generate_paper_artifacts import extract_sota_rows

        payload = {
            "rows": [
                {"model_key": "baseline", "horizon": 64, "metric": "mse", "n": 1, "mean": 0.010, "std": 0.0},
                {"model_key": "baseline", "horizon": 64, "metric": "mae", "n": 1, "mean": 0.080, "std": 0.0},
                {"model_key": "full", "horizon": 64, "metric": "mse", "n": 1, "mean": 0.004, "std": 0.0},
                {"model_key": "full", "horizon": 64, "metric": "mae", "n": 1, "mean": 0.040, "std": 0.0},
                {"model_key": "autoformer", "horizon": 64, "metric": "mse", "n": 1, "mean": 0.020, "std": 0.0},
            ]
        }

        rows = extract_sota_rows(payload, horizon="64")
        by_model = {row["model"]: row for row in rows}

        self.assertEqual([row["model"] for row in rows][:2], ["PLGAFormer (proposed)", "Transformer (baseline)"])
        self.assertAlmostEqual(by_model["PLGAFormer (proposed)"]["mse_improvement_vs_baseline_pct"], 60.0)
        self.assertAlmostEqual(by_model["Autoformer"]["mse"], 0.020)

    def test_extract_robustness_rows_flattens_saved_exp3_payload(self):
        from scripts.generate_paper_artifacts import extract_robustness_rows

        payload = {
            "config": {"prediction_length": 64},
            "results": {
                "PLGAFormer (proposed)": {
                    "noise": {
                        "0.1": {"mse": 0.003, "mae": 0.04},
                        "0.0": {"mse": 0.001, "mae": 0.02},
                    },
                    "input_length": {
                        "32": {"mse": 0.10, "mae": 0.20},
                        "64": {"mse": 0.01, "mae": 0.03},
                    },
                }
            },
        }

        rows = extract_robustness_rows(payload)

        self.assertEqual(
            [(row["scenario_type"], row["scenario_value"]) for row in rows],
            [("noise", 0.0), ("noise", 0.1), ("input_length", 32.0), ("input_length", 64.0)],
        )
        self.assertEqual(rows[0]["horizon"], 64)
        self.assertAlmostEqual(rows[1]["mse"], 0.003)

    def test_extract_physics_rows_computes_scaled_mse_gain(self):
        from scripts.generate_paper_artifacts import extract_physics_rows

        payload = {
            "config": {"prediction_length": 128},
            "results": {
                "Transformer (baseline)": {
                    "mse_scaled": 0.010,
                    "mse_physical": 100.0,
                    "violations": {
                        "height_violation_rate": 0.0,
                        "velocity_violation_rate": 0.02,
                        "velocity_relative_rmse": 0.40,
                        "acceleration_relative_rmse": 0.30,
                    },
                    "smoothness": {"position_smoothness_ratio": 1.30, "velocity_smoothness_ratio": 1.25},
                },
                "PLGAFormer (proposed)": {
                    "mse_scaled": 0.002,
                    "mse_physical": 20.0,
                    "violations": {
                        "height_violation_rate": 0.0,
                        "velocity_violation_rate": 0.01,
                        "velocity_relative_rmse": 0.20,
                        "acceleration_relative_rmse": 0.10,
                    },
                    "smoothness": {"position_smoothness_ratio": 1.10, "velocity_smoothness_ratio": 1.05},
                },
            },
        }

        rows = extract_physics_rows(payload)
        by_model = {row["model"]: row for row in rows}

        self.assertEqual([row["model"] for row in rows], ["PLGAFormer (proposed)", "Transformer (baseline)"])
        self.assertEqual(by_model["PLGAFormer (proposed)"]["horizon"], 128)
        self.assertAlmostEqual(by_model["PLGAFormer (proposed)"]["mse_scaled_improvement_vs_baseline_pct"], 80.0)
        self.assertAlmostEqual(by_model["PLGAFormer (proposed)"]["velocity_relative_rmse"], 0.20)
        self.assertAlmostEqual(by_model["Transformer (baseline)"]["position_smoothness_ratio"], 1.30)

    def test_write_csv_and_latex_tables(self):
        from scripts.generate_paper_artifacts import write_csv_table, write_latex_table

        rows = [
            {"model": "Transformer (baseline)", "mse": 0.01, "mae": 0.08},
            {"model": "PLGAFormer (A+B+C)", "mse": 0.006, "mae": 0.05},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = write_csv_table(tmp_path / "table.csv", rows, ["model", "mse", "mae"])
            tex_path = write_latex_table(
                tmp_path / "table.tex",
                rows,
                ["model", "mse", "mae"],
                caption="Ablation summary",
                label="tab:ablation",
            )

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))

            self.assertEqual(csv_rows[1]["model"], "PLGAFormer (A+B+C)")
            self.assertIn("\\caption{Ablation summary}", tex_path.read_text(encoding="utf-8"))

    def test_select_representative_indices_prefers_distinct_maneuvers(self):
        from scripts.generate_paper_artifacts import select_representative_indices

        labels = ["glide", "glide", "skip", "dive", "skip"]
        trajectory_ids = [10, 10, 11, 12, 11]
        window_starts = [30, 5, 9, 1, 40]

        indices = select_representative_indices(labels, trajectory_ids, window_starts, max_samples=3)

        self.assertEqual(indices, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
