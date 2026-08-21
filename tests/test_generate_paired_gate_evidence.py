import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_paired_gate_evidence import (
    crossed_bootstrap_ci,
    evidence_class,
    load_records,
    trajectory_values,
)


def test_crossed_bootstrap_is_deterministic_and_preserves_direction():
    differences = np.asarray(
        [
            [-3.0, -2.0, -1.0, -2.5],
            [-2.0, -1.5, -0.5, -1.0],
            [-4.0, -3.0, -2.0, -3.5],
        ]
    )
    first = crossed_bootstrap_ci(differences, n_resamples=2_000, seed=17)
    second = crossed_bootstrap_ci(differences, n_resamples=2_000, seed=17)
    assert first == second
    assert first["mean_difference_m"] < 0.0
    assert first["ci_high_m"] < 0.0


@pytest.mark.parametrize(
    ("mean_difference", "holm_p", "ci_high", "seed_differences", "expected"),
    [
        (-1.0, 0.01, -0.1, [-1.0, -2.0, -0.5], "seed_and_trajectory_robust_improvement"),
        (-1.0, 0.01, 0.2, [-1.0, -2.0, 0.1], "trajectory_conditional_average_improvement_only"),
        (0.1, 0.01, 0.3, [0.1, -0.1, 0.3], "no_supported_average_improvement"),
    ],
)
def test_evidence_class_is_restrained(
    mean_difference, holm_p, ci_high, seed_differences, expected
):
    assert evidence_class(
        mean_difference, holm_p, ci_high, seed_differences
    ) == expected


def test_trajectory_values_rejects_declared_count_mismatch():
    record = {
        "run_id": "fixture",
        "eval_results": {
            "32": {
                "trajectory_count": 3,
                "trajectory_metrics": {
                    "10": {"window_count": 2, "ade": 1.0, "fde": 2.0},
                    "11": {"window_count": 2, "ade": 1.5, "fde": 2.5},
                },
            }
        },
    }
    with pytest.raises(ValueError, match="Declared trajectory count"):
        trajectory_values(record, 32, "ade")


def test_load_records_rejects_non_final_record(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    record = {
        "model_key": "full",
        "seed": 42,
        "evidence_tier": "diagnostic",
        "test_evaluation_performed": True,
        "formal_config_sha256": "config",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "unused-because-gate-fails-first",
    }
    path = tmp_path / "formal_sota_seed42_full_fixture.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-final record"):
        load_records(tmp_path, "formal_sota_seed*_full_*.json", "full", "config")
