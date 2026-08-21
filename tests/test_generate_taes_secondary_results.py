from __future__ import annotations

import pytest

from scripts.generate_taes_secondary_results import (
    ABLATION_ROWS,
    EFFICIENCY_MODELS,
    render_ablation_table,
    render_efficiency_table,
    validate_efficiency,
)


def test_efficiency_validation_rejects_quick_run():
    payload = {
        "metadata": {
            "formal": False,
            "performance_reference_audit": {"run_signature": "sig"},
        },
        "results": [],
    }

    with pytest.raises(RuntimeError, match="not a formal"):
        validate_efficiency(payload, "sig")


def test_ablation_table_includes_final_mechanism_controls():
    lookup = {}
    for phase, model_key, _ in ABLATION_ROWS:
        for metric in ("ade", "fde"):
            lookup[(phase, model_key, 256, metric)] = {
                "mean": 1000.0,
                "std": 100.0,
            }

    table = render_ablation_table(lookup)

    assert "Transformer" not in table
    assert "Spherical prior + adaptive fusion" in table
    assert "Rotating-Earth prior + fixed schedule" in table
    assert r"\textbf{PLGAFormer (rotating-Earth prior + adaptive fusion)}" in table


def test_ablation_table_can_include_hash_bound_capacity_control():
    lookup = {}
    for phase, model_key, _ in ABLATION_ROWS:
        for metric in ("ade", "fde"):
            lookup[(phase, model_key, 256, metric)] = {
                "mean": 1000.0,
                "std": 100.0,
            }
    capacity = {
        "ade": {"mean": 2000.0, "std": 200.0},
        "fde": {"mean": 3000.0, "std": 300.0},
    }

    table = render_ablation_table(lookup, capacity)

    assert "PLGAFormer learned-only backbone" in table


def test_efficiency_table_covers_all_formal_models_and_cost_fields():
    rows = [
        {
            "model": name,
            "params": 1_000_000,
            "flops_mflops": 10.0,
            "peak_inference_memory_mb": 20.0,
            "latency_batch1_ms": 3.0,
            "throughput_trajectories_s": 4.0,
            "ADE_256_m": 5_000.0,
            "FDE_256_m": 6_000.0,
            "RMSE_256_m": 7_000.0,
        }
        for name in EFFICIENCY_MODELS
    ]

    table = render_efficiency_table(rows, {"device_name": "GPU"})

    for name in EFFICIENCY_MODELS:
        assert name in table
    assert "Spherical kinematics" not in table
    assert "Rotating-Earth 3-DOF" not in table
    assert "MFLOPs" in table
    assert "Peak memory (MB)" in table
