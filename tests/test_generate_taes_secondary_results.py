from __future__ import annotations

import pytest

from scripts.generate_taes_secondary_results import (
    ABLATION_ROWS,
    EFFICIENCY_MODELS,
    render_ablation_table,
    render_efficiency_table,
    render_robustness_table,
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
        for metric in ("ade", "fde", "rmse_cart_m"):
            lookup[(phase, model_key, 256, metric)] = {
                "mean": 1000.0,
                "std": 100.0,
            }

    table = render_ablation_table(lookup)

    assert "Transformer" not in table
    assert "Spherical prior + adaptive fusion" in table
    assert "Rotating-Earth prior + fixed schedule" in table
    assert r"\textbf{PLGAFormer (rotating-Earth prior + adaptive fusion)}" in table
    assert "RMSE (km)" in table


def test_ablation_table_can_include_hash_bound_capacity_control():
    lookup = {}
    for phase, model_key, _ in ABLATION_ROWS:
        for metric in ("ade", "fde", "rmse_cart_m"):
            lookup[(phase, model_key, 256, metric)] = {
                "mean": 1000.0,
                "std": 100.0,
            }
    capacity = {
        "ade": {"mean": 2000.0, "std": 200.0},
        "fde": {"mean": 3000.0, "std": 300.0},
        "rmse_cart_m": {"mean": 4000.0, "std": 400.0},
    }

    table = render_ablation_table(lookup, capacity)

    assert "PLGAFormer learned-only backbone" in table
    assert r"\resultnum" in table
    assert r"\begin{table*}" in table
    assert "RMSE (km)" in table
    assert r"$\Delta$ADE" not in table
    assert r"$+100.0$\%" not in table


def test_efficiency_table_is_compact_and_feasibility_focused():
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
    assert "Latency (ms)" in table
    assert "Peak memory (MB)" not in table
    assert "Traj./s" not in table
    assert r"ADE$_{256}$" not in table


def test_robustness_table_puts_nominal_first_and_bolds_plgaformer():
    results = {
        "__schema__": "formal_v3_dynamics_shift",
        "results": {
            "nominal": {
                "transformer": {"ade_m": 15073.0, "ade_std_m": 1314.0, "fde_m": 28879.0, "fde_std_m": 1231.0},
                "plgaformer": {"ade_m": 4776.0, "ade_std_m": 223.0, "fde_m": 14225.0, "fde_std_m": 1392.0},
            },
            "aero_shift": {
                "transformer": {"ade_m": 20130.0, "ade_std_m": 1689.0, "fde_m": 38904.0, "fde_std_m": 1710.0},
                "plgaformer": {"ade_m": 4775.0, "ade_std_m": 307.0, "fde_m": 12796.0, "fde_std_m": 1112.0},
            },
            "ballistic_shift": {
                "transformer": {"ade_m": 15869.0, "ade_std_m": 1214.0, "fde_m": 30597.0, "fde_std_m": 782.0},
                "plgaformer": {"ade_m": 4952.0, "ade_std_m": 21.0, "fde_m": 13142.0, "fde_std_m": 348.0},
            },
        },
    }

    table = render_robustness_table(results)
    nominal_at = table.find("Nominal")
    aero_at = table.find(r"$C_L-10\%$, $C_D+10\%$")
    mass_at = table.find(r"$m+15\%$, $S-10\%$")
    assert 0 < nominal_at < aero_at < mass_at
    assert table.count(r"\textbf{\resultnum{4.776}") == 1
    assert "Rotating-Earth 3-DOF" not in table
    assert r"\begin{table*}" in table
    assert "ADE (km)" in table
    assert "FDE (km)" in table
