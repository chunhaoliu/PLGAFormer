from __future__ import annotations

import pytest

from scripts.generate_taes_secondary_results import (
    ABLATION_ROWS,
    render_ablation_table,
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

    assert "Spherical prior + adaptive fusion" in table
    assert "Rotating-Earth prior + fixed schedule" in table
    assert r"\textbf{PLGAFormer (rotating-Earth prior + adaptive fusion)}" in table
