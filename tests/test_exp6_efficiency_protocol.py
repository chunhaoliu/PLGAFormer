import json

from experiments.exp6_efficiency.efficiency_experiment import load_performance_reference


def test_efficiency_join_is_pinned_to_active_exp1_signature(tmp_path, monkeypatch):
    active_signature = "a" * 64
    stale_signature = "b" * 64
    payload = {
        "active_config": {"run_signature": active_signature},
        "runs": {
            "active": {
                "run_signature": active_signature,
                "seed": 42,
                "model_name": "PLGAFormer (proposed)",
                "results": {"256": {"mse": 0.2, "ade": 1200.0, "fde": 2500.0}},
            },
            "stale": {
                "run_signature": stale_signature,
                "seed": 42,
                "model_name": "PLGAFormer (proposed)",
                "results": {"256": {"mse": 0.01, "ade": 100.0, "fde": 200.0}},
            },
        },
    }
    path = tmp_path / "partial_runs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("HGV_EXP1_RUN_SIGNATURE", raising=False)

    reference, _, audit = load_performance_reference(path, allow_legacy_diagnostic=True)

    assert reference["PLGAFormer"]["MSE_256"] == 0.2
    assert reference["PLGAFormer"]["ADE_256"] == 1200.0
    assert reference["PLGAFormer"]["FDE_256"] == 2500.0
    assert audit["run_signature"] == active_signature
    assert audit["seeds_by_model"]["PLGAFormer"] == [42]
