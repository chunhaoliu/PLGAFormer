import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.formal_pipeline import DEFAULT_CONFIG, build_parser, build_status
from utils.formal_evidence import load_formal_config, resolve_config_path
from scripts.generate_taes_secondary_results import validate_robustness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_FORMAL_DATA = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_dataset_v2_1.npz"
)
LOCAL_MAIN_RECORDS = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
)
LOCAL_MAIN_CHECKPOINTS = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "trained_models"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
)
HAS_COMPLETE_LOCAL_EVIDENCE = (
    LOCAL_FORMAL_DATA.is_file()
    and len(list(LOCAL_MAIN_RECORDS.glob("formal_sota_*.json"))) >= 21
    and len(list(LOCAL_MAIN_CHECKPOINTS.glob("*.pth"))) >= 15
)
requires_complete_local_evidence = pytest.mark.skipif(
    not HAS_COMPLETE_LOCAL_EVIDENCE,
    reason="complete local formal records/checkpoints are excluded from public Git",
)


def test_formal_parser_exposes_all_commands():
    parser = build_parser()
    args = parser.parse_args(["status", "--json"])
    assert args.command == "status"
    assert args.config == DEFAULT_CONFIG


def test_formal_parser_exposes_four_registered_studies_and_aliases():
    parser = build_parser()
    for command in ("main", "mechanism", "robustness", "efficiency", "sota", "ablation"):
        args = parser.parse_args([command, "--dry-run"])
        assert args.command == command
        assert args.dry_run is True


def test_formal_config_registers_exactly_four_studies():
    config = load_formal_config(DEFAULT_CONFIG)
    assert tuple(config["studies"]) == ("main", "mechanism", "robustness", "efficiency")
    assert config["studies"]["mechanism"]["input_bundle"] == "main"
    assert config["studies"]["robustness"]["retrain"] is False
    assert config["studies"]["efficiency"]["retrain"] is False


@requires_complete_local_evidence
def test_status_is_read_only_and_reports_current_eligible_state():
    config = load_formal_config(DEFAULT_CONFIG)
    main_manifest = resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json"
    ablation_manifest = resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json"
    before = (main_manifest.read_bytes(), ablation_manifest.read_bytes())

    report = build_status(config)
    assert report["dataset"]["passed"] is True
    assert report["paper_eligible"] is True
    assert report["main_results"]["paper_eligible"] is True
    assert report["ablation"]["paper_eligible"] is True
    assert report["main_results"]["blockers"] == []
    assert report["ablation"]["blockers"] == []
    assert "pit" not in config["main_results"]["trainable_model_keys"]
    assert (main_manifest.read_bytes(), ablation_manifest.read_bytes()) == before

def test_run_py_formal_help_routes_without_training_imports():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "formal evidence" in result.stdout.lower()
    assert "status" in result.stdout
    assert "paper" in result.stdout


def test_run_py_without_arguments_prints_help_only():
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
    assert "Paper-facing route" in result.stdout


@requires_complete_local_evidence
def test_status_json_is_machine_readable():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "status", "--json"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "formal status"
    config = load_formal_config(DEFAULT_CONFIG)
    minimum_required = (
        len(config["main_results"]["trainable_model_keys"])
        * len(config["training"]["seeds"])
        + len(config["main_results"]["analytical_models"])
    )
    assert payload["main_results"]["record_count"] >= minimum_required
    assert payload["paper_eligible"] is True

@requires_complete_local_evidence
def test_paper_dry_run_accepts_complete_machine_readable_bundles():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "paper", "--json", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["paper_eligible"] is True
    assert Path(payload["main_bundle"]).is_file()
    assert Path(payload["ablation_bundle"]).is_file()

def test_aggregate_dry_run_is_read_only():
    config = load_formal_config(DEFAULT_CONFIG)
    main_manifest = resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json"
    ablation_manifest = resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json"
    before = tuple(
        path.read_bytes() if path.is_file() else None
        for path in (main_manifest, ablation_manifest)
    )
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "aggregate", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "formal aggregate --dry-run" in result.stdout
    after = tuple(
        path.read_bytes() if path.is_file() else None
        for path in (main_manifest, ablation_manifest)
    )
    assert after == before

def _robustness_payload(trajectory_count: int):
    models = ["Transformer (baseline)", "PLGAFormer (proposed)"]
    conditions = (("noise", 0.0), ("noise", 1.0), ("missing", 0.10), ("input_length", 128))
    results = {}
    for model in models:
        results[model] = {}
        for condition, value in conditions:
            results[model].setdefault(condition, {})[str(value)] = {
                "model_seeds": [42, 123, 456],
                "trajectory_count": trajectory_count,
                "ade_m": 1.0,
                "fde_m": 2.0,
                "ade_m_std": 0.1,
                "fde_m_std": 0.2,
            }
    audits = {
        model: {
            str(seed): ({"architecture": "final"} if "PLGAFormer" in model else {"run_signature": "main"})
            for seed in (42, 123, 456)
        }
        for model in models
    }
    return {
        "config": {"prediction_length": 256, "seq_len": 256, "run_signature": "final"},
        "dataset_sha256": load_formal_config(DEFAULT_CONFIG)["dataset"]["sha256"],
        "results": results,
        "checkpoint_audits": audits,
    }


def test_formal_v21_robustness_uses_360_and_rejects_legacy_102():
    valid = _robustness_payload(360)
    assert validate_robustness(valid, "final", "main", valid["dataset_sha256"], 360)
    with pytest.raises(RuntimeError, match="360"):
        validate_robustness(_robustness_payload(102), "final", "main", valid["dataset_sha256"], 360)


def test_formal_dynamics_shift_schema_uses_registered_conditions_and_bundle_identity():
    conditions = ("nominal", "aero_shift", "ballistic_shift")
    model_types = ("transformer", "plgaformer", "rotating_3dof")
    results = {
        condition: {
            model_type: {
                "model_seeds": [42] if model_type == "rotating_3dof" else [42, 123, 456],
                "trajectory_count": 360,
                "ade_m": 1.0,
                "ade_std_m": 0.1,
                "fde_m": 2.0,
                "fde_std_m": 0.2,
            }
            for model_type in model_types
        }
        for condition in conditions
    }
    payload = {
        "artifact_kind": "formal_v3_taes_dynamics_shift",
        "config_sha256": "config",
        "dataset_sha256": load_formal_config(DEFAULT_CONFIG)["dataset"]["sha256"],
        "base_exp1_signature": "main",
        "run_signature": "final",
        "seq_len": 256,
        "pred_len": 256,
        "test_trajectory_count": 360,
        "model_types": list(model_types),
        "results": results,
        "checkpoint_audits": {
            model_type: {str(seed): {} for seed in (42, 123, 456)}
            for model_type in ("transformer", "plgaformer")
        },
    }
    audited = validate_robustness(
        payload,
        "final",
        "main",
        payload["dataset_sha256"],
        360,
        config_sha256="config",
    )
    assert audited["__schema__"] == "formal_v3_dynamics_shift"
