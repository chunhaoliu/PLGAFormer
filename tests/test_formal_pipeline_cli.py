import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.formal_pipeline import DEFAULT_CONFIG, build_parser, build_status
from utils.formal_evidence import load_formal_config, resolve_config_path
from scripts.generate_taes_secondary_results import validate_robustness


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_status_is_read_only_and_reports_expected_current_blockers():
    config_path = DEFAULT_CONFIG
    config = __import__("utils.formal_evidence", fromlist=["load_formal_config"]).load_formal_config(config_path)
    main_manifest = resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json"
    ablation_manifest = resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json"
    assert not main_manifest.exists()
    assert not ablation_manifest.exists()

    report = build_status(config)
    assert report["dataset"]["passed"] is True
    assert report["paper_eligible"] is False
    missing = {(item["context"]["model_key"], item["context"]["seed"]) for item in report["main_results"]["blockers"] if item["code"] == "MISSING_REQUIRED_RUN"}
    assert ("full", 456) in missing
    assert {("pit", 42), ("pit", 123), ("pit", 456)}.issubset(missing)
    assert not main_manifest.exists()
    assert not ablation_manifest.exists()


def test_run_py_formal_help_routes_without_training_imports():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "formal-v3" in result.stdout
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
    assert payload["main_results"]["record_count"] == 23

def test_paper_dry_run_refuses_with_machine_readable_blockers():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "paper", "--json", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["paper_eligible"] is False
    codes = {item["code"] for item in payload["blockers"]}
    assert "MISSING_REQUIRED_RUN" in codes
    assert "MISSING_REQUIRED_ABLATION_RUN" in codes

def test_aggregate_dry_run_is_read_only():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "aggregate", "--dry-run"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "formal aggregate --dry-run" in result.stdout
    config = __import__("utils.formal_evidence", fromlist=["load_formal_config"]).load_formal_config(DEFAULT_CONFIG)
    assert not (resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json").exists()
    assert not (resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json").exists()



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
