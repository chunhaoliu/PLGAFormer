import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.formal_pipeline import (
    DEFAULT_CONFIG,
    _validate_data,
    build_parser,
    build_status,
    main as formal_main,
)
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


def test_formal_parser_exposes_exact_public_command_set():
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert tuple(subparsers.choices) == (
        "status",
        "validate-data",
        "main",
        "mechanism",
        "robustness",
        "efficiency",
        "aggregate",
        "paper",
        "audit",
    )
    args = parser.parse_args(["status", "--json"])
    assert args.command == "status"
    assert args.config == DEFAULT_CONFIG


def test_formal_parser_exposes_registered_studies_without_aliases():
    parser = build_parser()
    for command in ("main", "mechanism", "robustness", "efficiency"):
        args = parser.parse_args([command, "--dry-run"])
        assert args.command == command
        assert args.dry_run is True
    for retired_alias in ("sota", "ablation"):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([retired_alias, "--dry-run"])
        assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("command", "module_name", "user_args", "expected_prefix"),
    [
        (
            "main",
            "scripts.run_formal_sota_unit",
            ["--models", "plgaformer", "--selection-only"],
            ["--config", str(DEFAULT_CONFIG)],
        ),
        (
            "mechanism",
            "scripts.run_formal_ablation_unit",
            ["--models", "schedule_only", "--selection-only"],
            [
                "--phase",
                "phase4_final_mechanism_controls",
                "--config",
                str(DEFAULT_CONFIG),
            ],
        ),
    ],
)
def test_formal_runner_arguments_preserve_user_order(
    monkeypatch, command, module_name, user_args, expected_prefix
):
    recorded = []
    module = types.ModuleType(module_name)

    def fake_main(argv):
        recorded.append(list(argv))
        return 0

    module.main = fake_main
    monkeypatch.setitem(sys.modules, module_name, module)
    assert formal_main([command, *user_args]) == 0
    assert recorded == [[*expected_prefix, *user_args]]


@pytest.mark.parametrize(
    ("command", "module_name"),
    [
        ("main", "scripts.run_formal_sota_unit"),
        ("mechanism", "scripts.run_formal_ablation_unit"),
    ],
)
def test_formal_runner_help_delegates_to_downstream_parser(
    monkeypatch, command, module_name
):
    recorded = []
    module = types.ModuleType(module_name)
    module.main = lambda argv: recorded.append(list(argv)) or 0
    monkeypatch.setitem(sys.modules, module_name, module)
    assert formal_main([command, "--help"]) == 0
    assert recorded == [["--help"]]


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


def test_run_py_exposes_only_formal():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "formal" in result.stdout
    assert "--task" not in result.stdout
    for legacy in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"):
        assert legacy not in result.stdout


def test_formal_help_has_no_duplicate_aliases():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "main" in result.stdout and "mechanism" in result.stdout
    assert "sota" not in result.stdout and "ablation" not in result.stdout


def test_run_py_rejects_unknown_first_command():
    result = subprocess.run(
        [sys.executable, "run.py", "exp1"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr.lower()


@pytest.mark.parametrize(
    "retired_args",
    [
        ["--task", "exp1"],
        ["formal", "sota"],
        ["formal", "ablation"],
    ],
)
def test_run_py_rejects_retired_public_routes(retired_args):
    result = subprocess.run(
        [sys.executable, "run.py", *retired_args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_agent_instructions_teach_the_exact_public_command_surface():
    instructions = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for command in (
        "status --json",
        "validate-data --json",
        "main [explicit runner options]",
        "mechanism [explicit runner options]",
        "robustness --dry-run",
        "efficiency --dry-run",
        "aggregate --kind main --dry-run",
        "aggregate --kind ablation --dry-run",
        "audit --json",
        "paper --dry-run --json",
    ):
        assert f"python run.py formal {command}" in instructions
    assert "python run.py --task" not in instructions
    assert "python run.py formal ablation" not in instructions
    assert (
        "Optional secondary efficiency inspection: "
        "`python run.py formal efficiency --dry-run`"
    ) in instructions


def test_agent_instructions_do_not_present_archival_branches_as_active_work():
    instructions = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8").lower()
    for retired_term in (
        "exp1",
        "exp2",
        "exp3",
        "exp4",
        "exp5",
        "exp6",
        "exp7",
        "--task",
        "--quick",
        "formal ablation",
        "pit",
        "af-ciln",
        "3-dof",
        "runnable but pending",
        "regenerated quickly",
    ):
        assert retired_term not in instructions
    assert (
        "numbered experiment source and outputs are archival/non-active provenance"
        in instructions
    )
    assert 'create_registered_model("plgaformer", ...)' in instructions
    assert "later structure review" not in instructions


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
    assert "Active route: python run.py formal --help" in result.stdout


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
    assert result.returncode in {0, 1}
    assert "formal aggregate --dry-run" in result.stdout
    payload = json.loads(result.stdout)
    assert payload["requested_kind"] == "all"
    assert payload["requested_eligible"] is (result.returncode == 0)
    after = tuple(
        path.read_bytes() if path.is_file() else None
        for path in (main_manifest, ablation_manifest)
    )
    assert after == before


@pytest.mark.parametrize(
    ("kind", "main_eligible", "ablation_eligible", "expected_code"),
    [
        ("main", True, False, 0),
        ("main", False, True, 1),
        ("ablation", True, True, 0),
        ("ablation", True, False, 1),
        ("all", True, True, 0),
        ("all", True, False, 1),
    ],
)
def test_aggregate_dry_run_is_a_kind_specific_fail_closed_gate(
    monkeypatch,
    capsys,
    kind,
    main_eligible,
    ablation_eligible,
    expected_code,
):
    monkeypatch.setattr(
        "scripts.formal_pipeline.build_status",
        lambda config: {
            "command": "formal status",
            "dataset": {"passed": True},
            "main_results": {"paper_eligible": main_eligible},
            "ablation": {"paper_eligible": ablation_eligible},
            "paper_eligible": main_eligible and ablation_eligible,
        },
    )
    assert formal_main(["aggregate", "--kind", kind, "--dry-run"]) == expected_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_kind"] == kind
    assert payload["requested_eligible"] is (expected_code == 0)


def test_validate_data_json_is_one_document_on_success(monkeypatch, capsys):
    import data_provider.validation as trajectory_validator
    import scripts.validate_multiregime_dataset as multiregime_validator

    def fake_trajectory_validation(**kwargs):
        print("captured trajectory validator note")
        return {"passed": True, "validator": "trajectory"}

    def fake_multiregime_validation(argv):
        print(json.dumps({"passed": True, "validator": "multiregime"}))
        return 0

    monkeypatch.setattr(
        trajectory_validator,
        "validate_hgv_dataset",
        fake_trajectory_validation,
    )
    monkeypatch.setattr(multiregime_validator, "main", fake_multiregime_validation)
    config = load_formal_config(DEFAULT_CONFIG)
    assert _validate_data(config, True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "formal validate-data"
    assert payload["passed"] is True
    assert payload["trajectory_validation"]["report"]["validator"] == "trajectory"
    assert payload["multiregime_validation"]["report"]["validator"] == "multiregime"
    assert payload["multiregime_validation"]["exit_code"] == 0
    assert payload["dataset"]["expected_sha256"] == config["dataset"]["sha256"]


def test_validate_data_json_is_one_document_on_failure(monkeypatch, capsys):
    import data_provider.validation as trajectory_validator

    def fail_trajectory_validation(**kwargs):
        raise FileNotFoundError("frozen dataset missing")

    monkeypatch.setattr(
        trajectory_validator,
        "validate_hgv_dataset",
        fail_trajectory_validation,
    )
    config = load_formal_config(DEFAULT_CONFIG)
    assert _validate_data(config, True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is False
    assert payload["trajectory_validation"]["error_type"] == "FileNotFoundError"
    assert payload["multiregime_validation"]["skipped"] is True

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
