# HGV Mainline Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the HGV repository around one audited dataset, one selected PLGAFormer contract, one formal command surface, and one protected Main/Ablation evidence chain while archiving inactive versions reversibly.

**Architecture:** A small standard-library contract owns active paths and PLGAFormer flags. Data, model, evidence, and runner code consume that contract. Historical movement occurs only after active imports are narrowed and is allowlist-based, hash-attested, dry-run-first, and prohibited from touching verified final evidence roots.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, JSON evidence manifests, Git, Windows PowerShell.

---

## Fixed execution environment and protected evidence

Run Python validation from the repository root with:

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest
```

Do not use the Windows Store Python alias. Do not push commits or tags. The recovery point is `before-mainline-unification-20260822` at `f19c4c2`.

These paths must not be moved, regenerated, or overwritten:

```text
configs/formal_v3.json
data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz
experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp1_sota/trained_models/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp2_ablation/trained_models/formal_v3/hgv_multiregime_state_v2_1/final
```

## File responsibility map

- `utils/mainline_contract.py`: active config, dataset path, model keys, and selected PLGAFormer flags.
- `utils/formal_runtime.py`: frozen runner defaults and incompatible-override rejection.
- `scripts/audit_mainline_inventory.py`: read-only active/archive classification and hashes.
- `scripts/archive_mainline_artifacts.py`: allowlisted archival with dry-run and restoration manifest.
- `tests/test_mainline_contract.py`: data/model/config single-source invariants.
- `tests/test_mainline_inventory.py`: archive containment and protected-root tests.
- `run.py` and `scripts/formal_pipeline.py`: sole command surface.
- `docs/ACTIVE_MAINLINE.md`: human-readable active data/model/evidence map.

### Task 1: Establish the active mainline contract

**Files:**
- Create: `utils/mainline_contract.py`
- Create: `tests/test_mainline_contract.py`
- Modify: `utils/formal_evidence.py`
- Modify: `utils/final_plgaformer.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from pathlib import Path

from utils.mainline_contract import (
    ACTIVE_CONFIG_PATH,
    ACTIVE_DATASET_RELATIVE_PATH,
    ACTIVE_MODEL_KEY,
    ACTIVE_PLGAFORMER_FLAGS,
    load_mainline_config,
)


def test_mainline_contract_resolves_the_frozen_dataset():
    config = load_mainline_config()
    assert ACTIVE_CONFIG_PATH.name == "formal_v3.json"
    assert config["dataset"]["protocol"] == "hgv_multiregime_state_v2_1"
    assert Path(config["dataset"]["relative_path"]) == ACTIVE_DATASET_RELATIVE_PATH
    assert config["dataset"]["sha256"] == (
        "526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7"
    )


def test_mainline_contract_has_one_selected_plgaformer():
    assert ACTIVE_MODEL_KEY == "full"
    assert dict(ACTIVE_PLGAFORMER_FLAGS) == {
        "use_sparse_attention": False,
        "use_physics_corrector": False,
        "use_multi_head_output": True,
        "use_prior_fusion": True,
        "use_channel_residual": False,
        "prior_type": "rotating_3dof",
        "prior_blend_mode": "adaptive",
    }
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run:

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'utils.mainline_contract'`.

- [ ] **Step 3: Implement the contract**

Create `utils/mainline_contract.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "formal_v3.json"
ACTIVE_DATASET_RELATIVE_PATH = Path(
    "data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz"
)
ACTIVE_MODEL_KEY = "full"
ACTIVE_PLGAFORMER_FLAGS = MappingProxyType({
    "use_sparse_attention": False,
    "use_physics_corrector": False,
    "use_multi_head_output": True,
    "use_prior_fusion": True,
    "use_channel_residual": False,
    "prior_type": "rotating_3dof",
    "prior_blend_mode": "adaptive",
})
ACTIVE_TRAINABLE_MODEL_KEYS = (
    "baseline", "full", "dlinear", "patchtst", "itransformer"
)


def load_mainline_config() -> dict[str, Any]:
    payload = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(payload["dataset"]["relative_path"]) != ACTIVE_DATASET_RELATIVE_PATH:
        raise RuntimeError("Active dataset path diverges from the mainline contract.")
    if tuple(payload["main_results"]["trainable_model_keys"]) != ACTIVE_TRAINABLE_MODEL_KEYS:
        raise RuntimeError("Active model matrix diverges from the mainline contract.")
    return payload
```

Import `ACTIVE_PLGAFORMER_FLAGS` in `utils/formal_evidence.py` and assign `FINAL_MODEL_FLAGS = ACTIVE_PLGAFORMER_FLAGS`. Import the same constant in `utils/final_plgaformer.py`; set `FINAL_MODEL_KEY = ACTIVE_MODEL_KEY` and return `dict(ACTIVE_PLGAFORMER_FLAGS)` from `final_plgaformer_kwargs()`.

- [ ] **Step 4: Run targeted tests**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py tests/test_formal_evidence.py tests/test_formal_physics_model.py -q
```

Expected: all selected tests pass; complete-local tests may skip only when documented local artifacts are absent.

- [ ] **Step 5: Commit**

```powershell
git add utils/mainline_contract.py utils/formal_evidence.py utils/final_plgaformer.py tests/test_mainline_contract.py
git commit -m "refactor: centralize the HGV mainline contract"
```

### Task 2: Bind dataset and PLGAFormer construction to the contract

**Files:**
- Modify: `data_generation/data_paths.py`
- Modify: `models/model_factory.py`
- Modify: `tests/test_mainline_contract.py`

- [ ] **Step 1: Add failing authority tests**

```python
def test_data_paths_use_the_mainline_dataset_constant():
    from data_generation.data_paths import get_dataset_npz_path
    from utils.mainline_contract import PROJECT_ROOT
    assert get_dataset_npz_path(PROJECT_ROOT) == PROJECT_ROOT / ACTIVE_DATASET_RELATIVE_PATH


def test_factory_reconstructs_the_selected_plgaformer_flags():
    from models.model_factory import create_registered_model
    model = create_registered_model("plgaformer", input_dim=6, device="cpu")
    for name, expected in ACTIVE_PLGAFORMER_FLAGS.items():
        assert getattr(model, name) == expected
```

- [ ] **Step 2: Run tests and confirm duplicated defaults are exposed**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py -q
```

Expected: at least one new assertion fails before wiring.

- [ ] **Step 3: Wire the active path and flags**

Use `ACTIVE_DATASET_RELATIVE_PATH` in `get_dataset_npz_path()`. In the PLGAFormer factory branch, build the constructor dictionary from the existing dimensions plus `**dict(ACTIVE_PLGAFORMER_FLAGS)`, then apply explicit candidate kwargs last. This preserves diagnostic overrides without changing the default model.

- [ ] **Step 4: Validate data and model reconstruction**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py tests/test_data_provider_and_validation.py tests/test_plgaformer_innovations.py tests/test_formal_physics_model.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```powershell
git add data_generation/data_paths.py models/model_factory.py tests/test_mainline_contract.py
git commit -m "refactor: bind data and PLGAFormer to the mainline"
```

### Task 3: Derive formal runner settings from the frozen contract

**Files:**
- Create: `utils/formal_runtime.py`
- Create: `tests/test_formal_runtime.py`
- Modify: `scripts/run_formal_sota_unit.py`
- Modify: `scripts/run_formal_ablation_unit.py`
- Modify: `tests/test_formal_sota_unit.py`
- Modify: `tests/test_formal_ablation_unit.py`

- [ ] **Step 1: Write failing runtime tests**

```python
from argparse import Namespace
import pytest
from utils.mainline_contract import load_mainline_config
from utils.formal_runtime import formal_runtime_defaults, validate_formal_runtime


def test_runtime_defaults_match_the_frozen_contract():
    defaults = formal_runtime_defaults(load_mainline_config())
    assert defaults["seeds"] == "42,123,456"
    assert defaults["epochs"] == 50
    assert defaults["batch_size"] == 128
    assert defaults["prediction_length"] == 256
    assert defaults["prediction_horizons"] == "32,64,128,256"
    assert defaults["label_len"] == 128
    assert defaults["amp"] is False


def test_runtime_rejects_drift_but_allows_seed_subsets():
    config = load_mainline_config()
    defaults = formal_runtime_defaults(config)
    validate_formal_runtime(Namespace(**dict(defaults, seeds="42")), config)
    with pytest.raises(ValueError, match="prediction_length"):
        validate_formal_runtime(Namespace(**dict(defaults, prediction_length=128)), config)
    with pytest.raises(ValueError, match="seeds"):
        validate_formal_runtime(Namespace(**dict(defaults, seeds="999")), config)
```

- [ ] **Step 2: Verify the missing-module failure**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_formal_runtime.py -q
```

Expected: collection fails because `utils.formal_runtime` does not exist.

- [ ] **Step 3: Implement defaults and validation**

`formal_runtime_defaults(config)` must map `task` and `training` fields to the current runner argument names, including seeds, epochs, batch size, horizons, supervision/evaluation protocols, optimizer values, patience, workers, precision, and prior-cache settings. Implement seed-subset validation and reject every incompatible non-seed field:

```python
allowed = tuple(int(value) for value in config["training"]["seeds"])
requested = tuple(int(value) for value in str(args.seeds).split(","))
if not requested or any(seed not in allowed for seed in requested):
    raise ValueError(f"seeds must be a non-empty subset of {allowed}")
for field, expected in formal_runtime_defaults(config).items():
    if field != "seeds" and getattr(args, field) != expected:
        raise ValueError(f"{field} conflicts with frozen value {expected!r}")
```

Use those values as parser defaults in both unit runners and call validation before importing training code.

- [ ] **Step 4: Run runner tests**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_formal_runtime.py tests/test_formal_sota_unit.py tests/test_formal_ablation_unit.py tests/test_formal_pipeline_cli.py -q
```

Expected: all selected tests pass and no training starts.

- [ ] **Step 5: Commit**

```powershell
git add utils/formal_runtime.py scripts/run_formal_sota_unit.py scripts/run_formal_ablation_unit.py tests/test_formal_runtime.py tests/test_formal_sota_unit.py tests/test_formal_ablation_unit.py
git commit -m "refactor: enforce frozen formal runtime settings"
```

### Task 4: Reduce the active model registry

**Files:**
- Modify: `models/model_factory.py`
- Modify: `models/__init__.py`
- Modify: `experiments/overall_prediction/main_results.py`
- Modify: `scripts/run_formal_sota_unit.py`
- Modify: `tests/test_formal_sota_unit.py`
- Modify: `tests/test_model_provenance.py`
- Modify: `tests/test_mainline_contract.py`

- [ ] **Step 1: Add a failing exact-registry test**

```python
def test_active_model_registry_is_exact():
    from models.model_factory import get_supported_model_types
    assert get_supported_model_types() == (
        "transformer", "kinematic", "rotating_3dof", "dlinear",
        "plgaformer", "patchtst", "itransformer",
    )
```

- [ ] **Step 2: Confirm inactive registrations fail the test**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py::test_active_model_registry_is_exact -q
```

Expected: FAIL because inactive models and aliases are still registered.

- [ ] **Step 3: Narrow the registry and imports**

Keep only the exact registry above. Remove eager PIT, AF-CILN, and catch-all `sota_models` imports. End unsupported construction with:

```python
raise ValueError(
    f"Unsupported active model type: {model_type!r}; "
    f"supported={get_supported_model_types()}"
)
```

Keep only Transformer, PLGAFormer, spherical kinematics, rotating-Earth 3-DOF, DLinear, PatchTST, and iTransformer in `COMPARISON_MODELS`. Remove inactive CLI options and PIT-specific training loss. Retain historical display names only where evidence normalization needs to read old records.

- [ ] **Step 4: Replace inactive tests with active-matrix assertions**

Remove AF-CILN parser/reconstruction, PIT identity, inactive matrix, and obsolete A/B/C candidate-file tests. Assert that the default formal trainable set is exactly Transformer, PLGAFormer, DLinear, PatchTST, and iTransformer.

- [ ] **Step 5: Run model/evidence tests**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_contract.py tests/test_formal_sota_unit.py tests/test_model_provenance.py tests/test_public_baselines.py tests/test_formal_evidence.py -q
```

Expected: all selected tests pass and historical final records remain readable.

- [ ] **Step 6: Commit**

```powershell
git add models/model_factory.py models/__init__.py experiments/overall_prediction/main_results.py scripts/run_formal_sota_unit.py tests/test_formal_sota_unit.py tests/test_model_provenance.py tests/test_mainline_contract.py
git commit -m "refactor: limit models to the HGV paper mainline"
```

### Task 5: Expose one command surface

**Files:**
- Modify: `run.py`
- Modify: `scripts/formal_pipeline.py`
- Modify: `tests/test_formal_pipeline_cli.py`
- Create: `docs/ACTIVE_MAINLINE.md`
- Modify: `README.md`

- [ ] **Step 1: Add failing CLI tests**

```python
def test_run_py_exposes_only_formal():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "formal" in result.stdout
    assert "--task" not in result.stdout
    for legacy in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"):
        assert legacy not in result.stdout


def test_formal_help_has_no_duplicate_aliases():
    result = subprocess.run(
        [sys.executable, "run.py", "formal", "--help"], cwd=PROJECT_ROOT,
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "main" in result.stdout and "mechanism" in result.stdout
    assert "sota" not in result.stdout and "ablation" not in result.stdout
```

- [ ] **Step 2: Confirm legacy help fails the tests**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_formal_pipeline_cli.py -q
```

Expected: new tests fail on `--task`, numbered experiments, or aliases.

- [ ] **Step 3: Simplify the dispatchers**

Make `run.py` accept only `formal`; unknown first arguments return code 2. Remove `sota` and `ablation` aliases from `scripts/formal_pipeline.py`. Keep `main`, `mechanism`, `status`, `validate-data`, `aggregate`, `robustness`, `efficiency`, `paper`, and `audit`.

- [ ] **Step 4: Document one route**

Create `docs/ACTIVE_MAINLINE.md` containing the exact dataset path/hash, active model flags, formal commands, protected evidence roots, and external Archive root. Remove numbered workflow and inactive launch instructions from README; mention them once as archived provenance.

- [ ] **Step 5: Validate CLI and status**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_formal_pipeline_cli.py -q
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py --help
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal status --json
```

Expected: one route is shown and status remains paper-eligible.

- [ ] **Step 6: Commit**

```powershell
git add run.py scripts/formal_pipeline.py tests/test_formal_pipeline_cli.py docs/ACTIVE_MAINLINE.md README.md
git commit -m "refactor: expose one formal experiment entrypoint"
```

### Task 6: Add reversible archive tooling

**Files:**
- Create: `scripts/audit_mainline_inventory.py`
- Create: `scripts/archive_mainline_artifacts.py`
- Create: `tests/test_mainline_inventory.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing archive-safety tests**

```python
def test_archive_plan_excludes_protected_paths(tmp_path):
    from scripts.archive_mainline_artifacts import build_archive_plan
    project, archive = tmp_path / "project", tmp_path / "archive"
    protected = project / "experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final/result.json"
    legacy = project / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    protected.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    protected.write_text("formal", encoding="utf-8")
    legacy.write_bytes(b"legacy")
    sources = {item.source for item in build_archive_plan(project, archive)}
    assert protected.resolve() not in sources
    assert legacy.resolve() in sources


def test_archive_apply_preserves_bytes_and_hash(tmp_path):
    import hashlib
    from scripts.archive_mainline_artifacts import apply_archive_plan, build_archive_plan
    project, archive = tmp_path / "project", tmp_path / "archive"
    source = project / "data_generation/data/processed/hgv_multiregime_dataset_v2.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"historical")
    manifest = apply_archive_plan(build_archive_plan(project, archive))
    archived = archive / source.relative_to(project)
    assert not source.exists() and archived.read_bytes() == b"historical"
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"historical").hexdigest()
```

- [ ] **Step 2: Confirm the missing implementation failure**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_inventory.py -q
```

Expected: collection fails because the archive module does not exist.

- [ ] **Step 3: Implement explicit allowlists and guards**

Use these generated-artifact allowlists:

```python
ARCHIVE_FILES = (
    "data_generation/data/processed/hgv_multiregime_dataset_v2.npz",
    "data_generation/data/processed/hgv_multiregime_pilot_v2.npz",
    "data_generation/data/processed/hgv_multiregime_pilot_v2_1.npz",
    "data_generation/data/processed/hgv_trajectory_dataset.npz",
    "data_generation/data/processed/pit_aligned_radar_dataset.npz",
    "data_generation/data/processed/pit_aligned_radar_pilot.npz",
)
ARCHIVE_TREES = (
    "data_generation/data/processed/derived_window_views",
    "tmp/AF-CILN",
    "tmp/robustness_protocol_view",
)
```

Define the protected prefixes from the plan header. Resolve sources under the project root and destinations under the explicit archive root; reject path escape, overlap, duplicate entries, and destination mismatch. Default CLI is JSON dry-run. Require `--apply` to move. Hash before and after movement and atomically write `archive_manifest.json` only after all matches pass.

- [ ] **Step 4: Implement the read-only inventory wrapper and ignore previews**

`audit_mainline_inventory.py` reports active, protected, archive-candidate, missing, and unclassified paths. Add `tmp/mainline_inventory/` to `.gitignore`.

- [ ] **Step 5: Test and dry-run against the real workspace**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest tests/test_mainline_inventory.py -q
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' scripts/audit_mainline_inventory.py --project-root 'D:\Research\HGV_Code\HGVTP_PLGAformer-main' --archive-root 'D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main'
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' scripts/archive_mainline_artifacts.py --project-root 'D:\Research\HGV_Code\HGVTP_PLGAformer-main' --archive-root 'D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main'
```

Expected: tests pass; real commands remain read-only; no protected path appears in the plan.

- [ ] **Step 6: Commit tooling before movement**

```powershell
git add scripts/audit_mainline_inventory.py scripts/archive_mainline_artifacts.py tests/test_mainline_inventory.py .gitignore
git commit -m "feat: add reversible HGV archive tooling"
```

### Task 7: Archive inactive generated artifacts and tracked source

**Files:**
- Archive externally: Task 6 allowlisted generated artifacts
- Remove after verified copy: explicit inactive tracked files below
- Modify: `docs/ACTIVE_MAINLINE.md`

- [ ] **Step 1: Capture pre-move status and evidence**

```powershell
git status --short --branch
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal status --json
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal audit --json
```

Expected: clean worktree; data, Main, and Ablation pass.

- [ ] **Step 2: Apply only the generated allowlist**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' scripts/archive_mainline_artifacts.py --project-root 'D:\Research\HGV_Code\HGVTP_PLGAformer-main' --archive-root 'D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main' --apply
```

Expected: every moved file has a matching hash in the external manifest; protected roots are unchanged.

- [ ] **Step 3: Re-run formal status**

```powershell
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal status --json
```

Expected: `paper_eligible` is `true`.

- [ ] **Step 4: Copy, verify, then remove inactive tracked source**

Archive these paths under `D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main\tracked_source` with repository-relative layout:

```text
models/PIT.py
models/external_baselines.py
models/sota_models.py
scripts/generate_pit_radar_dataset.py
scripts/validate_pit_radar_dataset.py
scripts/run_all_enhancements.py
scripts/run_all_experiments.py
scripts/run_best_candidate_ablation.py
scripts/run_plgaformer_candidate_search.py
scripts/run_exp1_sota.py
scripts/run_exp2_ablation.py
scripts/run_exp3_robustness.py
scripts/run_exp4_physics_consistency.py
experiments/exp3_robustness/robustness_analysis.py
experiments/exp3_robustness/visualize_results.py
experiments/exp4_physics_consistency/physics_consistency.py
experiments/exp5_missing_data/missing_data_experiment.py
experiments/exp5_missing_data/run_missing_data.py
experiments/exp7_longterm/extended_ablation.py
tests/test_autoformer_model.py
tests/test_best_candidate_ablation.py
tests/test_candidate_search.py
tests/test_exp3_robustness_protocol.py
tests/test_exp3_visualize.py
tests/test_exp4_physics_metrics.py
tests/test_exp5_missing_protocol.py
tests/test_pit_radar_protocol.py
tests/test_sota_protocol_integration.py
tests/test_validate_pit_radar_dataset.py
```

Before removal, use `rg` on every module stem. If an active import remains, stop and remove that dependency with a targeted test. After external hashes match, delete repository copies using patch-based deletion. Do not archive registered `generalization_robustness`, `efficiency`, formal generators, or protected result/checkpoint paths.

- [ ] **Step 5: Verify active references and run the full suite**

```powershell
rg -n "models\.PIT|external_baselines|sota_models|run_plgaformer_candidate_search|pit_aligned_radar|--task" run.py models experiments scripts utils tests README.md
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest -q
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal validate-data --json
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal status --json
```

Expected: no active launch/import reference; full active suite passes; formal evidence remains eligible.

- [ ] **Step 6: Commit the archive boundary**

```powershell
git add -A
git commit -m "chore: archive inactive HGV versions"
```

### Task 8: Final audit and handoff

**Files:**
- Modify: `PLAN.md`
- Modify: `docs/ACTIVE_MAINLINE.md`
- Create: `docs/MAINLINE_CONSOLIDATION_REPORT.md`

- [ ] **Step 1: Record post-consolidation identities**

Record HEAD and recovery tag; dataset protocol/path/hash/test-ID hash; active flags; Main/Ablation bundle IDs and eligibility; active model registry; archive file count and manifest hash; pytest counts; and the explicit statement that no experimental value changed.

- [ ] **Step 2: Run final verification**

```powershell
git status --short --branch
git diff --check before-mainline-unification-20260822..HEAD
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' -m pytest -q
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal validate-data --json
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal status --json
& 'D:\ProgramData\anaconda3\envs\tslib\python.exe' run.py formal audit --json
```

Expected: no whitespace errors; tests pass; data, Main, Ablation, and full audit pass.

- [ ] **Step 3: Inspect removals against the archive manifest**

```powershell
git diff --name-status before-mainline-unification-20260822..HEAD
git status --short
```

Expected: every material removal appears in the archive manifest and no protected file changed.

- [ ] **Step 4: Commit the consolidation report**

```powershell
git add PLAN.md docs/ACTIVE_MAINLINE.md docs/MAINLINE_CONSOLIDATION_REPORT.md
git commit -m "docs: record the consolidated HGV mainline"
```

- [ ] **Step 5: Stop before performance changes**

Start a separate design cycle for horizon-conditioned prior fusion and validation-only candidate selection. Do not tune the model under this engineering plan.

## Plan self-review result

- Spec coverage: data, model, configuration, command, archive, evidence protection, validation, and handoff each map to a task.
- Placeholder scan: no deferred implementation field remains.
- Type consistency: contract, runtime, and archive function names match across tests and implementation steps.
- Scope: PLGAFormer performance optimization begins only after consolidation passes.
