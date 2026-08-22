from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _as_relative_set(entries: list[dict[str, object]]) -> set[Path]:
    return {Path(item["path"]) for item in entries}


def test_build_archive_plan_excludes_protected_paths_and_reports_missing(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    protected = project_root / "experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final/result.json"
    archived = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    missing = Path("data_generation/data/processed/pit_aligned_radar_dataset.npz")

    protected.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    protected.write_text("formal", encoding="utf-8")
    archived.write_bytes(b"legacy")

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (
            Path("experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final/result.json"),
            Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),
            missing,
        ),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    plan = archive.build_archive_plan(project_root, archive_root)

    assert protected.resolve() not in {item.source for item in plan}
    assert archived.resolve() in {item.source for item in plan}
    assert missing in plan.missing
    assert Path("experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final/result.json") in plan.protected


def test_archive_plan_rejects_escape_overlap_and_duplicate_entries(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    project_root.mkdir()
    archive_root.mkdir()

    monkeypatch.setattr(archive, "ARCHIVE_FILES", (Path("../escape.txt"),))
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())
    with pytest.raises(ValueError, match="escape"):
        archive.build_archive_plan(project_root, archive_root)

    allowed = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"payload")

    monkeypatch.setattr(archive, "ARCHIVE_FILES", (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),))
    with pytest.raises(ValueError, match="overlap"):
        archive.build_archive_plan(project_root, project_root / "archive")

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (
            Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),
            Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        archive.build_archive_plan(project_root, archive_root)


def test_archive_apply_preserves_bytes_and_hash(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_multiregime_dataset_v2.npz"
    source.parent.mkdir(parents=True)
    payload = b"historical"
    source.write_bytes(payload)

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (Path("data_generation/data/processed/hgv_multiregime_dataset_v2.npz"),),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    plan = archive.build_archive_plan(project_root, archive_root)
    manifest = archive.apply_archive_plan(plan)

    destination = archive_root / source.relative_to(project_root)
    assert not source.exists()
    assert destination.read_bytes() == payload
    assert manifest["files"][0]["sha256_before"] == hashlib.sha256(payload).hexdigest()
    assert manifest["files"][0]["sha256_after"] == hashlib.sha256(payload).hexdigest()
    assert Path(manifest["files"][0]["destination"]) == source.relative_to(project_root)
    assert Path(manifest["files"][0]["destination_absolute"]) == destination
    assert manifest["files"][0]["source"] == str(source)


def test_archive_apply_rejects_destination_mismatch(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_multiregime_dataset_v2.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (Path("data_generation/data/processed/hgv_multiregime_dataset_v2.npz"),),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    plan = archive.build_archive_plan(project_root, archive_root)
    fake_item = SimpleNamespace(
        source=plan.items[0].source,
        destination=archive_root / "elsewhere" / "wrong.npz",
        sha256_before=plan.items[0].sha256_before,
        relative_path=plan.items[0].relative_path,
    )
    fake_plan = SimpleNamespace(
        project_root=plan.project_root,
        archive_root=plan.archive_root,
        items=(fake_item,),
        missing=(),
        protected=(),
    )

    with pytest.raises(ValueError, match="destination"):
        archive.apply_archive_plan(fake_plan)


def test_inventory_is_read_only_and_classifies_paths(tmp_path, monkeypatch):
    from scripts import audit_mainline_inventory as audit
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"

    active = project_root / "scripts/tool.py"
    protected = project_root / "configs/formal_v3.json"
    candidate = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    missing = Path("data_generation/data/processed/pit_aligned_radar_dataset.npz")
    unclassified = project_root / "notes/loose.txt"

    active.parent.mkdir(parents=True)
    protected.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    unclassified.parent.mkdir(parents=True)
    active.write_text("print('active')", encoding="utf-8")
    protected.write_text("frozen", encoding="utf-8")
    candidate.write_bytes(b"candidate")
    unclassified.write_text("misc", encoding="utf-8")

    monkeypatch.setattr(archive, "ARCHIVE_FILES", (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"), missing))
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    report = audit.build_inventory_report(project_root, archive_root)
    assert active.exists() and protected.exists() and candidate.exists() and unclassified.exists()
    assert report["active"] == [active.relative_to(project_root).as_posix()]
    assert report["protected"] == [protected.relative_to(project_root).as_posix()]
    assert report["archive_candidate"] == [candidate.relative_to(project_root).as_posix()]
    assert report["missing"] == [missing.as_posix()]
    assert report["unclassified"] == [unclassified.relative_to(project_root).as_posix()]


def test_archive_apply_rolls_back_on_late_failure(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    first = project_root / "data_generation/data/processed/hgv_multiregime_dataset_v2.npz"
    second = project_root / "data_generation/data/processed/hgv_multiregime_pilot_v2.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first_payload = b"first-payload"
    second_payload = b"second-payload"
    first.write_bytes(first_payload)
    second.write_bytes(second_payload)

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (
            Path("data_generation/data/processed/hgv_multiregime_dataset_v2.npz"),
            Path("data_generation/data/processed/hgv_multiregime_pilot_v2.npz"),
        ),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    original_copy2 = archive.shutil.copy2
    call_count = {"value": 0}

    def flaky_copy2(src, dst, *args, **kwargs):
        call_count["value"] += 1
        if Path(src).resolve() == second.resolve() and call_count["value"] == 2:
            raise RuntimeError("late copy failure")
        return original_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(archive.shutil, "copy2", flaky_copy2)

    with pytest.raises(RuntimeError, match="late copy failure"):
        archive.apply_archive_plan(archive.build_archive_plan(project_root, archive_root))

    assert first.read_bytes() == first_payload
    assert not (archive_root / first.relative_to(project_root)).exists()
    assert second.read_bytes() == second_payload


def test_archive_apply_rolls_back_when_manifest_write_fails(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    def fail_manifest(*args, **kwargs):
        raise OSError("manifest write failure")

    monkeypatch.setattr(archive, "_write_json_atomic", fail_manifest)

    with pytest.raises(OSError, match="manifest write failure"):
        archive.apply_archive_plan(archive.build_archive_plan(project_root, archive_root))

    assert source.read_bytes() == b"payload"
    assert not (archive_root / source.relative_to(project_root)).exists()
    assert not (archive_root / "archive_manifest.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute behavior")
def test_archive_apply_handles_readonly_source(tmp_path, monkeypatch):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"readonly-payload")
    os.chmod(source, stat.S_IREAD)

    monkeypatch.setattr(
        archive,
        "ARCHIVE_FILES",
        (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),),
    )
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    manifest = archive.apply_archive_plan(archive.build_archive_plan(project_root, archive_root))
    destination = archive_root / source.relative_to(project_root)
    assert not source.exists()
    assert destination.read_bytes() == b"readonly-payload"
    assert manifest["files"][0]["sha256_before"] == manifest["files"][0]["sha256_after"]
    os.chmod(destination, stat.S_IWRITE | stat.S_IREAD)


def test_archive_cli_emits_dry_run_json(tmp_path, monkeypatch, capsys):
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")

    monkeypatch.setattr(archive, "ARCHIVE_FILES", (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),))
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    exit_code = archive.main(
        [
            "--project-root",
            str(project_root),
            "--archive-root",
            str(archive_root),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["apply"] is False
    assert payload["project_root"] == str(project_root)
    assert payload["archive_root"] == str(archive_root)
    assert Path(payload["plan"]["items"][0]["source"]) == source
    assert all(not Path(item["source"]).is_relative_to(project_root / "experiments/exp1_sota") for item in payload["plan"]["items"])


def test_inventory_cli_emits_json_and_never_moves(tmp_path, monkeypatch, capsys):
    from scripts import audit_mainline_inventory as audit
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")

    monkeypatch.setattr(archive, "ARCHIVE_FILES", (Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),))
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    exit_code = audit.main(
        [
            "--project-root",
            str(project_root),
            "--archive-root",
            str(archive_root),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["archive_candidate"] == [Path("data_generation/data/processed/hgv_trajectory_dataset.npz").as_posix()]
    assert source.exists()


def test_inventory_reports_missing_protected_paths_separately(tmp_path, monkeypatch):
    from scripts import audit_mainline_inventory as audit
    from scripts import archive_mainline_artifacts as archive

    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    project_root.mkdir()
    monkeypatch.setattr(archive, "ARCHIVE_FILES", ())
    monkeypatch.setattr(archive, "ARCHIVE_TREES", ())

    report = audit.build_inventory_report(project_root, archive_root)

    assert "protected_missing" in report
    assert "configs/formal_v3.json" in report["protected_missing"]


def test_inventory_script_cli_bootstraps_project_import(tmp_path):
    project_root = tmp_path / "project"
    archive_root = tmp_path / "archive"
    source = project_root / "data_generation/data/processed/hgv_trajectory_dataset.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"payload")
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_mainline_inventory.py",
            "--project-root",
            str(project_root),
            "--archive-root",
            str(archive_root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["archive_candidate"] == [
        "data_generation/data/processed/hgv_trajectory_dataset.npz"
    ]
    assert source.exists()
