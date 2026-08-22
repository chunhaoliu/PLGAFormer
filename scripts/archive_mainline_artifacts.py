#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run-first archival tooling for non-mainline generated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_FILES = (
    Path("data_generation/data/processed/hgv_multiregime_dataset_v2.npz"),
    Path("data_generation/data/processed/hgv_multiregime_pilot_v2.npz"),
    Path("data_generation/data/processed/hgv_multiregime_pilot_v2_1.npz"),
    Path("data_generation/data/processed/hgv_trajectory_dataset.npz"),
    Path("data_generation/data/processed/pit_aligned_radar_dataset.npz"),
    Path("data_generation/data/processed/pit_aligned_radar_pilot.npz"),
)

ARCHIVE_TREES = (
    Path("data_generation/data/processed/derived_window_views"),
    Path("tmp/AF-CILN"),
    Path("tmp/robustness_protocol_view"),
)

PROTECTED_PREFIXES = (
    Path("configs/formal_v3.json"),
    Path("data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz"),
    Path("experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final"),
    Path("experiments/exp1_sota/trained_models/formal_v3/hgv_multiregime_state_v2_1/final"),
    Path("experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final"),
    Path("experiments/exp2_ablation/trained_models/formal_v3/hgv_multiregime_state_v2_1/final"),
)


@dataclass(frozen=True)
class ArchiveItem:
    source: Path
    destination: Path
    sha256_before: str

    @property
    def relative_path(self) -> Path:
        return self.destination


@dataclass(frozen=True)
class ArchivePlan:
    project_root: Path
    archive_root: Path
    items: tuple[ArchiveItem, ...]
    missing: tuple[Path, ...]
    protected: tuple[Path, ...]

    def __iter__(self) -> Iterator[ArchiveItem]:
        return iter(self.items)


def _resolve_root(root: Path | str) -> Path:
    resolved = Path(root).expanduser().resolve()
    if not resolved.exists():
        resolved = Path(root).expanduser().resolve()
    return resolved


def _relative_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        return path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"path escape detected: {path} is outside {root}") from exc


def _is_under(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_under(left, right) or _is_under(right, left)


def _resolve_entry(project_root: Path, entry: Path) -> Path:
    candidate = entry if entry.is_absolute() else project_root / entry
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"path escape detected for {entry!s}") from exc
    return resolved


def _normalize_relative(path: Path) -> Path:
    return Path(*path.parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        if os.name != "nt":
            raise
        os.chmod(path, path.stat().st_mode | stat.S_IWRITE)
        path.unlink(missing_ok=True)


def _iter_tree_files(tree_root: Path) -> Iterable[Path]:
    return sorted((path for path in tree_root.rglob("*") if path.is_file()), key=lambda value: value.relative_to(tree_root).as_posix())


def _protected_match(project_root: Path, candidate: Path) -> Path | None:
    relative = _relative_path(project_root, candidate)
    for prefix in PROTECTED_PREFIXES:
        prefix = Path(prefix)
        if relative == prefix or _is_under(relative, prefix):
            return prefix
    return None


def _record_item(
    *,
    project_root: Path,
    source: Path,
    seen_sources: set[Path],
    seen_destinations: set[Path],
    items: list[ArchiveItem],
) -> None:
    source = source.resolve()
    try:
        relative = source.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"source escape detected: {source}") from exc
    if source in seen_sources:
        raise ValueError(f"duplicate archive entry detected for {relative.as_posix()}")
    destination = _normalize_relative(relative)
    if destination in seen_destinations:
        raise ValueError(f"duplicate destination detected for {destination.as_posix()}")
    seen_sources.add(source)
    seen_destinations.add(destination)
    items.append(
        ArchiveItem(
            source=source,
            destination=destination,
            sha256_before=_sha256_file(source),
        )
    )


def build_archive_plan(project_root: Path | str, archive_root: Path | str) -> ArchivePlan:
    project_root = Path(project_root).expanduser().resolve()
    archive_root = Path(archive_root).expanduser().resolve()
    if _paths_overlap(project_root, archive_root):
        raise ValueError("project root and archive root must not overlap")

    items: list[ArchiveItem] = []
    missing: list[Path] = []
    protected: list[Path] = []
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()

    def handle_source(source: Path) -> None:
        protection = _protected_match(project_root, source)
        relative = _relative_path(project_root, source)
        if protection is not None:
            if relative not in protected:
                protected.append(relative)
            return
        if not source.exists():
            if relative not in missing:
                missing.append(relative)
            return
        if source.is_dir():
            raise ValueError(f"expected file but found directory: {relative.as_posix()}")
        _record_item(
            project_root=project_root,
            source=source,
            seen_sources=seen_sources,
            seen_destinations=seen_destinations,
            items=items,
        )

    for entry in ARCHIVE_FILES:
        handle_source(_resolve_entry(project_root, Path(entry)))

    for entry in ARCHIVE_TREES:
        tree_root = _resolve_entry(project_root, Path(entry))
        relative = _relative_path(project_root, tree_root)
        protection = _protected_match(project_root, tree_root)
        if protection is not None:
            if relative not in protected:
                protected.append(relative)
            continue
        if not tree_root.exists():
            if relative not in missing:
                missing.append(relative)
            continue
        if tree_root.is_file():
            raise ValueError(f"expected directory but found file: {relative.as_posix()}")
        for source in _iter_tree_files(tree_root):
            protection = _protected_match(project_root, source)
            rel = _relative_path(project_root, source)
            if protection is not None:
                if rel not in protected:
                    protected.append(rel)
                continue
            if not source.exists():
                if rel not in missing:
                    missing.append(rel)
                continue
            _record_item(
                project_root=project_root,
                    source=source,
                seen_sources=seen_sources,
                seen_destinations=seen_destinations,
                items=items,
            )

    return ArchivePlan(
        project_root=project_root,
        archive_root=archive_root,
        items=tuple(items),
        missing=tuple(missing),
        protected=tuple(protected),
    )


def _validate_plan(plan: Any) -> tuple[Path, Path, tuple[ArchiveItem, ...]]:
    try:
        project_root = Path(plan.project_root).expanduser().resolve()
        archive_root = Path(plan.archive_root).expanduser().resolve()
        items = tuple(plan.items)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError("plan must provide project_root, archive_root, and items") from exc
    if _paths_overlap(project_root, archive_root):
        raise ValueError("project root and archive root must not overlap")
    return project_root, archive_root, items


def _materialize_item(project_root: Path, archive_root: Path, item: ArchiveItem) -> dict[str, Any]:
    if not isinstance(item.destination, Path):
        raise TypeError("archive item destination must be a Path")
    source = Path(item.source).expanduser().resolve()
    try:
        relative = source.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"source escape detected: {source}") from exc
    expected_destination = _normalize_relative(relative)
    if item.destination != expected_destination:
        raise ValueError(
            "destination mismatch for "
            f"{relative.as_posix()}: expected {expected_destination.as_posix()}, got {item.destination.as_posix()}"
        )
    destination = (archive_root / item.destination).resolve()
    try:
        destination.relative_to(archive_root)
    except ValueError as exc:
        raise ValueError(f"destination escape detected: {destination}") from exc
    if _protected_match(project_root, source) is not None:
        raise ValueError(f"protected source cannot be archived: {relative.as_posix()}")
    if not source.exists():
        raise FileNotFoundError(f"missing source: {relative.as_posix()}")
    if not source.is_file():
        raise ValueError(f"expected file source: {relative.as_posix()}")
    return {
        "source": source,
        "destination": destination,
        "destination_absolute": destination,
        "relative_path": relative,
        "sha256_before": item.sha256_before,
    }


def _restore_archived_item(project_root: Path, materialized: dict[str, Any]) -> None:
    source = Path(materialized["source"])
    destination = Path(materialized["destination"])
    relative_path = Path(materialized["relative_path"])
    sha256_before = str(materialized["sha256_before"])
    if not destination.exists():
        return
    if source.exists():
        if _sha256_file(source) != sha256_before:
            raise RuntimeError(f"rollback source hash mismatch: {relative_path.as_posix()}")
        _unlink_file(destination)
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination, source)
    if _sha256_file(source) != sha256_before:
        raise RuntimeError(f"rollback restore hash mismatch: {relative_path.as_posix()}")
    _unlink_file(destination)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent, suffix=".tmp") as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _rollback_completed_items(
    project_root: Path,
    completed_items: Sequence[dict[str, Any]],
) -> None:
    rollback_error: Exception | None = None
    for completed in reversed(completed_items):
        try:
            _restore_archived_item(project_root, completed)
        except Exception as exc:  # pragma: no cover - defensive rollback guard
            if rollback_error is None:
                rollback_error = exc
    if rollback_error is not None:
        raise rollback_error


def apply_archive_plan(plan: Any) -> dict[str, Any]:
    project_root, archive_root, items = _validate_plan(plan)
    archive_root.mkdir(parents=True, exist_ok=True)
    prepared_items = [
        _materialize_item(project_root, archive_root, item)
        for item in items
    ]
    for materialized in prepared_items:
        source = materialized["source"]
        sha256_before = materialized["sha256_before"]
        relative_path = materialized["relative_path"]
        if _sha256_file(source) != sha256_before:
            raise ValueError(f"source hash changed before archival: {relative_path.as_posix()}")

    manifest_items: list[dict[str, Any]] = []
    completed_items: list[dict[str, Any]] = []

    try:
        for materialized in prepared_items:
            source = materialized["source"]
            destination = materialized["destination"]
            relative_path = materialized["relative_path"]
            sha256_before = materialized["sha256_before"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp_destination = destination.with_name(f"{destination.name}.tmp-{os.getpid()}-{len(manifest_items)}")
            copied = False
            moved = False
            try:
                shutil.copy2(source, tmp_destination)
                copied = True
                if _sha256_file(tmp_destination) != sha256_before:
                    raise ValueError(f"temporary copy hash mismatch: {relative_path.as_posix()}")
                tmp_destination.replace(destination)
                moved = True
                sha256_after = _sha256_file(destination)
                if sha256_after != sha256_before:
                    raise ValueError(f"destination hash mismatch: {relative_path.as_posix()}")
                try:
                    _unlink_file(source)
                except Exception:
                    if destination.exists():
                        _unlink_file(destination)
                    raise
                manifest_items.append(
                    {
                        "source": str(source),
                        "destination": str(relative_path),
                        "source_absolute": str(source),
                        "destination_absolute": str(destination),
                        "relative_path": str(relative_path),
                        "sha256_before": sha256_before,
                        "sha256_after": sha256_after,
                        "size_bytes": destination.stat().st_size,
                    }
                )
                completed_items.append(materialized)
            except Exception:
                if tmp_destination.exists():
                    _unlink_file(tmp_destination)
                if moved and destination.exists() and source.exists():
                    _unlink_file(destination)
                if copied and not moved and tmp_destination.exists():
                    _unlink_file(tmp_destination)
                raise
    except Exception:
        _rollback_completed_items(project_root, completed_items)
        raise

    manifest = {
        "archive_root": str(archive_root),
        "project_root": str(project_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": manifest_items,
        "item_count": len(manifest_items),
    }
    try:
        _write_json_atomic(archive_root / "archive_manifest.json", manifest)
    except Exception:
        _rollback_completed_items(project_root, completed_items)
        raise
    return manifest


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ArchiveItem):
        return {
            "source": str(value.source),
            "destination": str(value.destination),
            "sha256_before": value.sha256_before,
        }
    if isinstance(value, ArchivePlan):
        return {
            "project_root": str(value.project_root),
            "archive_root": str(value.archive_root),
            "items": [_json_ready(item) for item in value.items],
            "missing": [str(path) for path in value.missing],
            "protected": [str(path) for path in value.protected],
        }
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _plan_status(plan: ArchivePlan) -> dict[str, Any]:
    return {
        "item_count": len(plan.items),
        "missing_count": len(plan.missing),
        "protected_count": len(plan.protected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive non-mainline generated artifacts safely.")
    parser.add_argument("--project-root", type=Path, required=True, help="Repository root that owns the sources.")
    parser.add_argument("--archive-root", type=Path, required=True, help="External archive root that receives the files.")
    parser.add_argument("--apply", action="store_true", help="Move files instead of performing a dry-run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        plan = build_archive_plan(args.project_root, args.archive_root)
        if args.apply:
            manifest = apply_archive_plan(plan)
            payload = {
                "apply": True,
                "project_root": str(plan.project_root),
                "archive_root": str(plan.archive_root),
                "plan": _json_ready(plan),
                "manifest": _json_ready(manifest),
            }
        else:
            payload = {
                "apply": False,
                "project_root": str(plan.project_root),
                "archive_root": str(plan.archive_root),
                "plan": _json_ready(plan),
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - exercised via CLI smoke only
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
