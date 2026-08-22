#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only inventory audit for mainline and archive candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive_mainline_artifacts import (
    ARCHIVE_FILES,
    ARCHIVE_TREES,
    PROTECTED_PREFIXES,
    build_archive_plan,
)


def _resolve_root(root: Path | str) -> Path:
    return Path(root).expanduser().resolve()


def _relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _is_under(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _iter_project_entries(project_root: Path) -> Iterable[Path]:
    for path in sorted(project_root.rglob("*"), key=lambda value: value.relative_to(project_root).as_posix()):
        if ".git" in path.parts:
            continue
        if path.is_file():
            yield path


def _active_match(relative: Path) -> bool:
    if relative in {Path("run.py"), Path("README.md"), Path("AGENTS.md"), Path(".gitignore")}:
        return True
    if not relative.parts:
        return False
    head = relative.parts[0]
    if head in {"scripts", "tests", "models", "utils", "docs"}:
        return True
    if head == "configs" and relative != Path("configs/formal_v3.json"):
        return True
    if head == "data_generation" and relative.suffix == ".py":
        return True
    if head == "experiments" and relative.suffix == ".py":
        return not any(part in {"results", "trained_models", "checkpoints"} for part in relative.parts)
    return False


def _matches_archive_candidate(relative: Path) -> bool:
    for entry in ARCHIVE_FILES:
        candidate = Path(entry)
        if relative == candidate:
            return True
    for tree in ARCHIVE_TREES:
        candidate = Path(tree)
        if relative == candidate or _is_under(relative, candidate):
            return True
    return False


def _matches_protected(relative: Path) -> bool:
    for entry in PROTECTED_PREFIXES:
        candidate = Path(entry)
        if relative == candidate or _is_under(relative, candidate):
            return True
    return False


def build_inventory_report(project_root: Path | str, archive_root: Path | str) -> dict[str, list[str] | str]:
    project_root = _resolve_root(project_root)
    archive_root = _resolve_root(archive_root)
    plan = build_archive_plan(project_root, archive_root)
    archive_candidates = {item.source.resolve() for item in plan.items}

    active: list[str] = []
    protected: list[str] = []
    archive_candidate: list[str] = []
    missing: list[str] = [path.as_posix() for path in plan.missing]
    protected_missing: list[str] = []
    unclassified: list[str] = []

    for entry in _iter_project_entries(project_root):
        relative = Path(_relative_path(project_root, entry))
        relative_str = relative.as_posix()
        if _matches_protected(relative):
            protected.append(relative_str)
            continue
        if entry.resolve() in archive_candidates or _matches_archive_candidate(relative):
            archive_candidate.append(relative_str)
            continue
        if _active_match(relative):
            active.append(relative_str)
            continue
        unclassified.append(relative_str)

    for entry in PROTECTED_PREFIXES:
        relative = Path(entry).as_posix()
        if not (project_root / Path(entry)).exists():
            protected_missing.append(relative)

    return {
        "project_root": str(project_root),
        "archive_root": str(archive_root),
        "active": sorted(active),
        "protected": sorted(protected),
        "archive_candidate": sorted(archive_candidate),
        "missing": sorted(dict.fromkeys(missing)),
        "protected_missing": sorted(dict.fromkeys(protected_missing)),
        "unclassified": sorted(unclassified),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _format_text(report: dict[str, list[str] | str]) -> str:
    lines = [
        f"project_root: {report['project_root']}",
        f"archive_root: {report['archive_root']}",
    ]
    for key in (
        "active",
        "protected",
        "archive_candidate",
        "missing",
        "protected_missing",
        "unclassified",
    ):
        values = report[key]
        lines.append(f"{key}: {len(values)}")
        for entry in values:
            lines.append(f"  - {entry}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only classification for mainline inventory paths.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--text", action="store_true", help="Emit human-readable text instead of JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = build_inventory_report(args.project_root, args.archive_root)
        if args.text:
            print(_format_text(report))
        else:
            print(json.dumps(_json_ready(report), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - defensive
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
