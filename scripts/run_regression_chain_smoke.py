#!/usr/bin/env python3
"""Read-only smoke check for the active HGV formal command surface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_formal_check(project_root: Path, command: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "run.py", "formal", command, "--json"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"formal {command} failed with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"formal {command} did not emit JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"formal {command} emitted a non-object JSON payload")
    return payload


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    _run_formal_check(project_root, "validate-data")
    status = _run_formal_check(project_root, "status")
    if status.get("paper_eligible") is not True:
        raise RuntimeError("formal status did not report paper_eligible=true")
    print("[regression] active formal command smoke completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
