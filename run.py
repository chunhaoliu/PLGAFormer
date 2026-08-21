#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single supported entrypoint for the HGV paper experiment mainline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HGV paper experiment runner",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("formal",),
        help="Run the frozen formal experiment and evidence pipeline.",
    )
    return parser


def _bootstrap_project_root() -> Path:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    return project_root


def main() -> int:
    _bootstrap_project_root()
    if len(sys.argv) == 1:
        build_parser().print_help()
        print("\nActive route: python run.py formal --help")
        return 0
    if sys.argv[1].lower() == "formal":
        from scripts.formal_pipeline import main as formal_main

        return int(formal_main(sys.argv[2:]))
    build_parser().parse_args()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
