#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified launcher for exp1_sota."""

from pathlib import Path
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.console import ensure_utf8_console
    ensure_utf8_console()

    from experiments.overall_prediction.main_results import main as run_main

    run_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
