#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry for exp1_sota."""

from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.exp1_sota.SOTA_comparison import main as run_main

    print("Starting exp1_sota...")
    run_main()


if __name__ == "__main__":
    main()

