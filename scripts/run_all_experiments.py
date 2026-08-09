#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run exp1->exp4 sequentially with one command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments sequentially.")
    parser.add_argument("--skip-exp1", action="store_true", help="Skip exp1_sota")
    parser.add_argument("--skip-exp2", action="store_true", help="Skip exp2_ablation")
    parser.add_argument("--skip-exp3", action="store_true", help="Skip exp3_robustness")
    parser.add_argument("--skip-exp4", action="store_true", help="Skip exp4_physics_consistency")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    args = parse_args()

    from run import run_pipeline

    return int(
        run_pipeline(
            skip_exp1=args.skip_exp1,
            skip_exp2=args.skip_exp2,
            skip_exp3=args.skip_exp3,
            skip_exp4=args.skip_exp4,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
