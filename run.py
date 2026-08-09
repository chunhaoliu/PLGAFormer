#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entrypoint aligned with minimalist time-series project style."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified HGV experiment runner")
    parser.add_argument(
        "--task",
        default=None,
        choices=[
            "all",
            "enhancements",
            "exp1",
            "exp2",
            "exp3",
            "exp4",
            "exp5",
            "exp6",
            "exp7",
            "smoke",
            "data",
            "validate-data",
        ],
        help=(
            "Explicit legacy task to run. Paper-facing work uses "
            "python run.py formal {main,mechanism,robustness,efficiency}."
        ),
    )
    parser.add_argument("--skip-exp1", action="store_true", help="Only used when --task all")
    parser.add_argument("--skip-exp2", action="store_true", help="Only used when --task all")
    parser.add_argument("--skip-exp3", action="store_true", help="Only used when --task all")
    parser.add_argument("--skip-exp4", action="store_true", help="Only used when --task all")
    parser.add_argument("--skip-exp5", action="store_true", help="Only used for enhancement tasks")
    parser.add_argument("--skip-exp6", action="store_true", help="Only used for enhancement tasks")
    parser.add_argument("--skip-exp7", action="store_true", help="Only used for enhancement tasks")
    parser.add_argument("--include-enhancements", action="store_true", help="Include Exp5-Exp7 when --task all")
    parser.add_argument("--quick", action="store_true", help="Use quick mode for Exp5-Exp7")
    parser.add_argument("--dry-run", action="store_true", help="Only used when --task all")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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


def run_exp1_sota() -> None:
    from experiments.overall_prediction.main_results import main as run_main

    run_main()


def run_exp2_ablation() -> None:
    from experiments.mechanism_analysis.ablation_study import main as run_main

    run_main()


def run_exp3_robustness() -> None:
    from experiments.exp3_robustness.robustness_analysis import main as run_main

    run_main()


def run_exp4_physics() -> None:
    from experiments.exp4_physics_consistency.physics_consistency import main as run_main

    run_main()


def run_exp5_missing_data(child_args: list[str] | None = None) -> None:
    from experiments.exp5_missing_data.missing_data_experiment import main as run_main

    run_main(child_args or [])


def run_exp6_efficiency(child_args: list[str] | None = None) -> None:
    from experiments.exp6_efficiency.efficiency_experiment import main as run_main

    run_main(child_args or [])


def run_exp7_longterm(child_args: list[str] | None = None) -> None:
    from experiments.exp7_longterm.extended_ablation import main as run_main

    run_main(child_args or [])


def run_smoke() -> int:
    from scripts.run_regression_chain_smoke import main as run_main

    return int(run_main())


def run_pipeline(
    *,
    skip_exp1: bool = False,
    skip_exp2: bool = False,
    skip_exp3: bool = False,
    skip_exp4: bool = False,
    skip_exp5: bool = False,
    skip_exp6: bool = False,
    skip_exp7: bool = False,
    include_enhancements: bool = False,
    quick: bool = False,
    dry_run: bool = False,
) -> int:
    child_args = ["--quick"] if quick else []
    steps: list[tuple[str, bool, Callable[[], None]]] = [
        ("exp1_sota", skip_exp1, run_exp1_sota),
        ("exp2_ablation", skip_exp2, run_exp2_ablation),
        ("exp3_robustness", skip_exp3, run_exp3_robustness),
        ("exp4_physics_consistency", skip_exp4, run_exp4_physics),
    ]
    if include_enhancements:
        steps.extend(
            [
                ("exp5_missing_data", skip_exp5, lambda: run_exp5_missing_data(child_args)),
                ("exp6_efficiency", skip_exp6, lambda: run_exp6_efficiency(child_args)),
                ("exp7_longterm", skip_exp7, lambda: run_exp7_longterm(child_args)),
            ]
        )

    for step_name, skip, run_fn in steps:
        if skip:
            print(f"[pipeline] skip {step_name}")
            continue
        print(f"[pipeline] run {step_name}")
        if not dry_run:
            run_fn()

    print("[pipeline] all requested experiments finished")
    return 0


def main() -> int:
    _bootstrap_project_root()
    if len(sys.argv) == 1:
        build_parser().print_help()
        print("\nPaper-facing route: python run.py formal --help")
        return 0
    if len(sys.argv) > 1 and sys.argv[1].lower() == "formal":
        from scripts.formal_pipeline import main as formal_main

        return int(formal_main(sys.argv[2:]))
    args = parse_args()
    if args.task is None:
        build_parser().print_help()
        print("\nPaper-facing route: python run.py formal --help")
        return 0
    child_args = ["--quick"] if args.quick else []

    if args.task == "exp1":
        run_exp1_sota()
        return 0
    if args.task == "exp2":
        run_exp2_ablation()
        return 0
    if args.task == "exp3":
        run_exp3_robustness()
        return 0
    if args.task == "exp4":
        run_exp4_physics()
        return 0
    if args.task == "exp5":
        run_exp5_missing_data(child_args)
        return 0
    if args.task == "exp6":
        run_exp6_efficiency(child_args)
        return 0
    if args.task == "exp7":
        run_exp7_longterm(child_args)
        return 0
    if args.task == "smoke":
        return int(run_smoke())
    if args.task == "data":
        from scripts.generate_multiregime_dataset import main as run_main

        run_main(["--formal"])
        return 0
    if args.task == "validate-data":
        from scripts.validate_multiregime_dataset import main as run_multiregime_audit
        from data_provider.validation import validate_hgv_dataset

        report = validate_hgv_dataset(require_trajectory_level=True)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["passed"]:
            return 1
        return int(
            run_multiregime_audit(
                [
                    "--dataset",
                    str(
                        _bootstrap_project_root()
                        / "data_generation"
                        / "data"
                        / "processed"
                        / "hgv_multiregime_dataset_v2_1.npz"
                    ),
                    "--require-formal-count",
                ]
            )
        )

    if args.task == "enhancements":
        return int(
            run_pipeline(
                skip_exp1=True,
                skip_exp2=True,
                skip_exp3=True,
                skip_exp4=True,
                skip_exp5=args.skip_exp5,
                skip_exp6=args.skip_exp6,
                skip_exp7=args.skip_exp7,
                include_enhancements=True,
                quick=args.quick,
                dry_run=args.dry_run,
            )
        )

    return int(
        run_pipeline(
            skip_exp1=args.skip_exp1,
            skip_exp2=args.skip_exp2,
            skip_exp3=args.skip_exp3,
            skip_exp4=args.skip_exp4,
            skip_exp5=args.skip_exp5,
            skip_exp6=args.skip_exp6,
            skip_exp7=args.skip_exp7,
            include_enhancements=args.include_enhancements,
            quick=args.quick,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
