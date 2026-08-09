#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal end-to-end regression run for exp1 -> exp4."""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _configure_global_speed_mode():
    from models import HGVConfig

    HGVConfig.TRAIN_MODE = "speed"
    HGVConfig.TRAIN_PARAMS["epochs"] = 1
    HGVConfig.TRAIN_PARAMS["num_runs"] = 1
    HGVConfig.TRAIN_PARAMS["random_seeds"] = [42]
    HGVConfig.TRAIN_PARAMS["data_subset_ratio"] = 0.02
    HGVConfig.TRAIN_PARAMS["dataloader_workers"] = 0
    HGVConfig.TRAIN_PARAMS["pin_memory"] = False
    HGVConfig.TRAIN_PARAMS["persistent_workers"] = False
    HGVConfig.TRAIN_PARAMS["verbose"] = False


def _run_exp1(project_root: Path):
    from experiments.overall_prediction import main_results as exp1

    exp1.NUM_RUNS = 1
    exp1.RANDOM_SEEDS = [42]
    exp1.PREDICTION_HORIZONS = [32]
    exp1.SOTA_DATA_SUBSET_RATIO = 0.02

    keep = ["Transformer (baseline)", "PLGAFormer (proposed)", "Informer"]
    exp1.COMPARISON_MODELS = OrderedDict(
        (k, v) for k, v in exp1.COMPARISON_MODELS.items() if k in keep
    )
    model_dir = project_root / "experiments" / "exp1_sota" / "trained_models"
    required = [
        model_dir / "best_transformer_sota.pth",
        model_dir / "best_plgaformer_sota.pth",
        model_dir / "best_informer_sota.pth",
    ]
    if all(p.exists() for p in required):
        print("[regression] skip exp1_sota (required smoke models already exist)")
        return
    print("[regression] running exp1_sota (smoke config)")
    exp1.main()


def _run_exp2():
    from experiments.mechanism_analysis import ablation_study as exp2

    smoke_results = Path(exp2.PROJECT_ROOT) / "experiments" / "exp2_ablation" / "results" / "smoke"
    smoke_models = Path(exp2.PROJECT_ROOT) / "experiments" / "exp2_ablation" / "trained_models" / "smoke"
    smoke_results.mkdir(parents=True, exist_ok=True)
    smoke_models.mkdir(parents=True, exist_ok=True)
    exp2.get_experiment_dirs = lambda _root, _name: (smoke_results, smoke_models)

    exp2.NUM_RUNS = 1
    exp2.RANDOM_SEEDS = [42]
    exp2.BASE_RANDOM_SEEDS = [42]
    exp2.PREDICTION_HORIZONS = [64]
    exp2.TRAIN_CONFIG["epochs"] = 1
    exp2.TRAIN_CONFIG["batch_size"] = min(32, exp2.TRAIN_CONFIG.get("batch_size", 32))
    exp2.TRAIN_CONFIG["num_runs"] = 1
    exp2.TRAIN_CONFIG["random_seeds"] = [42]

    phase_name, phase_cfg = next(iter(exp2.ABLATION_PHASES.items()))
    model_name, model_cfg = next(iter(phase_cfg["models"].items()))
    exp2.ABLATION_PHASES = OrderedDict(
        [
            (
                phase_name,
                {
                    "num_runs": 1,
                    "random_seeds": [42],
                    "models": OrderedDict([(model_name, model_cfg)]),
                },
            )
        ]
    )
    print("[regression] running exp2_ablation (smoke config)")
    exp2.main()


def _run_exp3(project_root: Path):
    exp3_path = project_root / "experiments" / "exp3_robustness" / "robustness_analysis.py"
    exp3 = _load_module_from_path("exp3_smoke_module", exp3_path)
    exp3.RESULTS_DIR = project_root / "experiments" / "exp3_robustness" / "results" / "smoke"
    exp3.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp3.NOISE_LEVELS = exp3.NOISE_LEVELS[:1]
    exp3.MISSING_RATES = exp3.MISSING_RATES[:1]
    exp3.INPUT_LENGTHS = exp3.INPUT_LENGTHS[:1]
    exp3.PREDICTION_LENGTH = min(int(exp3.PREDICTION_LENGTH), 32)
    print("[regression] running exp3_robustness (smoke config)")
    exp3.main()


def _run_exp4(project_root: Path):
    exp4_path = project_root / "experiments" / "exp4_physics_consistency" / "physics_consistency.py"
    exp4 = _load_module_from_path("exp4_smoke_module", exp4_path)
    exp4.RESULTS_DIR = project_root / "experiments" / "exp4_physics_consistency" / "results" / "smoke"
    exp4.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp4.COPY_FIGURES_TO_LATEX = False
    exp4.PREDICTION_LENGTH = min(int(exp4.PREDICTION_LENGTH), 32)
    print("[regression] running exp4_physics_consistency (smoke config)")
    exp4.main()


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.console import ensure_utf8_console

    ensure_utf8_console()

    _configure_global_speed_mode()
    _run_exp1(project_root)
    _run_exp2()
    _run_exp3(project_root)
    _run_exp4(project_root)
    print("[regression] exp1->exp4 smoke chain completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
