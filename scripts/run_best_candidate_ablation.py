#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run A/B/C ablation with the best candidate-search hyperparameters."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_ablation_model_configs(candidate: dict[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    """Build baseline/full/w-o A/B/C configs from one candidate row."""
    alpha = float(candidate.get("alpha", candidate.get("physics_loss_weight", 0.0)))
    dropout = float(candidate.get("dropout", 0.15))
    warmup_epochs = int(candidate.get("warmup_epochs", 2))

    def full_cfg(description: str, innovations: dict[str, bool]) -> dict[str, Any]:
        return {
            "model_type": "plgaformer",
            "description": description,
            "physics_loss_weight": alpha,
            "physics_alpha": alpha,
            "warmup_epochs": warmup_epochs,
            "dropout": dropout,
            "innovations": innovations,
        }

    return OrderedDict(
        [
            (
                "Transformer (baseline)",
                {
                    "model_type": "baseline",
                    "description": "Standard Transformer baseline",
                    "physics_loss_weight": 0.0,
                    "innovations": None,
                },
            ),
            (
                "PLGAFormer (A+B+C)",
                full_cfg(
                    "Full PLGAFormer using best candidate-search hyperparameters",
                    {
                        "use_sparse_attention": True,
                        "use_physics_corrector": True,
                        "use_multi_head_output": True,
                    },
                ),
            ),
            (
                "PLGAFormer w/o A",
                full_cfg(
                    "Remove physics-aware attention",
                    {
                        "use_sparse_attention": False,
                        "use_physics_corrector": True,
                        "use_multi_head_output": True,
                    },
                ),
            ),
            (
                "PLGAFormer w/o B",
                {
                    **full_cfg(
                        "Remove gated physics corrector and physics consistency loss",
                        {
                            "use_sparse_attention": True,
                            "use_physics_corrector": False,
                            "use_multi_head_output": True,
                        },
                    ),
                    "physics_loss_weight": 0.0,
                    "physics_alpha": 0.0,
                },
            ),
            (
                "PLGAFormer w/o C",
                full_cfg(
                    "Remove multi-head residual trajectory decoder",
                    {
                        "use_sparse_attention": True,
                        "use_physics_corrector": True,
                        "use_multi_head_output": False,
                    },
                ),
            ),
        ]
    )


def load_best_candidate(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    candidate = payload.get("best_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"No best_candidate found in {path}")
    return candidate


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    return seeds or [42]


def parse_prediction_horizons(value: str, fallback: int = 64) -> list[int]:
    """Parse a comma-separated horizon list into sorted unique positive ints."""
    horizons = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon <= 0:
            raise ValueError(f"Prediction horizon must be positive: {horizon}")
        horizons.append(horizon)
    if not horizons:
        horizons = [int(fallback)]
    return sorted(set(horizons))


def _configure_exp2(exp2: Any, args: argparse.Namespace, candidate: dict[str, Any]) -> list[int]:
    seeds = _parse_seeds(args.seeds)
    lr = float(candidate.get("learning_rate", args.learning_rate))
    clip = float(candidate.get("gradient_clip_norm", args.gradient_clip_norm))
    horizons = parse_prediction_horizons(args.prediction_horizons, fallback=int(args.prediction_length))
    max_horizon = max(horizons)

    exp2.BASE_RANDOM_SEEDS = seeds
    exp2.NUM_RUNS = int(args.num_runs)
    exp2.RANDOM_SEEDS = seeds[: exp2.NUM_RUNS]
    exp2.TRAIN_CONFIG["epochs"] = int(args.epochs)
    exp2.TRAIN_CONFIG["batch_size"] = int(args.batch_size)
    exp2.TRAIN_CONFIG["lr"] = lr
    exp2.TRAIN_CONFIG["weight_decay"] = float(args.weight_decay)
    exp2.TRAIN_CONFIG["warmup_epochs"] = int(candidate.get("warmup_epochs", args.warmup_epochs))
    exp2.TRAIN_CONFIG["early_stopping_patience"] = int(args.patience)
    exp2.TRAIN_CONFIG["gradient_clip_norm"] = clip
    exp2.TRAIN_CONFIG["use_mixed_precision"] = bool(args.amp)
    exp2.TRAIN_CONFIG["dataloader_workers"] = int(args.workers)
    exp2.TRAIN_CONFIG["pin_memory"] = bool(args.workers > 0)
    exp2.ABLATION_SUBSET_RATIO = float(args.subset_ratio)
    exp2.ABLATION_PRED_LEN = max_horizon
    exp2.PREDICTION_HORIZONS = horizons
    exp2.TRAIN_SUPERVISION_PROTOCOL = str(args.train_supervision_protocol)
    exp2.EVAL_PROTOCOL = str(args.eval_protocol)
    exp2.EVAL_AR_SEED_MODE = str(args.eval_ar_seed_mode)
    exp2.train_config["label_len"] = int(args.label_len)
    return exp2.RANDOM_SEEDS


def _write_metadata(
    output_dir: Path,
    timestamp: str,
    candidate: dict[str, Any],
    run_config: dict[str, Any],
    saved_paths: dict[str, str],
) -> dict[str, Path]:
    payload = {
        "candidate": candidate,
        "run_config": run_config,
        "saved_paths": saved_paths,
    }
    timestamped = output_dir / f"best_candidate_ablation_{timestamp}.json"
    latest = output_dir / "latest_best_candidate_ablation.json"
    with timestamped.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    shutil.copyfile(timestamped, latest)
    return {"timestamped": timestamped, "latest": latest}


def run_best_candidate_ablation(args: argparse.Namespace) -> dict[str, Any]:
    from utils.console import ensure_utf8_console

    ensure_utf8_console()

    import torch
    from experiments.mechanism_analysis import ablation_study as exp2

    candidate_path = Path(args.candidate_file)
    if not candidate_path.is_absolute():
        candidate_path = PROJECT_ROOT / candidate_path
    candidate = load_best_candidate(candidate_path)
    seeds = _configure_exp2(exp2, args, candidate)
    model_configs = build_ablation_model_configs(candidate)

    print("[best-ablation] candidate:", candidate.get("candidate_id", "unknown"))
    print("[best-ablation] models:", ", ".join(model_configs.keys()))

    exp2.set_random_seed(seeds[0])
    train_loader, val_loader, test_loader, _x_scaler, y_scaler = exp2.load_and_prepare_data(
        exp2.TRAIN_CONFIG["batch_size"],
        subset_ratio=float(args.subset_ratio),
        initial_load_seed=seeds[0],
    )
    if train_loader is None:
        raise RuntimeError("data loading failed")

    device = exp2.TRAIN_CONFIG["device"]
    scaler_mean = torch.from_numpy(y_scaler.mean_.astype("float32")).to(device)
    scaler_scale = torch.from_numpy(y_scaler.scale_.astype("float32")).to(device)

    results_dir_path, models_dir_path = exp2.get_experiment_dirs(PROJECT_ROOT, "exp2_ablation")
    phase_name = "best_candidate_ablation"
    phase_output = exp2.run_ablation_phase(
        phase_name=phase_name,
        models_to_run=model_configs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
        num_runs=int(args.num_runs),
        random_seeds=seeds,
        models_save_dir=str(models_dir_path),
    )
    phase_outputs = OrderedDict([(phase_name, phase_output)])
    saved_paths = exp2.save_phase_outputs_to_files(phase_outputs, str(results_dir_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_paths = _write_metadata(
        output_dir=results_dir_path,
        timestamp=timestamp,
        candidate=candidate,
        run_config={
            "subset_ratio": float(args.subset_ratio),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "num_runs": int(args.num_runs),
            "seeds": seeds[: int(args.num_runs)],
            "prediction_length": int(args.prediction_length),
            "prediction_horizons": parse_prediction_horizons(
                args.prediction_horizons,
                fallback=int(args.prediction_length),
            ),
            "train_supervision_protocol": str(args.train_supervision_protocol),
            "eval_protocol": str(args.eval_protocol),
            "eval_ar_seed_mode": str(args.eval_ar_seed_mode),
            "label_len": int(args.label_len),
            "candidate_file": str(candidate_path),
        },
        saved_paths=saved_paths,
    )
    print(f"[best-ablation] wrote {metadata_paths['latest']}")
    return {"phase_output": phase_output, "saved_paths": saved_paths, "metadata_paths": metadata_paths}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-file",
        type=str,
        default="experiments/exp2_ablation/results/best_candidate_config.json",
    )
    parser.add_argument("--subset-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--prediction-horizons", type=str, default="")
    parser.add_argument("--train-supervision-protocol", type=str, default="source_context_pred_window")
    parser.add_argument("--eval-protocol", type=str, default="source_context_decoder")
    parser.add_argument("--eval-ar-seed-mode", type=str, default="zero")
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_best_candidate_ablation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
