#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate validation-selected formal ablation checkpoints once on test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_formal_ablation_unit import (  # noqa: E402
    _configure_exp2,
    _flatten_metric_rows,
    _model_key_for_name,
    _parse_seeds,
    append_metric_rows,
    build_ablation_protocol_identity,
    find_existing_ablation_selection,
    select_model_configs_by_key,
    sha256_file,
)


def find_existing_frozen_ablation(
    output_dir: str | Path,
    *,
    source_record_sha256: str,
    checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted(
        Path(output_dir).glob("frozen_ablation_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("test_evaluation_performed") is True
            and payload.get("source_selection_record_sha256") == source_record_sha256
            and payload.get("checkpoint_sha256") == checkpoint_sha256
        ):
            return path, payload
    return None


def convert_to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {key: convert_to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_to_serializable(item) for item in value]
    return value


def evaluate_frozen_ablation(args: argparse.Namespace) -> dict[str, Any]:
    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    from data_generation.data_paths import (
        configure_protocol_scaler_paths,
        get_dataset_npz_path,
    )
    from utils.experiment_io import get_formal_protocol_dirs
    from utils.trajectory_protocol import dataset_file_identity

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path
        os.environ["HGV_DATASET_PATH"] = str(dataset_path.resolve())
    elif not os.getenv("HGV_DATASET_PATH"):
        os.environ["HGV_DATASET_PATH"] = str(
            (
                PROJECT_ROOT
                / "data_generation"
                / "data"
                / "processed"
                / "hgv_multiregime_dataset_v2_1.npz"
            ).resolve()
        )
    dataset_identity = dataset_file_identity(get_dataset_npz_path(PROJECT_ROOT))
    scaler_paths = configure_protocol_scaler_paths(
        dataset_identity, force=True, namespace="exp2_ablation"
    )

    import torch
    from experiments.exp2_ablation import ablation_study as exp2

    if args.phase not in exp2.ABLATION_PHASES:
        raise ValueError(
            f"Unknown phase {args.phase!r}; choose from {list(exp2.ABLATION_PHASES)}"
    )
    candidate = {"candidate_id": "locked_current_design"}
    horizons = _configure_exp2(exp2, args, candidate, dataset_identity)
    seeds = _parse_seeds(args.seeds)
    model_configs = select_model_configs_by_key(
        exp2.ABLATION_PHASES[args.phase]["models"], args.models
    )
    protocol_identity = build_ablation_protocol_identity(
        args, horizons, dataset_identity
    )

    default_selection_dir, _ = get_formal_protocol_dirs(
        PROJECT_ROOT,
        "exp2_ablation",
        str(dataset_identity["dataset_protocol"]),
        "convergence_pilot",
    )
    selection_dir = (
        Path(args.selection_dir)
        if args.selection_dir
        else default_selection_dir
    )
    if not selection_dir.is_absolute():
        selection_dir = PROJECT_ROOT / selection_dir
    default_output_dir, _ = get_formal_protocol_dirs(
        PROJECT_ROOT,
        "exp2_ablation",
        str(dataset_identity["dataset_protocol"]),
        "final_frozen",
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "formal_ablation_frozen_runs.csv"

    dataset_path = Path(str(dataset_identity["dataset_path"]))
    dataset_sha256 = str(dataset_identity["dataset_sha256"])
    device = exp2.TRAIN_CONFIG["device"]

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        exp2.set_random_seed(seed)
        train_loader, val_loader, test_loader, x_scaler, y_scaler = (
            exp2.load_and_prepare_data(
                exp2.TRAIN_CONFIG["batch_size"],
                subset_ratio=float(args.subset_ratio),
                initial_load_seed=seed,
            )
        )
        if train_loader is None or val_loader is None or test_loader is None:
            raise RuntimeError("formal ablation data loading failed")
        scaler_mean = torch.from_numpy(y_scaler.mean_.astype("float32")).to(device)
        scaler_scale = torch.from_numpy(y_scaler.scale_.astype("float32")).to(device)
        output_scaler_mean = y_scaler.mean_.astype("float32")
        output_scaler_scale = y_scaler.scale_.astype("float32")
        source_scaler_mean = np.concatenate(
            [y_scaler.mean_, x_scaler.mean_[3:]]
        ).astype("float32")
        source_scaler_scale = np.concatenate(
            [y_scaler.scale_, x_scaler.scale_[3:]]
        ).astype("float32")

        for model_name, model_config in model_configs.items():
            model_key = _model_key_for_name(model_name)
            resolved = find_existing_ablation_selection(
                selection_dir,
                phase=args.phase,
                seed=seed,
                model_key=model_key,
                protocol_identity=protocol_identity,
                model_config=model_config,
            )
            if resolved is None:
                raise FileNotFoundError(
                    "No exact validation-selected ablation checkpoint for "
                    f"phase={args.phase}, model={model_key}, seed={seed}"
                )
            source_path, source_payload = resolved
            checkpoint = Path(source_payload["checkpoint"]).resolve()
            source_sha256 = sha256_file(source_path)
            checkpoint_sha256 = sha256_file(checkpoint)
            existing = find_existing_frozen_ablation(
                output_dir,
                source_record_sha256=source_sha256,
                checkpoint_sha256=checkpoint_sha256,
            )
            if existing is not None:
                existing_path, existing_payload = existing
                completed.append(
                    {
                        "run_id": existing_payload["run_id"],
                        "json": str(existing_path),
                        "checkpoint": str(checkpoint),
                        "reused": True,
                    }
                )
                print(
                    "[frozen-ablation] reuse completed test evaluation "
                    f"{existing_path}"
                )
                continue

            model = exp2.create_model(
                model_type=model_config["model_type"],
                input_dim=6,
                device=device,
                innovations=model_config.get("innovations"),
                model_config_override=model_config,
                input_scaler_mean=source_scaler_mean,
                input_scaler_scale=source_scaler_scale,
                output_scaler_mean=output_scaler_mean,
                output_scaler_scale=output_scaler_scale,
            )
            try:
                state_dict = torch.load(
                    checkpoint, map_location=device, weights_only=True
                )
            except TypeError:
                state_dict = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            eval_results = exp2.run_evaluation(
                model,
                test_loader,
                scaler_mean,
                scaler_scale,
                device,
                horizons,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = (
                f"frozen_ablation_{args.phase}_seed{seed}_{model_key}_{timestamp}"
            )
            rows = _flatten_metric_rows(
                run_id=run_id,
                timestamp=timestamp,
                seed=seed,
                model_name=model_name,
                eval_results=eval_results,
                args=args,
                candidate=candidate,
                checkpoint=checkpoint,
                phase=args.phase,
                source="frozen_validation_selection",
                source_run_signature=protocol_identity["run_signature"],
                dataset_identity=dataset_identity,
            )
            metrics_csv = append_metric_rows(metrics_csv, rows)
            payload = {
                "schema_version": 1,
                "run_id": run_id,
                "timestamp": timestamp,
                "seed": int(seed),
                "model": model_name,
                "model_key": model_key,
                "model_config": model_config,
                "phase": args.phase,
                "candidate": candidate,
                "config": vars(args),
                "horizons": horizons,
                "eval_results": convert_to_serializable(eval_results),
                "evidence_tier": "final_frozen",
                "test_evaluation_performed": True,
                "protocol_identity": protocol_identity,
                "dataset_path": str(dataset_path),
                "dataset_sha256": dataset_sha256,
                "dataset_identity": dataset_identity,
                "scaler_paths": {key: str(path) for key, path in scaler_paths.items()},
                "source_selection_record": str(source_path.resolve()),
                "source_selection_record_sha256": source_sha256,
                "selection_best_epoch": source_payload.get("history", {}).get(
                    "best_epoch"
                ),
                "selection_best_val_loss": source_payload.get("history", {}).get(
                    "best_val_loss"
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "metrics_csv": str(metrics_csv),
            }
            run_json = output_dir / f"{run_id}.json"
            run_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed.append(
                {
                    "run_id": run_id,
                    "json": str(run_json),
                    "checkpoint": str(checkpoint),
                    "reused": False,
                }
            )
            print(f"[frozen-ablation] wrote {run_json}")
    return {"metrics_csv": str(metrics_csv), "completed": completed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--selection-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=256)
    parser.add_argument("--prediction-horizons", type=str, default="32,64,128,256")
    parser.add_argument(
        "--train-supervision-protocol",
        type=str,
        default="source_context_pred_window",
    )
    parser.add_argument(
        "--eval-protocol", type=str, default="source_context_decoder"
    )
    parser.add_argument("--eval-ar-seed-mode", type=str, default="zero")
    parser.add_argument("--label-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = evaluate_frozen_ablation(parse_args(argv))
    print("[frozen-ablation] completed:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
