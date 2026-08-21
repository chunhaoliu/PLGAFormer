#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate validation-selected formal SOTA checkpoints exactly once on test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_formal_sota_unit import (  # noqa: E402
    _canonical_model_key,
    _configure_exp1,
    _flatten_metric_rows,
    _trajectory_ids_for_eval,
    append_metric_rows,
    build_protocol_identity,
    find_existing_selection_record,
    select_model_configs_by_key,
    sha256_file,
    validate_final_plgaformer_contract,
)
from utils.formal_runtime import formal_runtime_defaults, validate_formal_runtime  # noqa: E402
from utils.mainline_contract import (  # noqa: E402
    ACTIVE_CONFIG_PATH,
    ACTIVE_PLGAFORMER_FLAGS,
    load_mainline_config,
)


def find_existing_frozen_evaluation(
    output_dir: str | Path,
    *,
    source_record_sha256: str,
    checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Find a completed test evaluation for the exact frozen source."""
    for path in sorted(
        Path(output_dir).glob("frozen_sota_*.json"),
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


def _configure_tslib_checkout(path_value: str | None) -> None:
    raw = str(path_value or os.getenv("HGV_TSLIB_ROOT", "")).strip()
    if not raw:
        return
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(f"TSLib checkout is missing: {path}")
    os.environ["HGV_TSLIB_ROOT"] = str(path.resolve())


def _validate_source_plgaformer_config(model_config: dict[str, Any]) -> None:
    """Reject frozen checkpoints whose recorded active flags have drifted."""
    for field, expected in ACTIVE_PLGAFORMER_FLAGS.items():
        actual = model_config.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Frozen PLGAFormer source {field} conflicts with active value "
                f"{expected!r}; got {actual!r}"
            )


def evaluate_frozen_selections(args: argparse.Namespace) -> dict[str, Any]:
    formal_config = load_mainline_config()
    runtime_args = argparse.Namespace(**vars(args), selection_only=False)
    validate_formal_runtime(runtime_args, formal_config)

    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    _configure_tslib_checkout(args.tslib_root)

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
        dataset_identity, force=True, namespace="exp1_sota"
    )

    import numpy as np
    import torch
    from experiments.overall_prediction import main_results as exp1

    seeds, horizons = _configure_exp1(exp1, args)
    selected_models = select_model_configs_by_key(args.models, exp1.COMPARISON_MODELS)
    public_types = {"transformer", "dlinear", "patchtst", "itransformer"}
    if (
        any(str(config.get("model_type", "")).lower() in public_types for config in selected_models.values())
        and not os.getenv("HGV_TSLIB_ROOT", "").strip()
    ):
        raise RuntimeError(
            "Frozen public-baseline evaluation requires --tslib-root or HGV_TSLIB_ROOT."
        )
    default_selection_dir, _ = get_formal_protocol_dirs(
        PROJECT_ROOT,
        "exp1_sota",
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
        "exp1_sota",
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
    metrics_csv = output_dir / "formal_sota_frozen_runs.csv"

    protocol_identity: dict[str, Any] | None = None
    dataset_path = Path(str(dataset_identity["dataset_path"]))
    dataset_sha256 = str(dataset_identity["dataset_sha256"])
    device = exp1._base_train_config["device"]
    trajectory_ids = _trajectory_ids_for_eval(exp1, float(args.subset_ratio))

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        exp1.set_random_seed(seed)
        train_loader, val_loader, test_loader, scaler, output_scaler = (
            exp1.load_and_prepare_data(
                exp1._base_train_config["batch_size"],
                seed=seed,
                subset_ratio=float(args.subset_ratio),
            )
        )
        if train_loader is None or val_loader is None or test_loader is None:
            raise RuntimeError("formal data loading failed")
        if protocol_identity is None:
            run_signature = exp1._current_run_signature()
            protocol_identity = build_protocol_identity(
                args, horizons, run_signature, dataset_identity
            )
        output_scaler_mean = torch.from_numpy(
            output_scaler.mean_.astype(np.float32)
        ).to(device)
        output_scaler_scale = torch.from_numpy(
            output_scaler.scale_.astype(np.float32)
        ).to(device)

        for model_name, default_model_config in selected_models.items():
            model_key = _canonical_model_key(
                model_name, default_model_config.get("model_type")
            )
            is_analytical = default_model_config.get("model_type") in {
                "kinematic",
                "rotating_3dof",
            }
            if is_analytical and seed != seeds[0]:
                continue

            if is_analytical:
                source_payload: dict[str, Any] = {
                    "model_config": default_model_config,
                    "config": {},
                    "training_history": {},
                }
                source_path = None
                source_sha256 = hashlib.sha256(
                    json.dumps(
                        {
                            "protocol_identity": protocol_identity,
                            "model": default_model_config,
                            "parameter_free": True,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                checkpoint = None
                checkpoint_sha256 = "parameter-free"
            else:
                resolved = find_existing_selection_record(
                    selection_dir,
                    seed=seed,
                    model_key=model_key,
                    expected_identity=protocol_identity,
                )
                if resolved is None:
                    raise FileNotFoundError(
                        "No exact validation-selected checkpoint for "
                        f"model={model_key}, seed={seed}, identity={protocol_identity}"
                    )
                source_path, source_payload = resolved
                checkpoint = Path(source_payload["checkpoint"]).resolve()
                source_sha256 = sha256_file(source_path)
                checkpoint_sha256 = sha256_file(checkpoint)
            existing = find_existing_frozen_evaluation(
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
                        "checkpoint": None if checkpoint is None else str(checkpoint),
                        "reused": True,
                    }
                )
                print(f"[frozen-sota] reuse completed test evaluation {existing_path}")
                continue

            model_config = dict(source_payload.get("model_config", default_model_config))
            if model_config.get("model_type") == "plgaformer":
                _validate_source_plgaformer_config(model_config)
            source_provenance = source_payload.get("external_source_provenance")
            if isinstance(source_provenance, dict):
                model_config["external_source_provenance"] = dict(source_provenance)
            reconstruction_kwargs = exp1._model_reconstruction_kwargs(
                model_config["model_type"], scaler, output_scaler
            ) or {}
            model = exp1.create_model(
                model_config["model_type"],
                input_dim=6,
                device=device,
                plgaformer_kwargs=reconstruction_kwargs,
            )
            if model_config["model_type"] == "plgaformer":
                validate_final_plgaformer_contract(model)
            if checkpoint is not None:
                try:
                    state_dict = torch.load(
                        checkpoint, map_location=device, weights_only=True
                    )
                except TypeError:
                    state_dict = torch.load(checkpoint, map_location=device)
                model.load_state_dict(state_dict, strict=True)
            model.eval()

            eval_results = exp1.run_evaluation(
                model,
                test_loader,
                output_scaler_mean,
                output_scaler_scale,
                device,
                horizons,
                trajectory_ids=trajectory_ids,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"frozen_sota_seed{seed}_{model_key}_{timestamp}"
            rows = _flatten_metric_rows(
                run_id=run_id,
                timestamp=timestamp,
                seed=seed,
                model_name=model_name,
                model_config=model_config,
                eval_results=eval_results,
                args=args,
                checkpoint=checkpoint or Path("parameter_free"),
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
                "config": vars(args),
                "horizons": horizons,
                "eval_results": exp1.convert_to_serializable(eval_results),
                "evidence_tier": "final_frozen",
                "test_evaluation_performed": True,
                "protocol_identity": protocol_identity,
                "dataset_path": str(dataset_path),
                "dataset_sha256": dataset_sha256,
                "dataset_identity": dataset_identity,
                "scaler_paths": {key: str(path) for key, path in scaler_paths.items()},
                "source_selection_record": (
                    None if source_path is None else str(source_path.resolve())
                ),
                "source_selection_record_sha256": source_sha256,
                "selection_best_epoch": source_payload.get(
                    "training_history", {}
                ).get("best_epoch"),
                "selection_best_val_loss": source_payload.get(
                    "training_history", {}
                ).get("best_val_loss"),
                "checkpoint": None if checkpoint is None else str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "metrics_csv": str(metrics_csv),
            }
            if isinstance(source_provenance, dict):
                payload["external_source_provenance"] = dict(source_provenance)
            run_json = output_dir / f"{run_id}.json"
            run_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed.append(
                {
                    "run_id": run_id,
                    "json": str(run_json),
                    "checkpoint": None if checkpoint is None else str(checkpoint),
                    "reused": False,
                }
            )
            print(f"[frozen-sota] wrote {run_json}")
    return {"metrics_csv": str(metrics_csv), "completed": completed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = formal_runtime_defaults(load_mainline_config())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(ACTIVE_CONFIG_PATH))
    parser.add_argument("--selection-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--tslib-root", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default="transformer,plgaformer,dlinear,patchtst,itransformer",
    )
    parser.add_argument("--seeds", type=str, default=defaults["seeds"])
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument(
        "--prediction-length", type=int, default=defaults["prediction_length"]
    )
    parser.add_argument(
        "--prediction-horizons", type=str, default=defaults["prediction_horizons"]
    )
    parser.add_argument(
        "--train-supervision-protocol",
        type=str,
        default=defaults["train_supervision_protocol"],
    )
    parser.add_argument(
        "--eval-protocol", type=str, default=defaults["eval_protocol"]
    )
    parser.add_argument(
        "--eval-ar-seed-mode", type=str, default=defaults["eval_ar_seed_mode"]
    )
    parser.add_argument("--label-len", type=int, default=defaults["label_len"])
    parser.add_argument(
        "--learning-rate", type=float, default=defaults["learning_rate"]
    )
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-epochs", type=int, default=defaults["warmup_epochs"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=defaults["gradient_clip_norm"]
    )
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=defaults["amp"]
    )
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--cache-physics-prior",
        action=argparse.BooleanOptionalAction,
        default=defaults["cache_physics_prior"],
    )
    parser.add_argument(
        "--physics-prior-cache-batch-size",
        type=int,
        default=defaults["physics_prior_cache_batch_size"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = evaluate_frozen_selections(parse_args(argv))
    print("[frozen-sota] completed:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
