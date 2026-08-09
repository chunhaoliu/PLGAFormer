#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run resumable formal SOTA comparison units and save each model/seed immediately."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.formal_evidence import FINAL_MODEL_FLAGS


MODEL_KEY_TO_NAME = {
    "baseline": "Transformer (baseline)",
    "transformer": "Transformer (baseline)",
    "full": "PLGAFormer (proposed)",
    "proposed": "PLGAFormer (proposed)",
    "plgaformer": "PLGAFormer (proposed)",
    "pit": "PIT",
    "kalman": "Kalman",
    "kinematic": "Spherical kinematics",
    "rotating_3dof": "Rotating-Earth 3-DOF",
    "dlinear": "DLinear",
    "af_ciln": "AF-CILN",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "patchtst": "PatchTST",
    "fedformer": "FEDformer",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
}


FORMAL_FIELDNAMES = [
    "run_id",
    "seed",
    "model_key",
    "model",
    "model_type",
    "dataset_protocol",
    "dataset_sha256",
    "sampling_interval_s",
    "horizon",
    "horizon_steps",
    "horizon_s",
    "metric",
    "value",
    "timestamp",
    "subset_ratio",
    "epochs",
    "batch_size",
    "prediction_horizons",
    "train_supervision_protocol",
    "eval_protocol",
    "label_len",
    "mixed_precision",
    "mixed_precision_dtype",
    "cache_physics_prior",
    "physics_prior_cache_batch_size",
    "checkpoint",
]


def _canonical_model_key(model_name: str, model_type: str | None = None) -> str:
    mt = str(model_type or "").lower().strip()
    if mt in {"transformer", "baseline"}:
        return "baseline"
    if mt == "plgaformer":
        return "full"
    for key, name in MODEL_KEY_TO_NAME.items():
        if name == model_name and key not in {"baseline", "proposed"}:
            return key
    return mt or model_name.lower().replace(" ", "_").replace("/", "_")


def select_model_configs_by_key(
    model_keys: str,
    comparison_models: OrderedDict[str, dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
) -> OrderedDict[str, dict[str, Any]]:
    """Select SOTA model configs by short keys while preserving CLI order."""
    if comparison_models is None:
        from experiments.overall_prediction import main_results as exp1

        comparison_models = exp1.COMPARISON_MODELS
    requested = [item.strip().lower() for item in str(model_keys).split(",") if item.strip()]
    if not requested:
        requested = [
            "transformer",
            "plgaformer",
            "pit",
            "kinematic",
            "rotating_3dof",
            "dlinear",
            "patchtst",
            "itransformer",
            "af_ciln",
        ]
    selected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for key in requested:
        if key not in MODEL_KEY_TO_NAME:
            raise ValueError(f"Unknown SOTA model key: {key}")
        model_name = MODEL_KEY_TO_NAME[key]
        if model_name not in comparison_models:
            raise ValueError(f"Model config not available for key {key}: {model_name}")
        selected[model_name] = dict(comparison_models[model_name])
    return selected


def append_metric_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Append metric rows to a CSV, writing the header only once."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        handle = out_path.open("a", encoding="utf-8", newline="")
    except PermissionError:
        fallback = out_path.with_name(
            f"{out_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{out_path.suffix}"
        )
        handle = fallback.open("w", encoding="utf-8", newline="")
        out_path = fallback
        write_header = True
    with handle as f:
        writer = csv.DictWriter(f, fieldnames=FORMAL_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return out_path


def apply_plgaformer_candidate_overrides(
    model_config: dict[str, Any], candidate_file: str | Path | None
) -> dict[str, Any]:
    """Apply the selected PLGAFormer candidate to the SOTA proposed model config."""
    if not candidate_file:
        return {}
    path = Path(candidate_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() or str(model_config.get("model_type", "")).lower() != "plgaformer":
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("best_candidate", payload)
    if not isinstance(candidate, dict):
        return {}
    if "alpha" in candidate:
        model_config["physics_loss_weight"] = float(candidate.get("alpha") or 0.0)
    innovations = candidate.get("innovations", {}) if isinstance(candidate.get("innovations", {}), dict) else {}
    kwargs = {
        "dropout": float(candidate["dropout"]) if "dropout" in candidate else None,
        "use_sparse_attention": bool(innovations.get("use_sparse_attention", False)),
        "use_physics_corrector": bool(innovations.get("use_physics_corrector", False)),
        "use_multi_head_output": bool(innovations.get("use_multi_head_output", True)),
        "use_adaptive_fusion": True,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return seeds or [42]


def _configure_exp1(exp1: Any, args: argparse.Namespace) -> tuple[list[int], list[int]]:
    from scripts.run_best_candidate_ablation import parse_prediction_horizons

    horizons = parse_prediction_horizons(args.prediction_horizons, fallback=int(args.prediction_length))
    seeds = _parse_seeds(args.seeds)
    exp1.RANDOM_SEEDS = seeds
    exp1.NUM_RUNS = len(seeds)
    exp1.PREDICTION_HORIZONS = horizons
    exp1.SOTA_DATA_SUBSET_RATIO = float(args.subset_ratio)
    exp1.TRAIN_SUPERVISION_PROTOCOL = str(args.train_supervision_protocol)
    exp1.EVAL_PROTOCOL = str(args.eval_protocol)
    exp1.EVAL_AR_SEED_MODE = str(args.eval_ar_seed_mode)
    exp1._base_train_config["epochs"] = int(args.epochs)
    exp1._base_train_config["batch_size"] = int(args.batch_size)
    exp1._base_train_config["pred_len"] = max(horizons)
    exp1._base_train_config["prediction_horizons"] = horizons
    exp1._base_train_config["label_len"] = int(args.label_len)
    exp1._base_train_config["early_stopping_patience"] = int(args.patience)
    exp1._base_train_config["warmup_epochs"] = int(args.warmup_epochs)
    exp1._base_train_config["learning_rate"] = float(args.learning_rate)
    exp1._base_train_config["lr"] = float(args.learning_rate)
    exp1._base_train_config["weight_decay"] = float(args.weight_decay)
    exp1._base_train_config["gradient_clip_norm"] = float(args.gradient_clip_norm)
    exp1._base_train_config["use_mixed_precision"] = bool(args.amp)
    exp1._base_train_config["mixed_precision_dtype"] = str(args.amp_dtype)
    exp1._base_train_config["cache_physics_prior"] = bool(args.cache_physics_prior)
    exp1._base_train_config["physics_prior_cache_batch_size"] = int(
        args.physics_prior_cache_batch_size
    )
    exp1._base_train_config["dataloader_workers"] = int(args.workers)
    exp1._base_train_config["pin_memory"] = bool(args.workers > 0)
    exp1._base_train_config["persistent_workers"] = bool(args.workers > 0)
    return seeds, horizons


def _trajectory_ids_for_eval(exp1: Any, subset_ratio: float):
    import numpy as np

    try:
        data = exp1.load_hgv_dataset(str(exp1.get_dataset_npz_path(PROJECT_ROOT)), require_trajectory_level=True).raw
        ids = np.asarray(data["trajectory_ids_test"], dtype=np.int64)
        if subset_ratio is not None and subset_ratio < 1.0:
            ids = ids[exp1._stable_indices(len(ids), subset_ratio, rng_seed=42)]
        return ids
    except Exception:
        return None


def _rebuild_train_loader_for_seed(loader: Any, exp1: Any, seed: int):
    """Re-seed shuffling without reloading and rescaling the formal dataset."""
    loader_kwargs: dict[str, Any] = {
        "num_workers": int(loader.num_workers),
        "pin_memory": bool(loader.pin_memory),
        "generator": exp1.build_torch_generator(int(seed)),
        "collate_fn": loader.collate_fn,
        "drop_last": bool(loader.drop_last),
        "timeout": float(loader.timeout),
    }
    if loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(loader.persistent_workers)
        if loader.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(loader.prefetch_factor)
        loader_kwargs["worker_init_fn"] = exp1.seed_worker
    return exp1.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        **loader_kwargs,
    )


def resolve_evidence_tier(subset_ratio: float, selection_only: bool) -> str:
    """Route validation-only pilots away from paper-facing test evidence."""
    if selection_only:
        return "convergence_pilot"
    return "final" if float(subset_ratio) >= 1.0 else "screening"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_formal_dataset_identity() -> dict[str, Any]:
    """Read the active dataset identity from the formal-v3 contract."""
    from utils.formal_evidence import load_formal_config

    config = load_formal_config(PROJECT_ROOT / "configs" / "formal_v3.json")
    return {
        "dataset_protocol": str(config["dataset"]["protocol"]),
        "dataset_sha256": str(config["dataset"]["sha256"]),
        "sampling_interval_s": float(config["dataset"]["sampling_interval_s"]),
    }


def build_protocol_identity(
    args: argparse.Namespace,
    horizons: list[int],
    run_signature: str,
    dataset_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact validation-selection identity used for safe resumption."""
    dataset_identity = dataset_identity or _default_formal_dataset_identity()
    sampling_interval_s = float(dataset_identity.get("sampling_interval_s", 1.0))
    return {
        "dataset_protocol": str(dataset_identity["dataset_protocol"]),
        "dataset_sha256": str(dataset_identity.get("dataset_sha256", "")),
        "sampling_interval_s": sampling_interval_s,
        "run_signature": str(run_signature),
        "subset_ratio": float(args.subset_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "prediction_length": int(args.prediction_length),
        "prediction_horizons": [int(item) for item in horizons],
        "prediction_horizons_s": [
            float(item) * sampling_interval_s for item in horizons
        ],
        "train_supervision_protocol": str(args.train_supervision_protocol),
        "eval_protocol": str(args.eval_protocol),
        "eval_ar_seed_mode": str(args.eval_ar_seed_mode),
        "label_len": int(args.label_len),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "patience": int(args.patience),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "mixed_precision": bool(args.amp),
        "mixed_precision_dtype": (
            str(args.amp_dtype) if bool(args.amp) else "float32"
        ),
        "cache_physics_prior": bool(args.cache_physics_prior),
        "physics_prior_cache_batch_size": (
            int(args.physics_prior_cache_batch_size)
            if bool(args.cache_physics_prior)
            else 0
        ),
    }


def selection_record_matches(
    payload: dict[str, Any],
    *,
    seed: int,
    model_key: str,
    expected_identity: dict[str, Any],
    allow_legacy_identity: bool = True,
) -> bool:
    """Return True only for a complete, validation-only record of this unit."""
    if int(payload.get("seed", -1)) != int(seed):
        return False
    if str(payload.get("model_key", "")).lower() != str(model_key).lower():
        return False
    if payload.get("evidence_tier") != "convergence_pilot":
        return False
    if payload.get("test_evaluation_performed") is not False:
        return False

    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.is_file():
        return False
    recorded_hash = payload.get("checkpoint_sha256")
    if recorded_hash and str(recorded_hash).lower() != sha256_file(checkpoint):
        return False

    history = payload.get("training_history", {})
    if history.get("fixed_parameter_model") is not True:
        if not history.get("epochs_completed") or history.get("best_epoch") is None:
            return False
        if history.get("best_val_loss") is None:
            return False

    recorded_identity = payload.get("protocol_identity")
    if recorded_identity:
        return recorded_identity == expected_identity
    # Records without the signed identity are historical pilots. They remain
    # readable but cannot be resumed by the formal-v3 runner, even when the
    # caller explicitly asks for a legacy-compatible scan.
    return False


def find_existing_selection_record(
    formal_dir: str | Path,
    *,
    seed: int,
    model_key: str,
    expected_identity: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    """Find the newest exact, complete validation-selection record."""
    candidates = sorted(
        Path(formal_dir).glob(f"formal_sota_seed{int(seed)}_{model_key}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if selection_record_matches(
            payload,
            seed=seed,
            model_key=model_key,
            expected_identity=expected_identity,
        ):
            return path, payload
    return None


def _flatten_metric_rows(
    *,
    run_id: str,
    timestamp: str,
    seed: int,
    model_name: str,
    model_config: dict[str, Any],
    eval_results: dict[int, dict[str, float]],
    args: argparse.Namespace,
    checkpoint: Path,
    dataset_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_type = str(model_config.get("model_type", ""))
    dataset_identity = dataset_identity or _default_formal_dataset_identity()
    sampling_interval_s = float(dataset_identity.get("sampling_interval_s", 1.0))
    for horizon, metrics in sorted(eval_results.items()):
        for metric, value in sorted(metrics.items()):
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "seed": seed,
                    "model_key": _canonical_model_key(model_name, model_type),
                    "model": model_name,
                    "model_type": model_type,
                    "dataset_protocol": dataset_identity["dataset_protocol"],
                    "dataset_sha256": dataset_identity.get("dataset_sha256", ""),
                    "sampling_interval_s": sampling_interval_s,
                    "horizon": horizon,
                    "horizon_steps": horizon,
                    "horizon_s": float(horizon) * sampling_interval_s,
                    "metric": metric,
                    "value": value,
                    "subset_ratio": float(args.subset_ratio),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "prediction_horizons": str(args.prediction_horizons),
                    "train_supervision_protocol": str(args.train_supervision_protocol),
                    "eval_protocol": str(args.eval_protocol),
                    "label_len": int(args.label_len),
                    "mixed_precision": bool(args.amp),
                    "mixed_precision_dtype": (
                        str(args.amp_dtype) if bool(args.amp) else "float32"
                    ),
                    "cache_physics_prior": bool(args.cache_physics_prior),
                    "physics_prior_cache_batch_size": (
                        int(args.physics_prior_cache_batch_size)
                        if bool(args.cache_physics_prior)
                        else 0
                    ),
                    "checkpoint": str(checkpoint),
                }
            )
    return rows


def run_formal_sota_units(args: argparse.Namespace) -> dict[str, Any]:
    from utils.console import ensure_utf8_console

    ensure_utf8_console()

    from utils.formal_evidence import load_formal_config

    formal_config_path = getattr(
        args,
        "config",
        str(PROJECT_ROOT / "configs" / "formal_v3.json"),
    )
    formal_config = load_formal_config(formal_config_path)
    if args.af_ciln_root:
        af_ciln_root = Path(args.af_ciln_root)
        if not af_ciln_root.is_absolute():
            af_ciln_root = PROJECT_ROOT / af_ciln_root
        if not af_ciln_root.is_dir():
            raise FileNotFoundError(f"AF-CILN checkout is missing: {af_ciln_root}")
        os.environ["HGV_AF_CILN_ROOT"] = str(af_ciln_root.resolve())

    from data_generation.data_paths import (
        configure_protocol_scaler_paths,
        get_dataset_npz_path,
    )
    from utils.experiment_io import get_formal_protocol_dirs
    from utils.trajectory_protocol import dataset_file_identity

    requested_dataset = getattr(args, "dataset_path", None)
    if requested_dataset:
        dataset_path = Path(requested_dataset)
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
    if (
        str(dataset_identity.get("dataset_protocol")) != str(formal_config["dataset"]["protocol"])
        or str(dataset_identity.get("dataset_sha256")) != str(formal_config["dataset"]["sha256"])
    ):
        raise RuntimeError(
            "Formal runner dataset identity does not match configs/formal_v3.json: "
            f"{dataset_identity.get('dataset_protocol')} / {dataset_identity.get('dataset_sha256')}"
        )

    import numpy as np
    import torch
    from experiments.overall_prediction import main_results as exp1
    seeds, horizons = _configure_exp1(exp1, args)
    selected_models = select_model_configs_by_key(args.models, exp1.COMPARISON_MODELS)
    evidence_tier = resolve_evidence_tier(
        float(args.subset_ratio), bool(args.selection_only)
    )
    formal_dir, formal_models_dir = get_formal_protocol_dirs(
        PROJECT_ROOT,
        "exp1_sota",
        str(dataset_identity["dataset_protocol"]),
        evidence_tier,
    )
    metrics_csv = formal_dir / "formal_sota_runs.csv"
    trajectory_ids = (
        None
        if args.selection_only
        else _trajectory_ids_for_eval(exp1, float(args.subset_ratio))
    )
    device = exp1._base_train_config["device"]
    exp1.set_random_seed(seeds[0])
    base_train_loader, val_loader, test_loader, scaler, output_scaler = (
        exp1.load_and_prepare_data(
            exp1._base_train_config["batch_size"],
            seed=seeds[0],
            subset_ratio=float(args.subset_ratio),
        )
    )
    if base_train_loader is None:
        raise RuntimeError("data loading failed")
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

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        exp1.set_random_seed(seed)
        train_loader = _rebuild_train_loader_for_seed(
            base_train_loader, exp1, seed
        )

        for model_name, model_config in selected_models.items():
            model_key = _canonical_model_key(model_name, model_config.get("model_type"))
            if args.selection_only and args.skip_existing:
                existing = find_existing_selection_record(
                    formal_dir,
                    seed=seed,
                    model_key=model_key,
                    expected_identity=protocol_identity,
                )
                if existing is not None:
                    existing_path, existing_payload = existing
                    completed.append(
                        {
                            "run_id": existing_payload["run_id"],
                            "json": str(existing_path),
                            "checkpoint": str(existing_payload["checkpoint"]),
                            "reused": True,
                        }
                    )
                    print(f"[formal-sota] reuse exact selection record {existing_path}")
                    continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"formal_sota_seed{seed}_{model_key}_{timestamp}"
            print(f"[formal-sota] run {run_id} horizons={horizons}")
            exp1.set_random_seed(seed)
            plgaformer_kwargs = exp1._model_reconstruction_kwargs(
                model_config["model_type"], scaler, output_scaler
            ) or {}
            plgaformer_kwargs.update(
                apply_plgaformer_candidate_overrides(model_config, args.candidate_file)
            )
            model = exp1.create_model(
                model_config["model_type"],
                input_dim=6,
                device=device,
                plgaformer_kwargs=plgaformer_kwargs,
            )
            trained_model, training_history = exp1.train_model(
                model_name,
                model_config,
                train_loader,
                val_loader,
                output_scaler_mean,
                output_scaler_scale,
                model_override=model,
            )
            checkpoint = formal_models_dir / f"{run_id}.pth"
            torch.save(trained_model.state_dict(), checkpoint)
            checkpoint_sha256 = sha256_file(checkpoint)
            if args.selection_only:
                eval_results: dict[int, dict[str, float]] = {}
            else:
                eval_results = exp1.run_evaluation(
                    trained_model,
                    test_loader,
                    output_scaler_mean,
                    output_scaler_scale,
                    device,
                    horizons,
                    trajectory_ids=trajectory_ids,
                )
                rows = _flatten_metric_rows(
                    run_id=run_id,
                    timestamp=timestamp,
                    seed=seed,
                    model_name=model_name,
                    model_config=model_config,
                    eval_results=eval_results,
                    args=args,
                    checkpoint=checkpoint,
                    dataset_identity=dataset_identity,
                )
                metrics_csv = append_metric_rows(metrics_csv, rows)
            run_json = formal_dir / f"{run_id}.json"
            record_model_config = dict(model_config)
            if model_key == "full":
                record_model_config.update(FINAL_MODEL_FLAGS)
            payload = {
                "schema_version": 2,
                "formal_config_sha256": formal_config.config_sha256,
                "run_id": run_id,
                "timestamp": timestamp,
                "seed": seed,
                "model": model_name,
                "model_key": model_key,
                "model_config": record_model_config,
                "config": vars(args),
                "horizons": horizons,
                "eval_results": exp1.convert_to_serializable(eval_results),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "training_history": training_history,
                "metrics_csv": None if args.selection_only else str(metrics_csv),
                "evidence_tier": evidence_tier,
                "test_evaluation_performed": not bool(args.selection_only),
                "protocol_identity": protocol_identity,
                "dataset_identity": dataset_identity,
                "scaler_paths": {key: str(path) for key, path in scaler_paths.items()},
            }
            run_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            completed.append({"run_id": run_id, "json": str(run_json), "checkpoint": str(checkpoint)})
            print(f"[formal-sota] wrote {run_json}")
    return {
        "metrics_csv": str(metrics_csv),
        "completed": completed,
        "dataset_identity": dataset_identity,
        "formal_config_sha256": formal_config.config_sha256,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "formal_v3.json"),
        help="Frozen formal-v3 orchestration configuration.",
    )
    parser.add_argument("--candidate-file", type=str, default=None)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Formal dataset artifact. Defaults to the audited multi-regime v2.1 dataset; "
            "HGV_DATASET_PATH remains an explicit override."
        ),
    )
    parser.add_argument(
        "--af-ciln-root",
        type=str,
        default=None,
        help="Audited official AF-CILN checkout used by the external adapter.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "transformer,plgaformer,pit,kinematic,rotating_3dof,"
            "dlinear,patchtst,itransformer,af_ciln"
        ),
    )
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=256)
    parser.add_argument("--prediction-horizons", type=str, default="32,64,128,256")
    parser.add_argument("--train-supervision-protocol", type=str, default="source_context_pred_window")
    parser.add_argument("--eval-protocol", type=str, default="source_context_decoder")
    parser.add_argument("--eval-ar-seed-mode", type=str, default="zero")
    parser.add_argument("--label-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. The formal tensors already reside in RAM; "
            "workers=0 avoids Windows spawn overhead and is the measured default."
        ),
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Autocast dtype; bfloat16 is the stable default for RTX 40-series GPUs.",
    )
    parser.add_argument(
        "--cache-physics-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute the deterministic PLGAFormer physics prior once per split.",
    )
    parser.add_argument("--physics-prior-cache-batch-size", type=int, default=512)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse only exact, complete validation-selection records matching "
            "the current protocol identity. Has an effect with --selection-only."
        ),
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Train and save validation histories/checkpoints without evaluating "
            "the test split. Outputs are isolated under "
            "formal_v3/<dataset_protocol>/convergence_pilot."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run_formal_sota_units(parse_args(argv))
    print("[formal-sota] completed:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
