#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run resumable formal ablation units and persist each model/seed immediately."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.formal_evidence import FINAL_MODEL_FLAGS


MODEL_KEY_TO_NAME = {
    "baseline": "Transformer (baseline)",
    "backbone": "PLGAFormer backbone (no A/B/C)",
    "prior_only": "PLGAFormer prior fusion only",
    "residual_only": "PLGAFormer channel residual only",
    "proposed": "PLGAFormer (proposed)",
    "ecef": "PLGAFormer (ECEF composite)",
    "dynamics_residual": "PLGAFormer (+ dynamics residual)",
    "attention_all": "PLGAFormer A only",
    "temporal_only": "A: temporal only",
    "phase_only": "A: phase only",
    "geometry_only": "A: geometry only",
    "without_temporal": "A w/o temporal",
    "without_phase": "A w/o phase",
    "without_geometry": "A w/o geometry",
    "spherical_prior": "PLGAFormer spherical-prior fusion",
    "schedule_only": "PLGAFormer rotating-prior schedule only",
    "learned_only": "PLGAFormer learned-only backbone",
    # Legacy aliases remain readable for old result summaries and tests.
    "full": "PLGAFormer (A+B+C)",
    "wo_a": "PLGAFormer w/o A",
    "wo_b": "PLGAFormer w/o B",
    "wo_c": "PLGAFormer w/o C",
}

FORMAL_FIELDNAMES = [
    "run_id",
    "seed",
    "model_key",
    "model",
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
    "candidate_id",
    "phase",
    "checkpoint",
    "source",
    "source_run_signature",
]


def _model_key_for_name(name: str) -> str:
    for key, model_name in MODEL_KEY_TO_NAME.items():
        if model_name == name:
            return key
    return name.lower().replace(" ", "_").replace("/", "_")


def select_model_configs_by_key(
    model_configs: OrderedDict[str, dict[str, Any]] | dict[str, dict[str, Any]],
    model_keys: str,
) -> OrderedDict[str, dict[str, Any]]:
    """Select ablation model configs by short keys while preserving CLI order."""
    requested = [item.strip().lower() for item in str(model_keys).split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return OrderedDict((name, dict(config)) for name, config in model_configs.items())
    selected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for key in requested:
        model_name = MODEL_KEY_TO_NAME.get(key)
        if model_name is None:
            normalized_matches = [
                name for name in model_configs
                if _model_key_for_name(name) == key
                or name.lower().replace(" ", "_").replace("/", "_") == key
            ]
            if len(normalized_matches) != 1:
                raise ValueError(f"Unknown or ambiguous model key: {key}")
            model_name = normalized_matches[0]
        if model_name not in model_configs:
            raise ValueError(f"Model config not available for key {key}: {model_name}")
        selected[model_name] = dict(model_configs[model_name])
    return selected


def append_metric_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Append metric rows to a CSV, writing the header only once."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FORMAL_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def _is_scalar_metric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_protocol_matches(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        str(row.get("subset_ratio", "")) == str(float(args.subset_ratio))
        and str(row.get("epochs", "")) == str(int(args.epochs))
        and str(row.get("batch_size", "")) == str(int(args.batch_size))
        and str(row.get("prediction_horizons", "")) == str(args.prediction_horizons)
        and str(row.get("train_supervision_protocol", ""))
        == str(args.train_supervision_protocol)
        and str(row.get("eval_protocol", "")) == str(args.eval_protocol)
        and str(row.get("label_len", "")) == str(int(args.label_len))
        and str(row.get("mixed_precision", "")).lower()
        == str(bool(args.amp)).lower()
        and str(row.get("mixed_precision_dtype", ""))
        == (str(args.amp_dtype) if bool(args.amp) else "float32")
        and str(row.get("cache_physics_prior", "")).lower()
        == str(bool(args.cache_physics_prior)).lower()
        and str(row.get("physics_prior_cache_batch_size", ""))
        == str(
            int(args.physics_prior_cache_batch_size)
            if bool(args.cache_physics_prior)
            else 0
        )
    )


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
        "dataset_path": str(config.resolve(config["dataset"]["relative_path"])),
        "dataset_protocol": str(config["dataset"]["protocol"]),
        "dataset_sha256": str(config["dataset"]["sha256"]),
        "sampling_interval_s": float(config["dataset"]["sampling_interval_s"]),
    }


def resolve_evidence_tier(selection_only: bool) -> str:
    return "convergence_pilot" if selection_only else "final"


def resolve_custom_output_dirs(
    args: argparse.Namespace,
) -> tuple[Path, Path] | None:
    """Resolve an explicitly isolated result/checkpoint pair for supplemental runs."""
    output_dir = getattr(args, "output_dir", None)
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if bool(output_dir) != bool(checkpoint_dir):
        raise ValueError("--output-dir and --checkpoint-dir must be provided together.")
    if not output_dir:
        return None
    results_path = Path(output_dir).expanduser().resolve()
    checkpoints_path = Path(checkpoint_dir).expanduser().resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    return results_path, checkpoints_path

def build_ablation_protocol_identity(
    args: argparse.Namespace,
    horizons: list[int],
    dataset_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_identity = dataset_identity or _default_formal_dataset_identity()
    tracked_files = {
        "dataset": Path(str(dataset_identity["dataset_path"])),
        "ablation": PROJECT_ROOT / "experiments" / "mechanism_analysis" / "ablation_study.py",
        "plgaformer": PROJECT_ROOT / "models" / "plgaformer.py",
        "model_factory": PROJECT_ROOT / "models" / "model_factory.py",
    }
    file_hashes = {name: sha256_file(path) for name, path in tracked_files.items()}
    sampling_interval_s = float(dataset_identity.get("sampling_interval_s", 1.0))
    identity = {
        "dataset_protocol": str(dataset_identity["dataset_protocol"]),
        "dataset_sha256": str(
            dataset_identity.get("dataset_sha256") or file_hashes["dataset"]
        ),
        "sampling_interval_s": sampling_interval_s,
        "phase": str(args.phase),
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
        "files": file_hashes,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["run_signature"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return identity


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def find_existing_ablation_selection(
    formal_dir: str | Path,
    *,
    phase: str,
    seed: int,
    model_key: str,
    protocol_identity: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    candidates = sorted(
        Path(formal_dir).glob(f"formal_{phase}_seed{int(seed)}_{model_key}_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint = Path(str(payload.get("checkpoint", "")))
        if (
            payload.get("evidence_tier") != "convergence_pilot"
            or payload.get("test_evaluation_performed") is not False
            or payload.get("protocol_identity") != protocol_identity
            or _json_normalize(payload.get("model_config", {}))
            != _json_normalize(model_config)
            or not checkpoint.is_file()
        ):
            continue
        recorded_hash = payload.get("checkpoint_sha256")
        if recorded_hash and recorded_hash != sha256_file(checkpoint):
            continue
        history = payload.get("history", {})
        if not history.get("epochs_completed") or history.get("best_epoch") is None:
            continue
        return path, payload
    return None


def completed_units(path: Path, args: argparse.Namespace) -> set[tuple[str, int, str]]:
    """Return phase/seed/model units already persisted under the same protocol."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[tuple[str, int, str], set[tuple[str, str]]] = {}
    for row in rows:
        if not _unit_protocol_matches(row, args):
            continue
        try:
            key = (str(row["phase"]), int(row["seed"]), str(row["model_key"]))
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(key, set()).add(
            (str(row.get("horizon", "")), str(row.get("metric", "")))
        )
    from scripts.run_best_candidate_ablation import parse_prediction_horizons

    requested_horizons = parse_prediction_horizons(
        args.prediction_horizons,
        fallback=int(getattr(args, "prediction_length", 256)),
    )
    required = {
        (str(horizon), metric)
        for horizon in requested_horizons
        for metric in ("ade", "fde")
    }
    return {key for key, available in grouped.items() if required.issubset(available)}


def _load_exp1_controls() -> tuple[str, dict[tuple[int, str], dict[str, Any]]]:
    path = PROJECT_ROOT / "experiments" / "exp1_sota" / "results" / "partial_runs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = str(payload.get("active_config", {}).get("run_signature", "")).strip()
    if not signature:
        raise RuntimeError("Cannot reuse Exp1 controls: active run signature is missing.")
    records = {}
    for record in payload.get("runs", {}).values():
        if record.get("run_signature") != signature:
            continue
        records[(int(record["seed"]), str(record["model_name"]))] = record
    return signature, records


def _exp1_control_name(phase: str, model_key: str) -> str | None:
    if model_key == "baseline" and phase in {
        "phase1_structural",
        "phase3_attention_prior_isolation",
    }:
        return "Transformer (baseline)"
    return None


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return seeds or [42]


def _configure_exp2(
    exp2: Any,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    dataset_identity: dict[str, Any] | None = None,
) -> list[int]:
    from scripts.run_best_candidate_ablation import parse_prediction_horizons

    horizons = parse_prediction_horizons(args.prediction_horizons, fallback=int(args.prediction_length))
    seeds = _parse_seeds(args.seeds)
    exp2.BASE_RANDOM_SEEDS = seeds
    exp2.NUM_RUNS = len(seeds)
    exp2.RANDOM_SEEDS = seeds
    exp2.TRAIN_CONFIG["epochs"] = int(args.epochs)
    exp2.TRAIN_CONFIG["batch_size"] = int(args.batch_size)
    exp2.TRAIN_CONFIG["lr"] = float(candidate.get("learning_rate", args.learning_rate))
    exp2.TRAIN_CONFIG["weight_decay"] = float(args.weight_decay)
    exp2.TRAIN_CONFIG["warmup_epochs"] = int(candidate.get("warmup_epochs", args.warmup_epochs))
    exp2.TRAIN_CONFIG["early_stopping_patience"] = int(args.patience)
    exp2.TRAIN_CONFIG["gradient_clip_norm"] = float(candidate.get("gradient_clip_norm", args.gradient_clip_norm))
    exp2.TRAIN_CONFIG["use_mixed_precision"] = bool(args.amp)
    exp2.TRAIN_CONFIG["mixed_precision_dtype"] = str(args.amp_dtype)
    exp2.TRAIN_CONFIG["cache_physics_prior"] = bool(args.cache_physics_prior)
    exp2.TRAIN_CONFIG["physics_prior_cache_batch_size"] = int(
        args.physics_prior_cache_batch_size
    )
    exp2.TRAIN_CONFIG["dataloader_workers"] = int(args.workers)
    exp2.TRAIN_CONFIG["pin_memory"] = bool(args.workers > 0)
    sampling_interval_s = float(
        (dataset_identity or {}).get("sampling_interval_s", 1.0)
    )
    exp2.TRAIN_CONFIG["sampling_interval_s"] = sampling_interval_s
    exp2.train_config["sampling_interval_s"] = sampling_interval_s
    exp2.PHYSICS_DT = sampling_interval_s
    exp2.ABLATION_SUBSET_RATIO = float(args.subset_ratio)
    exp2.ABLATION_PRED_LEN = max(horizons)
    exp2.PREDICTION_HORIZONS = horizons
    exp2.TRAIN_SUPERVISION_PROTOCOL = str(args.train_supervision_protocol)
    exp2.EVAL_PROTOCOL = str(args.eval_protocol)
    exp2.EVAL_AR_SEED_MODE = str(args.eval_ar_seed_mode)
    exp2.train_config["label_len"] = int(args.label_len)
    return horizons


def _flatten_metric_rows(
    *,
    run_id: str,
    timestamp: str,
    seed: int,
    model_name: str,
    eval_results: dict[int, dict[str, float]],
    args: argparse.Namespace,
    candidate: dict[str, Any],
    checkpoint: Path,
    phase: str,
    source: str = "formal_ablation_training",
    source_run_signature: str = "",
    dataset_identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_identity = dataset_identity or _default_formal_dataset_identity()
    sampling_interval_s = float(dataset_identity.get("sampling_interval_s", 1.0))
    for horizon, metrics in sorted(eval_results.items()):
        for metric, value in sorted(metrics.items()):
            if not _is_scalar_metric(value):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "seed": seed,
                    "model_key": _model_key_for_name(model_name),
                    "model": model_name,
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
                    "candidate_id": candidate.get("candidate_id", ""),
                    "phase": phase,
                    "checkpoint": str(checkpoint),
                    "source": source,
                    "source_run_signature": source_run_signature,
                }
            )
    return rows


def run_formal_ablation_units(args: argparse.Namespace) -> dict[str, Any]:
    from utils.console import ensure_utf8_console

    ensure_utf8_console()

    from utils.formal_evidence import load_formal_config

    formal_config_path = getattr(
        args,
        "config",
        str(PROJECT_ROOT / "configs" / "formal_v3.json"),
    )
    formal_config = load_formal_config(formal_config_path)

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
        dataset_identity, force=True, namespace="exp2_ablation"
    )
    if (
        str(dataset_identity.get("dataset_protocol")) != str(formal_config["dataset"]["protocol"])
        or str(dataset_identity.get("dataset_sha256")) != str(formal_config["dataset"]["sha256"])
    ):
        raise RuntimeError(
            "Formal runner dataset identity does not match configs/formal_v3.json: "
            f"{dataset_identity.get('dataset_protocol')} / {dataset_identity.get('dataset_sha256')}"
        )

    import torch
    from experiments.mechanism_analysis import ablation_study as exp2

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

    evidence_tier = resolve_evidence_tier(bool(args.selection_only))
    custom_output_dirs = resolve_custom_output_dirs(args)
    if custom_output_dirs is None:
        formal_dir, formal_models_dir = get_formal_protocol_dirs(
            PROJECT_ROOT,
            "exp2_ablation",
            str(dataset_identity["dataset_protocol"]),
            evidence_tier,
        )
    else:
        formal_dir, formal_models_dir = custom_output_dirs
    metrics_csv = formal_dir / "formal_ablation_runs.csv"
    protocol_identity = build_ablation_protocol_identity(
        args, horizons, dataset_identity
    )
    already_completed = (
        completed_units(metrics_csv, args)
        if args.skip_existing and not args.selection_only
        else set()
    )
    exp1_signature = ""
    exp1_controls: dict[tuple[int, str], dict[str, Any]] = {}
    if args.selection_only and args.reuse_exp1_controls:
        raise ValueError(
            "--selection-only cannot reuse legacy Exp1 test records; use the "
            "frozen Main Results controls when assembling the final ablation table."
        )
    if args.reuse_exp1_controls:
        if not args.allow_legacy_diagnostic:
            raise ValueError(
                "--reuse-exp1-controls is legacy-only; pass "
                "--allow-legacy-diagnostic for an explicit diagnostic run."
            )
        exp1_signature, exp1_controls = _load_exp1_controls()

    device = exp2.TRAIN_CONFIG["device"]
    train_loader = val_loader = test_loader = None
    scaler_mean = scaler_scale = None
    source_scaler_mean = source_scaler_scale = None
    output_scaler_mean = output_scaler_scale = None
    requires_training = any(
        (args.phase, int(seed), _model_key_for_name(model_name)) not in already_completed
        and _exp1_control_name(args.phase, _model_key_for_name(model_name)) is None
        for seed in seeds
        for model_name in model_configs
    )
    if requires_training:
        exp2.set_random_seed(seeds[0])
        train_loader, val_loader, test_loader, x_scaler, y_scaler = (
            exp2.load_and_prepare_data(
                exp2.TRAIN_CONFIG["batch_size"],
                subset_ratio=float(args.subset_ratio),
                initial_load_seed=seeds[0],
            )
        )
        if train_loader is None:
            raise RuntimeError("data loading failed")
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

    completed: list[dict[str, Any]] = []
    for seed in seeds:
        for model_name, model_config in model_configs.items():
            model_key = _model_key_for_name(model_name)
            unit_key = (args.phase, int(seed), model_key)
            if args.selection_only and args.skip_existing:
                existing = find_existing_ablation_selection(
                    formal_dir,
                    phase=args.phase,
                    seed=seed,
                    model_key=model_key,
                    protocol_identity=protocol_identity,
                    model_config=model_config,
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
                    print(
                        "[formal-ablation] reuse exact selection record "
                        f"{existing_path}"
                    )
                    continue
            if unit_key in already_completed:
                print(
                    "[formal-ablation] skipping completed unit "
                    f"phase={args.phase} seed={seed} model={model_key}"
                )
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"formal_{args.phase}_seed{seed}_{model_key}_{timestamp}"
            print(f"[formal-ablation] run {run_id} horizons={horizons}")

            reused_name = (
                _exp1_control_name(args.phase, model_key)
                if args.reuse_exp1_controls
                else None
            )
            reused_record = (
                exp1_controls.get((int(seed), reused_name)) if reused_name else None
            )
            if reused_name and reused_record is None:
                raise RuntimeError(
                    f"Cannot reuse Exp1 control seed={seed}, model={reused_name}, "
                    f"signature={exp1_signature}."
                )
            if reused_record is not None:
                eval_results = {
                    int(horizon): metrics
                    for horizon, metrics in reused_record["results"].items()
                    if int(horizon) in horizons
                }
                checkpoint = Path(str(reused_record.get("checkpoint_path", "")))
                if not checkpoint.is_file():
                    raise RuntimeError(f"Reused Exp1 checkpoint is missing: {checkpoint}")
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
                    source="reused_exp1_control",
                    source_run_signature=exp1_signature,
                    dataset_identity=dataset_identity,
                )
                append_metric_rows(metrics_csv, rows)
                run_json = formal_dir / f"{run_id}.json"
                payload = {
                    "run_id": run_id,
                    "seed": seed,
                    "model": model_name,
                    "model_key": model_key,
                    "candidate": candidate,
                    "phase": args.phase,
                    "config": vars(args),
                    "horizons": horizons,
                    "eval_results": eval_results,
                    "history": reused_record.get("training_history", {}),
                    "checkpoint": str(checkpoint),
                    "metrics_csv": str(metrics_csv),
                    "source": "reused_exp1_control",
                    "source_run_signature": exp1_signature,
                    "dataset_identity": dataset_identity,
                    "protocol_identity": protocol_identity,
                    "scaler_paths": {
                        key: str(path) for key, path in scaler_paths.items()
                    },
                }
                run_json.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                completed.append(
                    {
                        "run_id": run_id,
                        "json": str(run_json),
                        "checkpoint": str(checkpoint),
                    }
                )
                already_completed.add(unit_key)
                print(f"[formal-ablation] reused signed Exp1 control -> {run_json}")
                continue

            if train_loader is None or val_loader is None or test_loader is None:
                raise RuntimeError("Training data were not initialized for a non-reused unit.")
            exp2.set_random_seed(seed)
            generator = exp2.build_torch_generator(seed)
            loader_kwargs = {
                "num_workers": train_loader.num_workers,
                "pin_memory": train_loader.pin_memory,
                "generator": generator,
            }
            if train_loader.num_workers > 0:
                loader_kwargs["persistent_workers"] = train_loader.persistent_workers
                loader_kwargs["prefetch_factor"] = train_loader.prefetch_factor
                loader_kwargs["worker_init_fn"] = exp2.seed_worker
            train_loader_model = exp2.DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                **loader_kwargs,
            )
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
            trained_model, history = exp2.train_model(
                model_name=model_name,
                model_config=model_config,
                train_loader=train_loader_model,
                val_loader=val_loader,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                model_override=model,
            )
            checkpoint = formal_models_dir / f"{run_id}.pth"
            torch.save(trained_model.state_dict(), checkpoint)
            checkpoint_sha256 = sha256_file(checkpoint)
            if args.selection_only:
                eval_results: dict[int, dict[str, float]] = {}
            else:
                eval_results = exp2.run_evaluation(
                    trained_model,
                    test_loader,
                    scaler_mean,
                    scaler_scale,
                    device,
                    horizons,
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
                    dataset_identity=dataset_identity,
                )
                metrics_csv = append_metric_rows(metrics_csv, rows)
            run_json = formal_dir / f"{run_id}.json"
            record_model_key = _model_key_for_name(model_name)
            record_model_config = dict(model_config)
            if record_model_key in {"full", "proposed"}:
                record_model_config.update(FINAL_MODEL_FLAGS)
            payload = {
                "schema_version": 2,
                "formal_config_sha256": formal_config.config_sha256,
                "run_id": run_id,
                "timestamp": timestamp,
                "seed": seed,
                "model": model_name,
                "model_key": _model_key_for_name(model_name),
                "model_config": record_model_config,
                "candidate": candidate,
                "phase": args.phase,
                "config": vars(args),
                "horizons": horizons,
                "eval_results": eval_results,
                "history": history,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "metrics_csv": None if args.selection_only else str(metrics_csv),
                "evidence_tier": evidence_tier,
                "test_evaluation_performed": not bool(args.selection_only),
                "protocol_identity": protocol_identity,
                "dataset_identity": dataset_identity,
                "scaler_paths": {key: str(path) for key, path in scaler_paths.items()},
            }
            run_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            completed.append({"run_id": run_id, "json": str(run_json), "checkpoint": str(checkpoint)})
            already_completed.add(unit_key)
            print(f"[formal-ablation] wrote {run_json}")
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
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Formal dataset artifact. Defaults to the audited multi-regime v2.1 dataset; "
            "HGV_DATASET_PATH remains an explicit override."
        ),
    )
    parser.add_argument("--phase", type=str, default="phase4_final_mechanism_controls")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--models", type=str, default="all")
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
    parser.add_argument("--workers", type=int, default=0)
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
        "--selection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/select on validation only and isolate outputs from test evidence.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip phase/seed/model units already complete under the same protocol.",
    )
    parser.add_argument(
        "--reuse-exp1-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy compatibility only: reuse signed Exp1 test controls. Disabled "
            "for new validation-selection runs."
        ),
    )
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="Allow the legacy partial-run control adapter; never paper-facing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run_formal_ablation_units(parse_args(argv))
    print("[formal-ablation] completed:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
