#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, repeatable PLGAFormer candidate search on trajectory-level data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    learning_rate: float
    dropout: float
    alpha: float
    warmup_epochs: int
    gradient_clip_norm: float
    innovations: dict[str, bool]

    def model_config(self) -> dict[str, Any]:
        return {
            "model_type": "plgaformer",
            "description": (
                "Full PLGAFormer candidate "
                f"(lr={self.learning_rate:g}, dropout={self.dropout:.2f}, alpha={self.alpha:g})"
            ),
            "physics_loss_weight": float(self.alpha),
            "physics_alpha": float(self.alpha),
            "warmup_epochs": int(self.warmup_epochs),
            "dropout": float(self.dropout),
            "innovations": dict(self.innovations),
        }


def build_candidate_grid(max_candidates: int | None = None) -> list[CandidateConfig]:
    """Build a deterministic candidate grid for A+B+C only."""
    learning_rates = [1e-3, 7e-4, 5e-4]
    dropouts = [0.10, 0.05, 0.15]
    alphas = [0.0, 1e-4, 5e-4]
    candidates: list[CandidateConfig] = []
    full_innovations = {
        "use_sparse_attention": True,
        "use_physics_corrector": True,
        "use_multi_head_output": True,
    }
    for lr in learning_rates:
        for dropout in dropouts:
            for alpha in alphas:
                idx = len(candidates) + 1
                candidates.append(
                    CandidateConfig(
                        candidate_id=f"c{idx:02d}_lr{lr:g}_do{dropout:.2f}_a{alpha:g}",
                        learning_rate=lr,
                        dropout=dropout,
                        alpha=alpha,
                        warmup_epochs=2,
                        gradient_clip_norm=0.7,
                        innovations=full_innovations,
                    )
                )
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates
    return candidates


def select_candidates_by_id(
    candidates: list[CandidateConfig],
    candidate_ids: list[str],
) -> list[CandidateConfig]:
    """Select candidates by exact id while preserving the requested order."""
    if not candidate_ids:
        return candidates
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"Unknown candidate id(s): {missing}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def candidate_seed_for_id(base_seed: int, candidate_id: str) -> int:
    """Derive a stable per-candidate seed independent of search order."""
    from utils.repro import stable_seed_from_name

    return stable_seed_from_name(base_seed, candidate_id)


def model_seed_for_id(base_seed: int, model_id: str, seed_mode: str = "candidate") -> int:
    """Resolve the training seed for a candidate-search model."""
    if seed_mode == "shared":
        return int(base_seed)
    if seed_mode == "candidate":
        return candidate_seed_for_id(base_seed, model_id)
    raise ValueError(f"Unsupported seed_mode: {seed_mode}")


def _as_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def rank_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank successful PLGAFormer candidate rows by MSE, then improvement."""
    candidates = [
        dict(row)
        for row in rows
        if row.get("candidate_id") != "baseline" and row.get("status", "ok") == "ok"
    ]
    candidates.sort(
        key=lambda row: (
            _as_float(row.get("mse")),
            -_as_float(row.get("baseline_improvement_pct"), default=-math.inf),
            str(row.get("candidate_id", "")),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    return candidates


def write_candidate_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    run_config: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Path]:
    """Write timestamped and latest JSON/CSV outputs plus best config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    ranked = rank_candidate_rows(rows)
    best = ranked[0] if ranked else None

    payload = {
        "run_config": run_config,
        "ranked_candidates": ranked,
        "rows": rows,
        "best_candidate": best,
    }

    timestamped_json = output_dir / f"candidate_search_{timestamp}.json"
    latest_json = output_dir / "latest_candidate_search.json"
    timestamped_csv = output_dir / f"candidate_search_{timestamp}.csv"
    latest_csv = output_dir / "latest_candidate_search.csv"
    best_json = output_dir / "best_candidate_config.json"

    with timestamped_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    shutil.copyfile(timestamped_json, latest_json)

    fieldnames = sorted({key for row in rows for key in row.keys()} | {"rank"})
    with timestamped_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        ranked_by_id = {row["candidate_id"]: row for row in ranked}
        for row in rows:
            out = dict(row)
            if row.get("candidate_id") in ranked_by_id:
                out["rank"] = ranked_by_id[row["candidate_id"]]["rank"]
            writer.writerow(out)
    shutil.copyfile(timestamped_csv, latest_csv)

    with best_json.open("w", encoding="utf-8") as f:
        json.dump({"run_config": run_config, "best_candidate": best}, f, indent=2, ensure_ascii=False)

    return {
        "timestamped_json": timestamped_json,
        "latest_json": latest_json,
        "timestamped_csv": timestamped_csv,
        "latest_csv": latest_csv,
        "best_json": best_json,
    }


def _configure_exp2_module(exp2: Any, args: argparse.Namespace) -> None:
    exp2.TRAIN_CONFIG["epochs"] = int(args.epochs)
    exp2.TRAIN_CONFIG["batch_size"] = int(args.batch_size)
    exp2.TRAIN_CONFIG["lr"] = float(args.learning_rate)
    exp2.TRAIN_CONFIG["weight_decay"] = float(args.weight_decay)
    exp2.TRAIN_CONFIG["warmup_epochs"] = int(args.warmup_epochs)
    exp2.TRAIN_CONFIG["early_stopping_patience"] = int(args.patience)
    exp2.TRAIN_CONFIG["gradient_clip_norm"] = float(args.gradient_clip_norm)
    exp2.TRAIN_CONFIG["use_mixed_precision"] = bool(args.amp)
    exp2.TRAIN_CONFIG["dataloader_workers"] = int(args.workers)
    exp2.TRAIN_CONFIG["pin_memory"] = bool(args.workers > 0)
    exp2.ABLATION_SUBSET_RATIO = float(args.subset_ratio)
    exp2.ABLATION_PRED_LEN = int(args.prediction_length)
    exp2.PREDICTION_HORIZONS = [int(args.prediction_length)]
    exp2.TRAIN_SUPERVISION_PROTOCOL = str(args.train_supervision_protocol)
    exp2.EVAL_PROTOCOL = str(args.eval_protocol)
    exp2.EVAL_AR_SEED_MODE = str(args.eval_ar_seed_mode)
    exp2.train_config["label_len"] = int(args.label_len)


def _train_and_eval(
    exp2: Any,
    model_name: str,
    model_config: dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    scaler_mean: Any,
    scaler_scale: Any,
    device: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    model = exp2.create_model(
        model_type=model_config["model_type"],
        input_dim=6,
        device=device,
        innovations=model_config.get("innovations"),
        model_config_override=model_config,
    )
    trained_model, history = exp2.train_model(
        model_name=model_name,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        model_override=model,
    )
    metrics_by_horizon = exp2.run_evaluation(
        trained_model,
        test_loader,
        scaler_mean,
        scaler_scale,
        device,
        exp2.PREDICTION_HORIZONS,
    )
    horizon = int(exp2.PREDICTION_HORIZONS[0])
    return metrics_by_horizon[horizon], history


def _rebuild_train_loader(exp2: Any, train_loader: Any, seed: int) -> Any:
    """Rebuild the shuffled train loader with a fresh deterministic generator."""
    loader_kwargs = {
        "num_workers": train_loader.num_workers,
        "pin_memory": train_loader.pin_memory,
        "generator": exp2.build_torch_generator(seed),
    }
    if train_loader.num_workers > 0:
        loader_kwargs["persistent_workers"] = train_loader.persistent_workers
        loader_kwargs["prefetch_factor"] = train_loader.prefetch_factor
        loader_kwargs["worker_init_fn"] = exp2.seed_worker
    return exp2.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        **loader_kwargs,
    )


def run_search(args: argparse.Namespace) -> dict[str, Path]:
    from utils.console import ensure_utf8_console

    ensure_utf8_console()

    import torch
    from experiments.exp2_ablation import ablation_study as exp2

    _configure_exp2_module(exp2, args)
    exp2.set_random_seed(int(args.seed))

    print("[candidate-search] loading trajectory-level data")
    train_loader, val_loader, test_loader, _x_scaler, y_scaler = exp2.load_and_prepare_data(
        int(args.batch_size),
        subset_ratio=float(args.subset_ratio),
        initial_load_seed=int(args.seed),
    )
    if train_loader is None:
        raise RuntimeError("data loading failed")

    device = exp2.TRAIN_CONFIG["device"]
    scaler_mean = torch.from_numpy(y_scaler.mean_.astype("float32")).to(device)
    scaler_scale = torch.from_numpy(y_scaler.scale_.astype("float32")).to(device)

    baseline_cfg = {
        "model_type": "baseline",
        "description": "Standard Transformer baseline for candidate search",
        "physics_loss_weight": 0.0,
        "innovations": None,
    }
    print("[candidate-search] training baseline")
    baseline_seed = model_seed_for_id(int(args.seed), "baseline", str(args.seed_mode))
    exp2.set_random_seed(baseline_seed)
    baseline_train_loader = _rebuild_train_loader(exp2, train_loader, baseline_seed)
    baseline_metrics, baseline_history = _train_and_eval(
        exp2=exp2,
        model_name="Transformer (baseline)",
        model_config=baseline_cfg,
        train_loader=baseline_train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
    )
    baseline_mse = _as_float(baseline_metrics.get("mse"))
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": "baseline",
            "model": "Transformer (baseline)",
            "status": "ok",
            "mse": baseline_metrics.get("mse"),
            "mae": baseline_metrics.get("mae"),
            "rmse": baseline_metrics.get("rmse"),
            "best_val_loss": baseline_history.get("best_val_loss"),
            "baseline_improvement_pct": 0.0,
            "candidate_seed": baseline_seed,
        }
    ]

    requested_candidate_ids = [
        item.strip()
        for item in str(args.candidate_ids or "").split(",")
        if item.strip()
    ]
    if requested_candidate_ids:
        candidates = select_candidates_by_id(build_candidate_grid(max_candidates=None), requested_candidate_ids)
    else:
        candidates = build_candidate_grid(max_candidates=args.max_candidates)
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[candidate-search] candidate {idx}/{len(candidates)}: {candidate.candidate_id}")
        exp2.TRAIN_CONFIG["lr"] = float(candidate.learning_rate)
        exp2.TRAIN_CONFIG["gradient_clip_norm"] = float(candidate.gradient_clip_norm)
        candidate_seed = model_seed_for_id(int(args.seed), candidate.candidate_id, str(args.seed_mode))
        exp2.set_random_seed(candidate_seed)
        candidate_train_loader = _rebuild_train_loader(exp2, train_loader, candidate_seed)
        try:
            metrics, history = _train_and_eval(
                exp2=exp2,
                model_name=f"PLGAFormer {candidate.candidate_id}",
                model_config=candidate.model_config(),
                train_loader=candidate_train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                device=device,
            )
            mse = _as_float(metrics.get("mse"))
            improvement = ((baseline_mse - mse) / baseline_mse * 100.0) if baseline_mse > 0 else math.nan
            row = {
                **asdict(candidate),
                "model": "PLGAFormer (A+B+C)",
                "status": "ok",
                "mse": metrics.get("mse"),
                "mae": metrics.get("mae"),
                "rmse": metrics.get("rmse"),
                "best_val_loss": history.get("best_val_loss"),
                "baseline_mse": baseline_mse,
                "baseline_improvement_pct": improvement,
                "candidate_seed": candidate_seed,
            }
        except Exception as exc:
            row = {
                **asdict(candidate),
                "model": "PLGAFormer (A+B+C)",
                "status": "failed",
                "error": str(exc),
                "mse": math.inf,
                "baseline_mse": baseline_mse,
                "baseline_improvement_pct": -math.inf,
                "candidate_seed": candidate_seed,
            }
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_config = {
        "subset_ratio": float(args.subset_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "prediction_length": int(args.prediction_length),
        "max_candidates": args.max_candidates,
        "candidate_ids": requested_candidate_ids,
        "seed_mode": str(args.seed_mode),
        "train_supervision_protocol": str(args.train_supervision_protocol),
        "eval_protocol": str(args.eval_protocol),
        "eval_ar_seed_mode": str(args.eval_ar_seed_mode),
        "label_len": int(args.label_len),
        "device": str(device),
    }
    output_dir = PROJECT_ROOT / "experiments" / "exp2_ablation" / "results"
    paths = write_candidate_outputs(rows=rows, output_dir=output_dir, run_config=run_config)
    best = rank_candidate_rows(rows)[0] if rank_candidate_rows(rows) else None
    if best:
        print(
            "[candidate-search] best "
            f"{best['candidate_id']} mse={float(best['mse']):.6f} "
            f"improvement={float(best['baseline_improvement_pct']):+.2f}%"
        )
    print(f"[candidate-search] wrote {paths['latest_json']}")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--candidate-ids", type=str, default="")
    parser.add_argument("--seed-mode", choices=["candidate", "shared"], default="candidate")
    parser.add_argument("--train-supervision-protocol", type=str, default="source_context_pred_window")
    parser.add_argument("--eval-protocol", type=str, default="source_context_decoder")
    parser.add_argument("--eval-ar-seed-mode", type=str, default="zero")
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_search(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
