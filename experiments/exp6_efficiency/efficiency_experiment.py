#!/usr/bin/env python3
"""Experiment 6: efficiency and performance trade-off benchmark.

The benchmark is intentionally lightweight: it instantiates each model with the
project's registered constructors, measures forward latency with the same
source-context decoder protocol used elsewhere, and joins those efficiency
numbers with the formal Exp1 performance references used by the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import joblib
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import HGVConfig, create_registered_model
from data_generation.data_paths import get_input_scaler_path, get_output_scaler_path
from utils.inference_protocol import predict_by_eval_protocol
from utils.final_plgaformer import final_plgaformer_kwargs
from utils.formal_evidence import (
    DISPLAY_NAMES,
    load_evidence_bundle,
    load_formal_config,
    normalize_run_record,
    validate_evidence_bundle,
)
from utils.repro import set_global_seed


DEFAULT_EXP1_RESULTS = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "main_run_set_manifest.json"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "formal_v3.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run efficiency benchmark")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate the Main bundle and print the benchmark contract.")
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="Allow historical non-bundle performance input; never paper-facing.",
    )
    parser.add_argument("--quick", action="store_true", help="Use fewer repeats and smaller batch")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--performance-results", type=Path, default=DEFAULT_EXP1_RESULTS)
    parser.add_argument(
        "--models",
        default="PLGAFormer,Transformer,AF-CILN,iTransformer,PatchTST,DLinear,PIT,Spherical kinematics,Rotating-Earth 3-DOF",
        help="Comma-separated model names",
    )
    return parser.parse_args(argv)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_performance_reference(
    path: Path,
    config=None,
    *,
    allow_legacy_diagnostic: bool = False,
) -> tuple[dict, str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Exp1 result file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    aliases = {
        "PLGAFormer (proposed)": "PLGAFormer",
        "Transformer (baseline)": "Transformer",
    }
    reference = {}
    audit = {"schema": "legacy_result_payload"}
    if payload.get("bundle_kind") == "main":
        formal_config = config or load_formal_config(DEFAULT_CONFIG)
        verification = validate_evidence_bundle(path, formal_config, expected_kind="main")
        if not verification["passed"]:
            codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
            raise RuntimeError(
                "Efficiency performance reference requires an eligible formal-v3 Main bundle: "
                + ", ".join(codes)
            )
        bundle = verification["bundle"]
        grouped: dict[tuple[str, int, str], list[float]] = {}
        for entry in bundle.get("source_records", []):
            record = normalize_run_record(entry["source_path"])
            display_name = DISPLAY_NAMES.get(str(record.get("model_key")), str(record.get("model", "")))
            for horizon in (32, 64, 128, 256):
                item = record.get("eval_results", {}).get(str(horizon), {})
                for metric, aliases_for_metric in {
                    "ade": ("ade", "trajectory_window_ade"),
                    "fde": ("fde", "trajectory_window_fde"),
                    "rmse_cart_m": ("rmse_cart_m",),
                }.items():
                    value = next((item.get(alias) for alias in aliases_for_metric if item.get(alias) is not None), None)
                    if value is None:
                        raise RuntimeError(f"Formal-v3 Main bundle lacks {metric} at horizon {horizon} for {display_name}.")
                    grouped.setdefault((display_name, horizon, metric), []).append(float(value))
        for (display_name, horizon, metric), values in grouped.items():
            reference.setdefault(display_name, {})[f"{metric.upper()}_{horizon}"] = float(np.mean(values))
        audit = {
            "schema": "formal_v3_main_results_bundle",
            "bundle_id": bundle.get("bundle_id"),
            "config_sha256": bundle.get("config_sha256"),
            "dataset_sha256": bundle.get("dataset_sha256"),
            "seeds_by_model": {
                name: sorted(
                    int(entry.get("seed", -1))
                    for entry in bundle.get("source_records", [])
                    if DISPLAY_NAMES.get(str(entry.get("model_key")), str(entry.get("model", ""))) == name
                )
                for name in reference
            },
        }
    elif not allow_legacy_diagnostic:
        raise RuntimeError(
            "Efficiency performance input must be an eligible formal-v3 Main bundle; "
            "pass --allow-legacy-diagnostic only for historical diagnostics."
        )
    elif "seed_statistics" in payload:
        for model_name, horizon_results in payload["seed_statistics"].items():
            display_name = aliases.get(model_name, model_name)
            reference[display_name] = {}
            for horizon, metrics in horizon_results.items():
                for metric in ("ade", "fde"):
                    reference[display_name][f"{metric.upper()}_{horizon}"] = float(
                        metrics[metric]["mean_m"]
                    )
        audit = {
            "schema": "taes_final_main_summary",
            "run_signature": payload.get("run_signature"),
            "base_exp1_signature": payload.get("base_exp1_signature"),
        }
    elif "runs" in payload and "active_config" in payload:
        active = payload.get("active_config", {})
        signature = os.getenv("HGV_EXP1_RUN_SIGNATURE", "").strip() or active.get("run_signature")
        if not signature:
            raise RuntimeError("No active exp1 signature is available for efficiency joins.")
        grouped = {}
        seeds = {}
        for record in payload.get("runs", {}).values():
            if record.get("run_signature") != signature:
                continue
            model_name = aliases.get(record.get("model_name"), record.get("model_name"))
            if not model_name or not record.get("results"):
                continue
            seeds.setdefault(model_name, set()).add(int(record.get("seed", -1)))
            for horizon, metrics in record["results"].items():
                for metric in ("ade", "fde", "rmse_cart_m", "mse"):
                    if metric in metrics:
                        grouped.setdefault((model_name, int(horizon), metric), []).append(
                            float(metrics[metric])
                        )
        for (model_name, horizon, metric), values in grouped.items():
            reference.setdefault(model_name, {})[f"{metric.upper()}_{horizon}"] = float(np.mean(values))
        audit = {
            "schema": "exp1_partial_runs",
            "run_signature": signature,
            "seeds_by_model": {name: sorted(values) for name, values in seeds.items()},
        }
    else:
        for model_name, horizon_results in payload.items():
            display_name = aliases.get(model_name, model_name)
            reference[display_name] = {}
            for horizon, metrics in horizon_results.items():
                for metric in ("ade", "fde", "rmse_cart_m", "mse"):
                    if metric in metrics:
                        reference[display_name][f"{metric.upper()}_{horizon}"] = float(metrics[metric])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return reference, digest, audit


def build_physics_kwargs() -> dict:
    input_scaler = joblib.load(get_input_scaler_path())
    output_scaler = joblib.load(get_output_scaler_path())
    return {
        "input_scaler_mean": np.concatenate([output_scaler.mean_, input_scaler.mean_[3:]]).astype(np.float32),
        "input_scaler_scale": np.concatenate([output_scaler.scale_, input_scaler.scale_[3:]]).astype(np.float32),
        "output_scaler_mean": output_scaler.mean_.astype(np.float32),
        "output_scaler_scale": output_scaler.scale_.astype(np.float32),
        "sampling_interval_s": float(HGVConfig.get_train_config().get("sampling_interval_s", 1.0)),
        "require_physical_scaler": True,
    }


def _plgaformer_kwargs() -> dict:
    cfg = HGVConfig.get_model_config("plgaformer")
    allowed = {
        "input_dim",
        "d_model",
        "nhead",
        "output_dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "dropout",
    }
    return {k: cfg[k] for k in allowed if k in cfg}


def build_model(
    model_name: str,
    device: torch.device,
    seq_len: int,
    pred_len: int,
    physics_kwargs: dict | None = None,
) -> torch.nn.Module:
    normalized = model_name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "spherical_kinematics": "kinematic",
        "rotating_earth_3_dof": "rotating_3dof",
        "af_ciln": "af_ciln",
    }
    model_type = aliases.get(normalized, normalized)
    needs_scalers = model_type in {"plgaformer", "kinematic", "rotating_3dof", "af_ciln"}
    model_kwargs = dict(physics_kwargs or {}) if needs_scalers else None
    if model_type == "plgaformer":
        model_kwargs.update(final_plgaformer_kwargs())
    return create_registered_model(
        model_type=model_type,
        input_dim=6,
        device=device,
        plgaformer_kwargs=model_kwargs,
    )


def protocol_predict(model: torch.nn.Module, x: torch.Tensor, y_ref: torch.Tensor, device: torch.device) -> torch.Tensor:
    return predict_by_eval_protocol(
        model=model,
        x=x,
        y_true_scaled=y_ref,
        pred_length=y_ref.size(1),
        device=device,
        eval_protocol="source_context_decoder",
        eval_ar_seed_mode="zero",
        label_len=min(48, x.size(1)),
    )


def measure_latency(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    pred_len: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    x = torch.zeros(batch_size, seq_len, 6, device=device)
    y_ref = torch.zeros(batch_size, pred_len, 3, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            protocol_predict(model, x, y_ref, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            protocol_predict(model, x, y_ref, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000 / max(1, repeats)


def measure_profiled_flops(
    model: torch.nn.Module,
    *,
    seq_len: int,
    pred_len: int,
    device: torch.device,
) -> float | None:
    x = torch.zeros(1, seq_len, 6, device=device)
    y_ref = torch.zeros(1, pred_len, 3, device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    try:
        with profile(activities=activities, with_flops=True) as prof:
            with torch.no_grad():
                protocol_predict(model, x, y_ref, device)
        total_flops = sum(float(event.flops or 0.0) for event in prof.key_averages())
        return total_flops / 1e6 if total_flops > 0.0 else None
    except Exception:
        return None


def measure_peak_memory_mb(model, *, batch_size, seq_len, pred_len, device):
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.zeros(batch_size, seq_len, 6, device=device)
    y_ref = torch.zeros(batch_size, pred_len, 3, device=device)
    with torch.no_grad():
        protocol_predict(model, x, y_ref, device)
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated(device) / 1e6)


def run_benchmark(args: argparse.Namespace, config=None) -> tuple[list[dict], dict]:
    config = config or load_formal_config(args.config)
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = int(config["task"]["seq_len"])
    pred_len = int(config["task"]["pred_len"])
    batch_size = args.batch_size if args.batch_size is not None else (8 if args.quick else 64)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 10)
    repeats = args.repeats if args.repeats is not None else (2 if args.quick else 50)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    performance_reference, performance_sha256, performance_audit = load_performance_reference(
        args.performance_results,
        config,
        allow_legacy_diagnostic=bool(args.allow_legacy_diagnostic),
    )
    physics_kwargs = build_physics_kwargs()

    rows = []
    for model_name in model_names:
        print(f"[exp6] benchmarking {model_name}", flush=True)
        model = build_model(
            model_name,
            device=device,
            seq_len=seq_len,
            pred_len=pred_len,
            physics_kwargs=physics_kwargs,
        )
        params = count_params(model)
        flops_m = measure_profiled_flops(model, seq_len=seq_len, pred_len=pred_len, device=device)
        latency_batch1_ms = measure_latency(
            model,
            batch_size=1,
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        latency_batch_ms = measure_latency(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        peak_memory_mb = measure_peak_memory_mb(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
        )
        ref = performance_reference.get(model_name, {})
        ade_256 = ref.get("ADE_256")
        rows.append(
            {
                "model": model_name,
                "params": params,
                "params_k": params / 1000,
                "flops_mflops": flops_m,
                "memory_mb": params * 4 / 1e6,
                "parameter_memory_mb": params * 4 / 1e6,
                "peak_inference_memory_mb": peak_memory_mb,
                "latency_batch1_ms": latency_batch1_ms,
                "throughput_batch_size": batch_size,
                "latency_batch_ms": latency_batch_ms,
                "throughput_trajectories_s": 1000.0 * batch_size / latency_batch_ms,
                "ADE_256_m": ade_256,
                "FDE_256_m": ref.get("FDE_256"),
                "RMSE_256_m": ref.get("RMSE_CART_M_256"),
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "experiment": str(config["studies"]["efficiency"]["study_id"]),
        "study_id": str(config["studies"]["efficiency"]["study_id"]),
        "formal_config_sha256": config.config_sha256,
        "dataset_protocol": config["dataset"]["protocol"],
        "dataset_sha256": config["dataset"]["sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inference_dtype": "float32",
        "seq_len": seq_len,
        "pred_len": pred_len,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "quick": bool(args.quick),
        "formal": not bool(args.quick) and batch_size == 64 and warmup >= 10 and repeats >= 50,
        "protocol": "source_context_decoder",
        "latency_protocol": "batch-1 end-to-end forecast latency; batch-64 throughput reported separately",
        "timing_synchronization": "torch.cuda.synchronize before and after timed repeats" if device.type == "cuda" else "not applicable",
        "performance_reference": str(args.performance_results.resolve()),
        "performance_reference_sha256": performance_sha256,
        "performance_reference_audit": performance_audit,
        "checkpoint_source": "formal Main bundle records; efficiency benchmark does not retrain",
        "flops_method": "torch.profiler operator FLOPs for one batch-size-1 forward; unsupported operators omitted",
    }
    return rows, metadata


def save_outputs(out_dir: Path, rows: list[dict], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "efficiency_results.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": rows}, f, indent=2)
    with open(out_dir / "efficiency_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(out_dir: Path, rows: list[dict]) -> None:
    colors = ["#B33A3A", "#2F5597", "#E28A25", "#3B7A57", "#6D5A8D", "#8A5A44", "#C07A9E", "#53777A"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, x_key, x_label in (
        (axes[0], "params_k", "Parameters (K)"),
        (axes[1], "latency_batch1_ms", "Batch-1 latency (ms)"),
    ):
        for i, row in enumerate(rows):
            if row.get("ADE_256_m") is None:
                continue
            ax.scatter(
                row[x_key],
                row["ADE_256_m"] / 1000.0,
                s=140,
                color=colors[i % len(colors)],
                edgecolors="black",
                linewidth=0.5,
                label=row["model"],
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel("ADE at 256 s (km)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Exp 6: model size vs prediction accuracy", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_efficiency_pareto.png", dpi=300)
    plt.close(fig)

    names = [row["model"] for row in rows]
    bar_colors = [colors[i % len(colors)] for i in range(len(rows))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(names, [row["params_k"] for row in rows], color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Parameters (K)")
    axes[0].set_title("Model size")
    axes[1].bar(names, [row["flops_mflops"] or 0.0 for row in rows], color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Approx. MFLOPs")
    axes[1].set_title("Compute cost")
    axes[2].bar(names, [row["latency_batch1_ms"] for row in rows], color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Batch-1 end-to-end latency")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Exp 6: efficiency benchmark", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_efficiency_bars.png", dpi=300)
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    print("\nEFFICIENCY COMPARISON")
    print("-" * 104)
    print(f"{'Model':<22} {'Params(K)':>10} {'MFLOPs':>10} {'B1 ms':>10} {'B64 traj/s':>12} {'ADE@256m':>12} {'FDE@256m':>12}")
    for row in rows:
        print(
            f"{row['model']:<22} {row['params_k']:>10.1f} {(row['flops_mflops'] or 0):>10.1f} "
            f"{row['latency_batch1_ms']:>10.2f} {row['throughput_trajectories_s']:>12.1f} "
            f"{(row.get('ADE_256_m') or 0):>12.1f} {(row.get('FDE_256_m') or 0):>12.1f}"
        )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    config = load_formal_config(args.config)
    if args.dry_run:
        verification = validate_evidence_bundle(
            args.performance_results, config, expected_kind="main"
        )
        report = {
            "command": "formal efficiency --dry-run",
            "study_id": config["studies"]["efficiency"]["study_id"],
            "paper_eligible_input_bundle": bool(verification["passed"]),
            "config_sha256": config.config_sha256,
            "dataset_sha256": config["dataset"]["sha256"],
            "blockers": verification.get("blockers", []),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report
    rows, metadata = run_benchmark(args, config)
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "results"
        if args.quick:
            out_dir = out_dir / "quick"
    save_outputs(out_dir, rows, metadata)
    generate_figures(out_dir, rows)
    print_table(rows)
    print(f"Saved exp6 outputs to {out_dir}")
    return {"metadata": metadata, "results": rows}


if __name__ == "__main__":
    main()
