#!/usr/bin/env python3
"""Experiment 7: extended ablation for TRANS-ready supplementary evidence.

The experiment covers layer depth, physics-loss sensitivity, and input-length
impact. It uses the same source-context decoder protocol as the formal SOTA
experiments and writes JSON/CSV/figure artifacts under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_paths import (
    get_dataset_npz_path,
    get_input_scaler_path,
    get_output_scaler_path,
)
from models import HGVConfig, HGVPhysicsLoss, PLGAFormerTransformer
from utils.inference_protocol import predict_by_eval_protocol
from utils.repro import set_global_seed
from utils.seq2seq_protocol import align_source_position_scale
from utils.train_protocol import build_adamw_optimizer, build_warmup_cosine_scheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extended ablation experiment")
    parser.add_argument("--quick", action="store_true", help="Use fewer samples and epochs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args(argv)


def model_kwargs(overrides: dict | None = None, physics_kwargs: dict | None = None) -> dict:
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
    kwargs = {k: cfg[k] for k in allowed if k in cfg}
    kwargs.update({
        "use_sparse_attention": False,
        "use_physics_corrector": False,
        "use_multi_head_output": True,
        "use_adaptive_fusion": True,
    })
    if physics_kwargs:
        kwargs.update(physics_kwargs)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def predict(model, x: torch.Tensor, y: torch.Tensor, device) -> torch.Tensor:
    return predict_by_eval_protocol(
        model=model,
        x=x,
        y_true_scaled=y,
        pred_length=y.size(1),
        device=device,
        eval_protocol="source_context_decoder",
        eval_ar_seed_mode="zero",
        label_len=min(48, x.size(1)),
    )


def subset(arr: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None:
        return arr
    return arr[: min(max_samples, arr.shape[0])]


def build_base_arrays(max_samples: int | None = None):
    data = np.load(get_dataset_npz_path(), allow_pickle=True)
    input_scaler = joblib.load(get_input_scaler_path())
    output_scaler = joblib.load(get_output_scaler_path())

    def scale_x(values):
        shape = values.shape
        scaled = input_scaler.transform(values.reshape(-1, shape[-1])).reshape(shape)
        return align_source_position_scale(
            scaled,
            input_mean=input_scaler.mean_,
            input_scale=input_scaler.scale_,
            output_mean=output_scaler.mean_,
            output_scale=output_scaler.scale_,
            output_dim=3,
        ).astype(np.float32, copy=False)

    def scale_y(values):
        shape = values.shape
        return output_scaler.transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    physics_kwargs = {
        "input_scaler_mean": np.concatenate([output_scaler.mean_, input_scaler.mean_[3:]]).astype(np.float32),
        "input_scaler_scale": np.concatenate([output_scaler.scale_, input_scaler.scale_[3:]]).astype(np.float32),
        "output_scaler_mean": output_scaler.mean_.astype(np.float32),
        "output_scaler_scale": output_scaler.scale_.astype(np.float32),
        "sampling_interval_s": float(np.asarray(data.get("sampling_interval_s", 1.0)).reshape(-1)[0]),
        "require_physical_scaler": True,
    }
    return (
        subset(scale_x(data["X_train"]), max_samples),
        subset(scale_y(data["y_train"]), max_samples),
        subset(scale_x(data["X_val"]), max_samples),
        subset(scale_y(data["y_val"]), max_samples),
        physics_kwargs,
    )


def make_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def quick_train(
    model: PLGAFormerTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    epochs: int,
    alpha: float | None = None,
    acceleration_weight: float | None = None,
    kinematic_prior_max_scale: float | None = None,
) -> tuple[float, int]:
    model = model.to(device)
    if kinematic_prior_max_scale is not None:
        if not hasattr(model, "kinematic_prior_max_scale"):
            raise ValueError("Selected model has no kinematic prior scale.")
        model.kinematic_prior_max_scale = float(kinematic_prior_max_scale)

    optimizer = build_adamw_optimizer(model, learning_rate=1e-3, weight_decay=5e-5)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=min(3, epochs), total_epochs=epochs)
    criterion = (
        HGVPhysicsLoss(
            alpha=alpha,
            acceleration_weight=acceleration_weight,
            scaler_mean=model.output_scaler_mean.detach().cpu().numpy(),
            scaler_std=model.output_scaler_scale.detach().cpu().numpy(),
        )
        if alpha is not None or acceleration_weight is not None
        else torch.nn.MSELoss()
    )
    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = predict(model, x, y, device)
            if isinstance(criterion, HGVPhysicsLoss):
                loss = criterion(pred, y, input_data=x, include_mse=True)
            else:
                loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = predict(model, x, y, device)
                val_loss += torch.nn.functional.mse_loss(pred, y).item()
        val_loss /= max(1, len(val_loader))
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 8:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(best_val), int(n_params)


def run_layer_ablation(train_loader, val_loader, device, epochs: int, physics_kwargs: dict) -> list[dict]:
    results = []
    for n_layers in [1, 2, 3, 4, 5, 6]:
        model = PLGAFormerTransformer(**model_kwargs({"num_encoder_layers": n_layers}, physics_kwargs))
        val_loss, n_params = quick_train(model, train_loader, val_loader, device, epochs=epochs)
        results.append({"layers": n_layers, "val_mse": val_loss, "params": n_params})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_physics_sensitivity(train_loader, val_loader, device, epochs: int, physics_kwargs: dict) -> dict[str, list[dict]]:
    results = {"alpha": [], "acceleration_weight": [], "kinematic_prior_max_scale": []}
    base = HGVConfig.get_physics_config()

    for alpha in [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        model = PLGAFormerTransformer(**model_kwargs(physics_kwargs=physics_kwargs))
        val_loss, _ = quick_train(
            model, train_loader, val_loader, device, epochs=epochs,
            alpha=alpha, acceleration_weight=base["acceleration_weight"]
        )
        results["alpha"].append({"alpha": alpha, "val_mse": val_loss})
        del model

    for acceleration_weight in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        model = PLGAFormerTransformer(**model_kwargs(physics_kwargs=physics_kwargs))
        val_loss, _ = quick_train(
            model, train_loader, val_loader, device, epochs=epochs,
            alpha=base["alpha"], acceleration_weight=acceleration_weight
        )
        results["acceleration_weight"].append({
            "acceleration_weight": acceleration_weight, "val_mse": val_loss
        })
        del model

    for prior_scale in [0.0, 0.025, 0.05, 0.1, 0.2]:
        model = PLGAFormerTransformer(**model_kwargs(physics_kwargs=physics_kwargs))
        val_loss, _ = quick_train(
            model, train_loader, val_loader, device, epochs=epochs,
            alpha=base["alpha"], acceleration_weight=base["acceleration_weight"],
            kinematic_prior_max_scale=prior_scale,
        )
        results["kinematic_prior_max_scale"].append({
            "kinematic_prior_max_scale": prior_scale, "val_mse": val_loss
        })
        del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_input_length_analysis(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device,
    epochs: int,
    physics_kwargs: dict,
) -> list[dict]:
    results = []
    for seq_len in [32, 64, 128, 256]:
        actual_len = min(seq_len, x_train.shape[1])
        batch_size = max(16, min(64, 512 // max(1, actual_len)))
        train_loader, val_loader = make_loaders(
            x_train[:, -actual_len:, :],
            y_train,
            x_val[:, -actual_len:, :],
            y_val,
            batch_size,
        )
        model = PLGAFormerTransformer(**model_kwargs(physics_kwargs=physics_kwargs))
        val_loss, n_params = quick_train(model, train_loader, val_loader, device, epochs=epochs)
        results.append({"seq_len": actual_len, "val_mse": val_loss, "params": n_params})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def save_csv(out_dir: Path, layer_results: list[dict], physics_results: dict, input_results: list[dict]) -> None:
    with open(out_dir / "extended_ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["study", "parameter", "value", "val_mse", "params"])
        writer.writeheader()
        for row in layer_results:
            writer.writerow({"study": "layer_depth", "parameter": "encoder_layers", "value": row["layers"], "val_mse": row["val_mse"], "params": row["params"]})
        for param, rows in physics_results.items():
            for row in rows:
                writer.writerow({"study": "physics_sensitivity", "parameter": param, "value": row[param], "val_mse": row["val_mse"], "params": ""})
        for row in input_results:
            writer.writerow({"study": "input_length", "parameter": "seq_len", "value": row["seq_len"], "val_mse": row["val_mse"], "params": row["params"]})


def generate_figures(out_dir: Path, layer_results: list[dict], physics_results: dict, input_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["layers"] for r in layer_results], [r["val_mse"] for r in layer_results], "o-", linewidth=2)
    ax.set_xlabel("Encoder layers")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Layer depth ablation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_layer_ablation.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, param in zip(
        axes, ["alpha", "acceleration_weight", "kinematic_prior_max_scale"]
    ):
        rows = physics_results[param]
        x_vals = [r[param] for r in rows]
        y_vals = [r["val_mse"] for r in rows]
        ax.plot(x_vals, y_vals, "o-", linewidth=2)
        ax.set_xlabel(param)
        ax.set_ylabel("Validation MSE")
        ax.set_title(f"{param} sensitivity")
        ax.grid(alpha=0.3)
        if any(v > 0 for v in x_vals):
            ax.set_xscale("symlog", linthresh=1e-5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_physics_sensitivity.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["seq_len"] for r in input_results], [r["val_mse"] for r in input_results], "o-", linewidth=2)
    ax.set_xlabel("Input sequence length")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Input length impact")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_input_length.png", dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(42)

    max_samples = args.max_samples if args.max_samples is not None else (256 if args.quick else None)
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 30)
    batch_size = 32 if args.quick else 64
    x_train, y_train, x_val, y_val, physics_kwargs = build_base_arrays(max_samples=max_samples)
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, batch_size)

    layer_results = run_layer_ablation(train_loader, val_loader, device, epochs, physics_kwargs)
    physics_results = run_physics_sensitivity(train_loader, val_loader, device, epochs, physics_kwargs)
    input_results = run_input_length_analysis(
        x_train,
        y_train,
        x_val,
        y_val,
        device,
        epochs,
        physics_kwargs,
    )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "experiment": "exp7_longterm",
        "seed": 42,
        "selected_structure": "C",
        "epochs": epochs,
        "max_samples": max_samples,
        "quick": bool(args.quick),
        "formal": max_samples is None and epochs >= 10,
        "data_space": "standardized with train-split scalers",
        "layer_ablation": layer_results,
        "physics_sensitivity": physics_results,
        "input_length": input_results,
    }
    if args.quick:
        out_dir = out_dir / "quick"
        out_dir.mkdir(exist_ok=True)
    with open(out_dir / "extended_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    save_csv(out_dir, layer_results, physics_results, input_results)
    generate_figures(out_dir, layer_results, physics_results, input_results)
    print(f"Saved exp7 outputs to {out_dir}")
    return payload


if __name__ == "__main__":
    main()
