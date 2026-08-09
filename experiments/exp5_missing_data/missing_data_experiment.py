#!/usr/bin/env python3
"""Experiment 5: missing-data robustness comparison.

Compares PLGAFormer, a matched Transformer baseline, and iTransformer under
random and continuous missing observations. The experiment uses the project
source-context decoder protocol instead of passing ``None`` into decoder-based
models.
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
from models import HGVConfig, HGVPhysicsLoss, create_registered_model
from utils.inference_protocol import predict_by_eval_protocol
from utils.repro import set_global_seed
from utils.seq2seq_protocol import align_source_position_scale
from utils.train_protocol import build_adamw_optimizer, build_warmup_cosine_scheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing-data robustness experiment")
    parser.add_argument("--quick", action="store_true", help="Use a small subset and fewer epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit train/val/test samples")
    parser.add_argument("--models", default="PLGAFormer,Transformer,iTransformer")
    return parser.parse_args(argv)


def create_random_mask(
    batch_size: int,
    seq_len: int,
    n_features: int,
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return (
        torch.rand(batch_size, seq_len, n_features, generator=generator) > ratio
    ).float()


def create_continuous_mask(batch_size: int, seq_len: int, n_features: int, ratio: float) -> torch.Tensor:
    mask = torch.ones(batch_size, seq_len, n_features)
    missing_len = int(seq_len * ratio)
    if missing_len > 0:
        start = (seq_len - missing_len) // 2
        mask[:, start : start + missing_len, :] = 0
    return mask


def create_mask(
    batch_size: int,
    seq_len: int,
    n_features: int,
    ratio: float,
    mask_type: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if ratio <= 0:
        return torch.ones(batch_size, seq_len, n_features)
    if mask_type == "R":
        return create_random_mask(batch_size, seq_len, n_features, ratio, generator)
    if mask_type == "C":
        return create_continuous_mask(batch_size, seq_len, n_features, ratio)
    raise ValueError(f"Unknown mask type: {mask_type}")


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
    seq_len: int,
    pred_len: int,
    physics_kwargs: dict | None = None,
) -> torch.nn.Module:
    del seq_len, pred_len
    model_type = model_name.lower().replace("-", "_").replace(" ", "_")
    needs_scalers = model_type in {"plgaformer", "kinematic", "af_ciln"}
    return create_registered_model(
        model_type=model_type,
        input_dim=6,
        device=torch.device("cpu"),
        plgaformer_kwargs=physics_kwargs if needs_scalers else None,
    )


def predict(model, x: torch.Tensor, y: torch.Tensor, device, target_length: int | None = None) -> torch.Tensor:
    target_length = int(target_length or y.size(1))
    return predict_by_eval_protocol(
        model=model,
        x=x,
        y_true_scaled=y,
        pred_length=target_length,
        device=device,
        eval_protocol="source_context_decoder",
        eval_ar_seed_mode="zero",
        label_len=min(48, x.size(1)),
    )


def train_one_config(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mask_type: str,
    ratio: float,
    device,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 5e-5,
    warmup: int = 5,
    patience: int = 15,
    mask_seed: int = 1001,
) -> torch.nn.Module:
    optimizer = build_adamw_optimizer(model, learning_rate=lr, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=min(warmup, epochs), total_epochs=epochs)
    is_plgaformer = model.__class__.__name__ == "PLGAFormerTransformer"
    if is_plgaformer:
        physics = HGVConfig.get_physics_config()
        criterion = HGVPhysicsLoss(
            alpha=physics["alpha"],
            acceleration_weight=physics["acceleration_weight"],
            scaler_mean=model.output_scaler_mean.detach().cpu().numpy(),
            scaler_std=model.output_scaler_scale.detach().cpu().numpy(),
        ).to(device)
    else:
        criterion = torch.nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stale_epochs = 0

    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(int(mask_seed))
    train_mask_generator = torch.Generator().manual_seed(int(mask_seed) + 11)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = create_mask(
                x.size(0), x.size(1), x.size(2), ratio, mask_type, train_mask_generator
            ).to(device)
            optimizer.zero_grad()
            pred = predict(model, x * mask, y, device)
            loss = (
                criterion(pred, y, input_data=x * mask, include_mse=True)
                if is_plgaformer
                else criterion(pred, y)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_mask_generator = torch.Generator().manual_seed(int(mask_seed) + 23)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mask = create_mask(
                    x.size(0), x.size(1), x.size(2), ratio, mask_type, val_mask_generator
                ).to(device)
                val_loss += torch.nn.functional.mse_loss(
                    predict(model, x * mask, y, device), y
                ).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    mask_type: str,
    ratio: float,
    device,
    pred_horizons: list[int],
    mask_seed: int = 1001,
) -> dict:
    model.eval()
    max_horizon = max(pred_horizons)
    preds, labels = [], []
    test_mask_generator = torch.Generator().manual_seed(int(mask_seed) + 37)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        mask = create_mask(
            x.size(0), x.size(1), x.size(2), ratio, mask_type, test_mask_generator
        ).to(device)
        preds.append(predict(model, x * mask, y, device, target_length=max_horizon).cpu().numpy())
        labels.append(y[:, :max_horizon, :].cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    label_arr = np.concatenate(labels, axis=0)
    metrics = {}
    eps = 1e-6
    for horizon in pred_horizons:
        pred_h = pred_arr[:, :horizon, :]
        label_h = label_arr[:, :horizon, :]
        err = pred_h - label_h
        mse = np.mean(err**2)
        total_var = np.sum((label_h - label_h.mean()) ** 2) + eps
        metrics[f"{horizon}s"] = {
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - np.sum(err**2) / total_var),
        }
    return metrics


def subset(arr: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None:
        return arr
    return arr[: min(max_samples, arr.shape[0])]


def build_loaders(max_samples: int | None, batch_size: int):
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

    x_train = subset(scale_x(data["X_train"]), max_samples)
    y_train = subset(scale_y(data["y_train"]), max_samples)
    x_val = subset(scale_x(data["X_val"]), max_samples)
    y_val = subset(scale_y(data["y_val"]), max_samples)
    x_test = subset(scale_x(data["X_test"]), max_samples)
    y_test = subset(scale_y(data["y_test"]), max_samples)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)
    physics_kwargs = {
        "input_scaler_mean": np.concatenate([output_scaler.mean_, input_scaler.mean_[3:]]).astype(np.float32),
        "input_scaler_scale": np.concatenate([output_scaler.scale_, input_scaler.scale_[3:]]).astype(np.float32),
        "output_scaler_mean": output_scaler.mean_.astype(np.float32),
        "output_scaler_scale": output_scaler.scale_.astype(np.float32),
        "sampling_interval_s": float(np.asarray(data.get("sampling_interval_s", 1.0)).reshape(-1)[0]),
        "require_physical_scaler": True,
    }
    return train_loader, val_loader, test_loader, x_train.shape[1], y_train.shape[1], physics_kwargs


def save_outputs(out_dir: Path, results: dict, metadata: dict) -> None:
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "missing_data_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(out_dir / "missing_data_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "scenario", "horizon", "MSE", "RMSE", "MAE", "R2"])
        writer.writeheader()
        for key, horizon_map in results.items():
            model_name, scenario = key.split("_", 1)
            for horizon, metrics in horizon_map.items():
                writer.writerow({"model": model_name, "scenario": scenario, "horizon": horizon, **metrics})


def plot_results(out_dir: Path, results: dict, model_names: list[str], horizons: list[int]) -> None:
    colors = {"PLGAFormer": "#E31A1C", "Transformer": "#1F78B4", "iTransformer": "#FF7F00"}
    ratios = [0, 5, 10, 20, 40]
    fig, axes = plt.subplots(2, len(horizons), figsize=(5.5 * len(horizons), 8))
    if len(horizons) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, horizon in enumerate(horizons):
        for row, mask_type in enumerate(["R", "C"]):
            ax = axes[row, col]
            for model_name in model_names:
                values = []
                for ratio in ratios:
                    key = f"{model_name}_complete" if ratio == 0 else f"{model_name}_{mask_type}_{ratio}"
                    values.append(results.get(key, {}).get(f"{horizon}s", {}).get("MSE", np.nan))
                ax.plot(ratios, values, "o-", color=colors.get(model_name), linewidth=2, markersize=6, label=model_name)
            ax.set_xlabel("Missing ratio (%)")
            ax.set_ylabel("MSE")
            ax.set_title(f"{horizon}s, {'random' if mask_type == 'R' else 'continuous'} missing")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle("Missing-data robustness comparison", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_missing_data_mse.png", dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    set_global_seed(seed)

    quick_samples = 256 if args.quick else None
    max_samples = args.max_samples if args.max_samples is not None else quick_samples
    epochs = args.epochs if args.epochs is not None else (2 if args.quick else 50)
    batch_size = 32 if args.quick else 64
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    mask_types = ["R", "C"]
    missing_ratios = [0.05, 0.10, 0.20, 0.40]
    pred_horizons = [32, 64, 128, 256]

    train_loader, val_loader, test_loader, seq_len, pred_len, physics_kwargs = build_loaders(
        max_samples=max_samples,
        batch_size=batch_size,
    )
    all_results = {}
    count = 0
    total = len(model_names) * (1 + len(mask_types) * len(missing_ratios))

    for model_name in model_names:
        for mask_type, ratio, scenario in [("R", 0.0, "complete")] + [
            (mt, r, f"{mt}_{int(r * 100)}") for mt in mask_types for r in missing_ratios
        ]:
            count += 1
            key = f"{model_name}_{scenario}"
            print(f"[{count}/{total}] {key}", flush=True)
            set_global_seed(seed)
            scenario_seed = 1001 + int(ratio * 1000) + (10000 if mask_type == "C" else 0)
            model = build_model(
                model_name,
                seq_len=seq_len,
                pred_len=pred_len,
                physics_kwargs=physics_kwargs if model_name.lower() == "plgaformer" else None,
            ).to(device)
            model = train_one_config(
                model, train_loader, val_loader, mask_type, ratio, device,
                epochs=epochs, mask_seed=scenario_seed,
            )
            all_results[key] = evaluate_model(
                model, test_loader, mask_type, ratio, device, pred_horizons,
                mask_seed=scenario_seed,
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out_dir = Path(__file__).resolve().parent / "results"
    if args.quick:
        out_dir = out_dir / "quick"
    metadata = {
        "experiment": "exp5_missing_data",
        "models": model_names,
        "mask_types": mask_types,
        "missing_ratios": missing_ratios,
        "prediction_horizons": pred_horizons,
        "protocol": "source_context_decoder",
        "seed": seed,
        "mask_seed_protocol": "fixed per scenario and shared across models",
        "epochs": epochs,
        "max_samples": max_samples,
        "quick": bool(args.quick),
        "formal": max_samples is None and epochs >= 10,
        "data_space": "standardized with train-split scalers",
        "completed_experiments": count,
        "total_experiments": total,
    }
    save_outputs(out_dir, all_results, metadata)
    plot_results(out_dir, all_results, model_names, pred_horizons)
    print(f"Saved exp5 outputs to {out_dir}")
    return all_results


if __name__ == "__main__":
    main()
