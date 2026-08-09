#!/usr/bin/env python3
"""Generate continuous, checkpoint-grounded TAES horizon-error profiles.

The script performs no training. It reconstructs the frozen formal models,
evaluates every lead time from 1 to 256 s on the complete formal test split,
and refuses to export figures unless the 32/64/128/256 s anchors reproduce
the signed TAES main-results artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "main_run_set_manifest.json"
)
DEFAULT_MAIN_RESULTS = DEFAULT_REGISTRY
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "taes_submission_artifacts" / "generated" / "formal_v3"
FORMAL_SEEDS = (42, 123, 456)
FORMAL_HORIZONS = (32, 64, 128, 256)
PRED_LEN = 256
MODEL_ORDER = ("plgaformer", "af_ciln", "transformer", "rotating_3dof")
MODEL_NAMES = {
    "plgaformer": "PLGAFormer",
    "af_ciln": "AF-CILN",
    "transformer": "Transformer",
    "rotating_3dof": "3-DOF predictor",
}
SUMMARY_NAMES = {
    "plgaformer": "PLGAFormer (proposed)",
    "af_ciln": "AF-CILN",
    "transformer": "Transformer (baseline)",
    "rotating_3dof": "Rotating-Earth 3-DOF",
}
COLORS = {
    "plgaformer": "#0F4D92",
    "af_ciln": "#C27A2C",
    "transformer": "#6B7280",
    "rotating_3dof": "#2E8B57",
}
LINESTYLES = {
    "plgaformer": "-",
    "af_ciln": "-",
    "transformer": "-",
    "rotating_3dof": "--",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_paths import (  # noqa: E402
    get_dataset_npz_path,
    get_input_scaler_path,
    get_output_scaler_path,
)
from models import HGVConfig  # noqa: E402
from models.model_factory import create_registered_model  # noqa: E402
from models.plgaformer import spherical_to_cartesian_torch  # noqa: E402
from utils.final_plgaformer import (  # noqa: E402
    FINAL_ARCHITECTURE,
    final_evidence_signature,
    final_plgaformer_kwargs,
    resolve_final_plgaformer,
)
from utils.inference_protocol import predict_by_eval_protocol  # noqa: E402
from utils.formal_evidence import (  # noqa: E402
    bundle_as_registry,
    load_evidence_bundle,
    load_formal_config,
    normalize_run_record,
    validate_evidence_bundle,
)
from utils.seq2seq_protocol import align_source_position_scale  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "formal_v3.json")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--main-results", type=Path, default=DEFAULT_MAIN_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--af-ciln-root",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "AF-CILN",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Reuse the already validated profile manifest and regenerate only figure files.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("bundle_kind") == "main":
        return bundle_as_registry(path)
    raise RuntimeError("Horizon profiles require an eligible formal-v3 Main Results bundle.")


def main_summary_from_bundle(path: Path) -> dict[str, Any]:
    bundle = load_evidence_bundle(path)
    if bundle.get("bundle_kind") != "main" or bundle.get("paper_eligible") is not True:
        raise RuntimeError("Horizon profiles require an eligible formal-v3 Main Results bundle.")
    key_to_summary = {
        "full": "PLGAFormer (proposed)",
        "baseline": "Transformer (baseline)",
        "af_ciln": "AF-CILN",
        "rotating_3dof": "Rotating-Earth 3-DOF",
    }
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for entry in bundle.get("source_records", []):
        key = str(entry.get("model_key", ""))
        if key not in key_to_summary:
            continue
        record = normalize_run_record(entry["source_path"])
        seed = 42 if key == "rotating_3dof" else int(record["seed"])
        grouped.setdefault(key, {})[seed] = {}
        for horizon in FORMAL_HORIZONS:
            item = record["eval_results"].get(str(horizon), record["eval_results"].get(horizon, {}))
            for metric, aliases in {"ade": ("ade", "trajectory_window_ade"), "fde": ("fde", "trajectory_window_fde"), "rmse_cart_m": ("rmse_cart_m",)}.items():
                grouped[key][seed].setdefault(str(horizon), {})[metric] = float(next(item[alias] for alias in aliases if alias in item))
    seed_statistics = {}
    for key, summary_name in key_to_summary.items():
        if key not in grouped:
            raise RuntimeError(f"Main bundle lacks anchor records for {key}.")
        seed_statistics[summary_name] = {}
        seeds = [42] if key == "rotating_3dof" else list(FORMAL_SEEDS)
        for horizon in FORMAL_HORIZONS:
            seed_statistics[summary_name][str(horizon)] = {}
            for metric in ("ade", "fde", "rmse_cart_m"):
                values = [grouped[key][seed][str(horizon)][metric] for seed in seeds]
                seed_statistics[summary_name][str(horizon)][metric] = {"values_m": values}
    return {"bundle_id": bundle["bundle_id"], "seed_statistics": seed_statistics}


def find_run(
    registry: dict[str, Any],
    signature: str,
    seed: int,
    model_type: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in registry.get("runs", {}).values()
        if record.get("run_signature") == signature
        and int(record.get("seed", -1)) == int(seed)
        and record.get("model_type") == model_type
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one formal run for model={model_type}, seed={seed}; "
            f"found {len(matches)}."
        )
    return matches[0]


def physical_kwargs(input_scaler, output_scaler) -> dict[str, Any]:
    return {
        "input_scaler_mean": np.concatenate(
            [output_scaler.mean_, input_scaler.mean_[3:]]
        ).astype(np.float32),
        "input_scaler_scale": np.concatenate(
            [output_scaler.scale_, input_scaler.scale_[3:]]
        ).astype(np.float32),
        "output_scaler_mean": output_scaler.mean_.astype(np.float32),
        "output_scaler_scale": output_scaler.scale_.astype(np.float32),
        "sampling_interval_s": 1.0,
        "require_physical_scaler": True,
    }


def load_formal_test_loader(
    config,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, np.ndarray, Any, Any, dict[str, Any]]:
    dataset_path = get_dataset_npz_path(PROJECT_ROOT)
    with np.load(dataset_path, allow_pickle=False) as data:
        protocol = str(np.asarray(data["dataset_protocol"]).item())
        seq_len = int(np.asarray(data["seq_len"]).item())
        pred_len = int(np.asarray(data["pred_len"]).item())
        sampling_interval_s = float(np.asarray(data["sampling_interval_s"]).item())
        expected_protocol = str(config["dataset"]["protocol"])
        expected_seq_len = int(config["task"]["seq_len"])
        expected_pred_len = int(config["task"]["pred_len"])
        expected_sampling_interval = float(config["dataset"]["sampling_interval_s"])
        if (
            protocol != expected_protocol
            or seq_len != expected_seq_len
            or pred_len != expected_pred_len
            or not np.isclose(sampling_interval_s, expected_sampling_interval)
        ):
            raise RuntimeError(
                "Formal dataset contract mismatch: "
                f"protocol={protocol}, seq_len={seq_len}, pred_len={pred_len}, "
                f"dt={sampling_interval_s}."
            )
        x_physical = np.asarray(data["X_test"], dtype=np.float32)
        y_physical = np.asarray(data["y_test"], dtype=np.float32)
        trajectory_ids = np.asarray(data["trajectory_ids_test"], dtype=np.int64)
        test_split_ids = np.asarray(
            data["trajectory_split_test_ids"], dtype=np.int64
        )

    expected_test_count = int(config["split"]["complete_trajectory_counts"]["test"])
    expected_input_dim = int(config["task"]["input_dim"])
    expected_output_dim = int(config["task"]["output_dim"])
    if (
        x_physical.ndim != 3
        or y_physical.ndim != 3
        or x_physical.shape[1:] != (expected_seq_len, expected_input_dim)
        or y_physical.shape[1:] != (expected_pred_len, expected_output_dim)
    ):
        raise RuntimeError(
            f"Unexpected formal test shapes: X={x_physical.shape}, y={y_physical.shape}."
        )
    unique_ids = np.unique(trajectory_ids)
    if unique_ids.size != expected_test_count or not np.array_equal(unique_ids, np.sort(test_split_ids)):
        raise RuntimeError(
            "The formal test windows do not map to the configured complete-trajectory "
            f"test split of {expected_test_count} trajectories."
        )

    input_scaler = joblib.load(get_input_scaler_path(PROJECT_ROOT))
    output_scaler = joblib.load(get_output_scaler_path(PROJECT_ROOT))
    x_scaled = input_scaler.transform(
        x_physical.reshape(-1, x_physical.shape[-1])
    ).reshape(x_physical.shape)
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=3,
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(
        y_physical.reshape(-1, y_physical.shape[-1])
    ).reshape(y_physical.shape).astype(np.float32, copy=False)

    id_to_row = {int(value): index for index, value in enumerate(unique_ids)}
    trajectory_rows = np.asarray(
        [id_to_row[int(value)] for value in trajectory_ids],
        dtype=np.int64,
    )
    dataset = TensorDataset(
        torch.from_numpy(x_scaled),
        torch.from_numpy(y_scaled),
        torch.from_numpy(trajectory_rows),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=bool(workers > 0),
        persistent_workers=bool(workers > 0),
    )
    audit = {
        "path": str(dataset_path.resolve()),
        "sha256": sha256_file(dataset_path),
        "dataset_protocol": protocol,
        "config_sha256": config.config_sha256,
        "expected_test_trajectories": expected_test_count,
        "test_windows": int(x_physical.shape[0]),
        "test_trajectories": int(unique_ids.size),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "sampling_interval_s": sampling_interval_s,
    }
    return loader, unique_ids, input_scaler, output_scaler, audit


def build_model(
    model_type: str,
    seed: int,
    device: torch.device,
    registry: dict[str, Any],
    signature: str,
    input_scaler,
    output_scaler,
    bundle_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(int(seed))
    kwargs = (
        physical_kwargs(input_scaler, output_scaler)
        if model_type in {"plgaformer", "af_ciln", "rotating_3dof"}
        else None
    )
    if model_type == "plgaformer":
        kwargs.update(final_plgaformer_kwargs())
    model = create_registered_model(
        model_type=model_type,
        input_dim=6,
        device=device,
        plgaformer_kwargs=kwargs,
    )

    if model_type == "rotating_3dof":
        return model, {
            "checkpoint": None,
            "source": "parameter-free analytical propagation",
        }

    if model_type == "plgaformer":
        checkpoint, source_payload, audit = resolve_final_plgaformer(seed, bundle_path=bundle_path)
        protocol = source_payload.get("config", {})
        if protocol.get("eval_protocol") != "source_context_decoder":
            raise RuntimeError("Final PLGAFormer evaluation protocol is not source_context_decoder.")
    else:
        record = find_run(registry, signature, seed, model_type)
        checkpoint = Path(record.get("checkpoint_path", "")).resolve()
        audit = {
            "seed": int(seed),
            "record_completed_at": record.get("completed_at"),
            "record_model_name": record.get("model_name"),
        }
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing formal checkpoint: {checkpoint}")

    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    return model, {
        **audit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def profile_from_accumulators(
    trajectory_lead_sum_m: np.ndarray,
    trajectory_window_count: np.ndarray,
    component_square_sum_m2: np.ndarray,
    total_windows: int,
) -> dict[str, np.ndarray]:
    """Return trajectory-equal ADE/FDE and pooled ECEF RMSE for all horizons."""
    lead_sum = np.asarray(trajectory_lead_sum_m, dtype=np.float64)
    counts = np.asarray(trajectory_window_count, dtype=np.float64)
    square_sum = np.asarray(component_square_sum_m2, dtype=np.float64)
    if lead_sum.ndim != 2 or lead_sum.shape[1] != PRED_LEN:
        raise ValueError("trajectory_lead_sum_m must have shape [trajectories, 256].")
    if counts.shape != (lead_sum.shape[0],) or np.any(counts <= 0):
        raise ValueError("Every trajectory must have a positive window count.")
    if square_sum.shape != (PRED_LEN,) or int(total_windows) <= 0:
        raise ValueError("Invalid pooled Cartesian square-error accumulator.")

    trajectory_lead_mean = lead_sum / counts[:, None]
    fde = trajectory_lead_mean.mean(axis=0)
    ade = np.cumsum(trajectory_lead_mean, axis=1).mean(axis=0) / np.arange(
        1, PRED_LEN + 1
    )
    rmse = np.sqrt(
        np.cumsum(square_sum)
        / (float(total_windows) * 3.0 * np.arange(1, PRED_LEN + 1))
    )
    return {"ade": ade, "fde": fde, "rmse_cart_m": rmse}


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    trajectory_count: int,
    output_scaler,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    trajectory_lead_sum = np.zeros((trajectory_count, PRED_LEN), dtype=np.float64)
    trajectory_window_count = np.zeros(trajectory_count, dtype=np.int64)
    component_square_sum = np.zeros(PRED_LEN, dtype=np.float64)
    total_windows = 0
    label_len = int(HGVConfig.get_train_config().get("label_len", 128))
    output_mean = torch.as_tensor(
        output_scaler.mean_, dtype=torch.float32, device=device
    )
    output_scale = torch.as_tensor(
        output_scaler.scale_, dtype=torch.float32, device=device
    )

    with torch.inference_mode():
        for x_scaled, y_scaled, trajectory_rows in loader:
            x_scaled = x_scaled.to(device, non_blocking=True)
            y_scaled = y_scaled.to(device, non_blocking=True)
            prediction_scaled = predict_by_eval_protocol(
                model=model,
                x=x_scaled,
                y_true_scaled=y_scaled,
                pred_length=PRED_LEN,
                device=device,
                eval_protocol="source_context_decoder",
                eval_ar_seed_mode="zero",
                label_len=label_len,
            )
            if prediction_scaled.shape != y_scaled.shape:
                raise RuntimeError(
                    f"Incomplete prediction: {tuple(prediction_scaled.shape)} "
                    f"vs target {tuple(y_scaled.shape)}."
                )
            if not torch.isfinite(prediction_scaled).all():
                raise FloatingPointError("Non-finite prediction in formal evaluation.")

            prediction_physical = prediction_scaled * output_scale + output_mean
            target_physical = y_scaled * output_scale + output_mean
            prediction_ecef = spherical_to_cartesian_torch(prediction_physical)
            target_ecef = spherical_to_cartesian_torch(target_physical)
            difference = prediction_ecef - target_ecef
            displacement = torch.linalg.vector_norm(difference, dim=-1)

            displacement_np = displacement.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            rows_np = trajectory_rows.numpy()
            np.add.at(trajectory_lead_sum, rows_np, displacement_np)
            np.add.at(trajectory_window_count, rows_np, 1)
            component_square_sum += (
                difference.square().sum(dim=(0, 2)).detach().cpu().numpy()
            )
            total_windows += int(x_scaled.size(0))

    if total_windows != len(loader.dataset):
        raise RuntimeError(
            f"Evaluated {total_windows} windows, expected {len(loader.dataset)}."
        )
    return profile_from_accumulators(
        trajectory_lead_sum,
        trajectory_window_count,
        component_square_sum,
        total_windows,
    )


def validate_formal_anchors(
    profiles: dict[str, dict[str, dict[str, np.ndarray]]],
    main_results: dict[str, Any],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for model_type in MODEL_ORDER:
        summary_name = SUMMARY_NAMES[model_type]
        seed_profiles = profiles[model_type]
        seeds = (42,) if model_type == "rotating_3dof" else FORMAL_SEEDS
        for seed in seeds:
            seed_key = str(seed)
            for horizon in FORMAL_HORIZONS:
                for metric in ("ade", "fde", "rmse_cart_m"):
                    actual = float(seed_profiles[seed_key][metric][horizon - 1])
                    expected_values = main_results["seed_statistics"][summary_name][
                        str(horizon)
                    ][metric]["values_m"]
                    expected = (
                        float(expected_values[0])
                        if model_type == "rotating_3dof"
                        else float(expected_values[FORMAL_SEEDS.index(seed)])
                    )
                    difference = actual - expected
                    if not np.isclose(actual, expected, rtol=2e-5, atol=0.25):
                        raise RuntimeError(
                            "Formal anchor mismatch: "
                            f"model={model_type}, seed={seed}, horizon={horizon}, "
                            f"metric={metric}, actual={actual}, expected={expected}."
                        )
                    audit.append(
                        {
                            "model": MODEL_NAMES[model_type],
                            "seed": int(seed),
                            "horizon_s": int(horizon),
                            "metric": metric,
                            "actual_m": actual,
                            "expected_m": expected,
                            "difference_m": difference,
                        }
                    )
    return audit


def summarize_profiles(
    profiles: dict[str, dict[str, dict[str, np.ndarray]]]
) -> dict[str, dict[str, np.ndarray]]:
    summary: dict[str, dict[str, np.ndarray]] = {}
    for model_type in MODEL_ORDER:
        seed_keys = sorted(profiles[model_type], key=int)
        summary[model_type] = {}
        for metric in ("ade", "fde", "rmse_cart_m"):
            values = np.stack(
                [profiles[model_type][key][metric] for key in seed_keys],
                axis=0,
            )
            summary[model_type][f"{metric}_mean"] = values.mean(axis=0)
            summary[model_type][f"{metric}_std"] = (
                values.std(axis=0, ddof=1)
                if values.shape[0] > 1
                else np.zeros(values.shape[1], dtype=np.float64)
            )
    return summary


def crossover_audit(
    summary: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, int | None]]:
    """Find the first lead after which PLGAFormer remains below each comparator."""
    output: dict[str, dict[str, int | None]] = {}
    for metric in ("ade", "fde", "rmse_cart_m"):
        proposed = summary["plgaformer"][f"{metric}_mean"]
        output[metric] = {}
        for comparator in ("af_ciln", "transformer", "rotating_3dof"):
            advantage = proposed < summary[comparator][f"{metric}_mean"]
            persistent_from = None
            for index in range(PRED_LEN):
                if bool(np.all(advantage[index:])):
                    persistent_from = index + 1
                    break
            output[metric][MODEL_NAMES[comparator]] = persistent_from
    return output


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.set_xlim(1, PRED_LEN)
    ax.set_xticks(FORMAL_HORIZONS)
    ax.set_xlabel("Forecast horizon (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path_base: Path) -> list[str]:
    outputs = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = path_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025, **kwargs)
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs


def plot_profiles(
    summary: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> list[str]:
    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = np.arange(1, PRED_LEN + 1)
    generated: list[str] = []
    metric_specs = (
        ("ade", "ADE (km)", "fig_horizon_error_ade"),
        ("fde", "FDE (km)", "fig_horizon_error_fde"),
        ("rmse_cart_m", "ECEF RMSE (km)", "fig_horizon_error_rmse"),
    )
    for metric, ylabel, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(2.22, 1.88))
        for model_type in MODEL_ORDER:
            mean = summary[model_type][f"{metric}_mean"] / 1000.0
            std = summary[model_type][f"{metric}_std"] / 1000.0
            linewidth = 2.0 if model_type == "plgaformer" else 1.25
            zorder = 5 if model_type == "plgaformer" else 3
            ax.plot(
                horizons,
                mean,
                color=COLORS[model_type],
                linestyle=LINESTYLES[model_type],
                linewidth=linewidth,
                zorder=zorder,
            )
            anchor_index = np.asarray(FORMAL_HORIZONS) - 1
            marker_size = 3.8 if model_type == "plgaformer" else 3.2
            ax.plot(
                np.asarray(FORMAL_HORIZONS),
                mean[anchor_index],
                linestyle="none",
                marker="o",
                markersize=marker_size,
                markerfacecolor="white",
                markeredgecolor=COLORS[model_type],
                markeredgewidth=0.75,
                zorder=zorder + 1,
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        finish_axes(ax)
        generated.extend(save_figure(fig, output_dir / filename))

    fig, ax = plt.subplots(figsize=(6.65, 0.22))
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model_type],
            linestyle=LINESTYLES[model_type],
            linewidth=2.0 if model_type == "plgaformer" else 1.25,
            label=MODEL_NAMES[model_type],
        )
        for model_type in MODEL_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.6,
    )
    generated.extend(save_figure(fig, output_dir / "fig_horizon_error_legend"))
    return generated


def write_csv_artifact(
    profiles: dict[str, dict[str, dict[str, np.ndarray]]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "seed", "horizon_s", "ade_m", "fde_m", "rmse_cart_m"]
        )
        for model_type in MODEL_ORDER:
            for seed_key in sorted(profiles[model_type], key=int):
                current = profiles[model_type][seed_key]
                for index in range(PRED_LEN):
                    writer.writerow(
                        [
                            MODEL_NAMES[model_type],
                            seed_key if model_type != "rotating_3dof" else "deterministic",
                            index + 1,
                            f"{current['ade'][index]:.9f}",
                            f"{current['fde'][index]:.9f}",
                            f"{current['rmse_cart_m'][index]:.9f}",
                        ]
                    )


def serializable_profiles(
    profiles: dict[str, dict[str, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    return {
        MODEL_NAMES[model_type]: {
            seed: {metric: values.tolist() for metric, values in current.items()}
            for seed, current in seed_profiles.items()
        }
        for model_type, seed_profiles in profiles.items()
    }


def profiles_from_manifest(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    stored = payload.get("profiles", {})
    aliases = {
        "plgaformer": ("PLGAFormer",),
        "af_ciln": ("AF-CILN",),
        "transformer": ("Transformer",),
        "rotating_3dof": ("3-DOF predictor", "Rotating-Earth 3-DOF"),
    }
    profiles: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model_type, names in aliases.items():
        source = next((stored[name] for name in names if name in stored), None)
        if source is None:
            raise RuntimeError(f"Validated manifest lacks profiles for {model_type}.")
        profiles[model_type] = {
            str(seed): {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in current.items()
            }
            for seed, current in source.items()
        }
    return profiles


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    config = load_formal_config(args.config)
    if not args.registry.is_file():
        raise FileNotFoundError(f"Formal-v3 Main Results bundle is missing: {args.registry}")
    verification = validate_evidence_bundle(args.registry, config, expected_kind="main")
    if not verification["passed"]:
        codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
        raise RuntimeError("Formal-v3 Main Results bundle failed verification: " + ", ".join(codes))
    bundle = verification["bundle"]
    if args.output_dir is None:
        if bundle is None or bundle.get("bundle_kind") != "main":
            raise RuntimeError("Horizon profiles require a formal-v3 Main bundle for default staging.")
        args.output_dir = DEFAULT_OUTPUT / "hgv_multiregime_state_v2_1" / str(bundle["bundle_id"])
    if args.replot_only:
        manifest_path = args.output_dir / "horizon_error_profiles_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Validated profile manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(manifest.get("formal_anchor_audit", [])) != 120:
            raise RuntimeError("Replot refused because the formal 120-anchor audit is incomplete.")
        profiles = profiles_from_manifest(manifest)
        generated = plot_profiles(summarize_profiles(profiles), args.output_dir)
        manifest["profiles"] = serializable_profiles(profiles)
        manifest["display_labels"] = {
            model_type: MODEL_NAMES[model_type] for model_type in MODEL_ORDER
        }
        manifest["restyled_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["figures"] = generated
        manifest["figure_sha256"] = {
            str(Path(path).name): sha256_file(Path(path)) for path in generated
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path.resolve()),
                    "replot_only": True,
                    "anchor_checks_preserved": 120,
                    "figures": generated,
                },
                indent=2,
            )
        )
        return 0

    if not args.af_ciln_root.is_dir():
        raise FileNotFoundError(f"AF-CILN checkout is missing: {args.af_ciln_root}")
    os.environ["HGV_AF_CILN_ROOT"] = str(args.af_ciln_root.resolve())

    device = torch.device(args.device)
    registry, signature = load_registry(args.registry)
    if bundle is None:
        bundle = load_evidence_bundle(args.registry)
    main_results = main_summary_from_bundle(args.main_results)
    if main_results.get("bundle_id") != signature:
        raise RuntimeError("Main-results bundle does not match the selected formal registry bundle.")
    final_signature, final_signature_audit = final_evidence_signature(
        signature, bundle_path=args.registry
    )

    loader, trajectory_ids, input_scaler, output_scaler, dataset_audit = (
        load_formal_test_loader(config, args.batch_size, args.workers)
    )
    profiles: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    checkpoint_audit: dict[str, Any] = {}
    for model_type in MODEL_ORDER:
        profiles[model_type] = {}
        checkpoint_audit[MODEL_NAMES[model_type]] = {}
        seeds = (42,) if model_type == "rotating_3dof" else FORMAL_SEEDS
        for seed in seeds:
            print(f"[horizon-profile] model={MODEL_NAMES[model_type]} seed={seed}")
            model, audit = build_model(
                model_type,
                seed,
                device,
                registry,
                signature,
                input_scaler,
                output_scaler,
                args.registry,
            )
            profiles[model_type][str(seed)] = evaluate_model(
                model,
                loader,
                int(trajectory_ids.size),
                output_scaler,
                device,
            )
            checkpoint_audit[MODEL_NAMES[model_type]][str(seed)] = audit
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    anchor_audit = validate_formal_anchors(profiles, main_results)
    summary = summarize_profiles(profiles)
    crossovers = crossover_audit(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "horizon_error_profiles.csv"
    write_csv_artifact(profiles, csv_path)
    generated = plot_profiles(summary, args.output_dir)

    manifest = {
        "artifact": "taes_continuous_horizon_error_profiles",
        "artifact_kind": "formal_v3_taes_horizon_error_profiles",
        "bundle_id": bundle.get("bundle_id"),
        "config_sha256": bundle.get("config_sha256"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "formal_protocol": {
            "dataset_protocol": config["dataset"]["protocol"],
            "observation_length_s": int(config["task"]["seq_len"]),
            "forecast_horizons_s": [1, int(config["task"]["pred_len"])],
            "window_stride_s": int(config["task"]["window_stride"]),
            "test_trajectory_count": int(dataset_audit["test_trajectories"]),
            "eval_protocol": "source_context_decoder",
            "eval_ar_seed_mode": "zero",
            "label_len": 128,
            "ade_fde_aggregation": (
                f"equal weight over {int(dataset_audit['test_trajectories'])} source trajectories "
                "after within-trajectory window averaging"
            ),
            "rmse_definition": "pooled ECEF component RMSE over test windows, lead times, and xyz components",
            "uncertainty": "sample standard deviation across seeds 42, 123, and 456 for learned models",
        },
        "base_exp1_signature": signature,
        "final_evidence_signature": final_signature,
        "final_architecture": FINAL_ARCHITECTURE,
        "final_signature_audit": final_signature_audit,
        "dataset_audit": dataset_audit,
        "checkpoint_audit": checkpoint_audit,
        "formal_anchor_audit": anchor_audit,
        "persistent_advantage_from_s": crossovers,
        "profiles": serializable_profiles(profiles),
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "figures": generated,
        "figure_sha256": {
            str(Path(path).name): sha256_file(Path(path)) for path in generated
        },
    }
    manifest_path = args.output_dir / "horizon_error_profiles_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "anchor_checks": len(anchor_audit),
                "persistent_advantage_from_s": crossovers,
                "figures": generated,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
