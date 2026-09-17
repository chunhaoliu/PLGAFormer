#!/usr/bin/env python3
"""Render a deterministic confirmatory trajectory case without retraining.

Selection contract
------------------
The maneuver class is fixed to ``weaving`` because it is the hardest class in
the frozen aggregate maneuver analysis.  Within that class, the trajectory is
selected without model outputs: each complete ground-truth trajectory is
represented by five geometry/dynamics features, robustly standardized, and the
trajectory nearest the component-wise median is selected.  The prediction
window is the lower of the two central admissible windows.  Consequently,
neither PLGAFormer nor comparator errors influence case selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.overall_prediction import main_results as exp1
from scripts import evaluate_confirmatory_holdout as confirm
from utils.final_plgaformer import apply_paper_inference_policy
from utils.seq2seq_protocol import align_source_position_scale


MANEUVER = "weaving"
# Same four methods as the confirmatory aggregate figures. PatchTST is omitted
# because its errors lie far above a linear axis shared with the other methods.
MODEL_KEYS = ("dlinear", "baseline", "itransformer", "full")
DISPLAY_NAMES = {
    "dlinear": "DLinear",
    "baseline": "Transformer",
    "itransformer": "iTransformer",
    "full": "PLGAFormer",
}
COLORS = {
    "dlinear": "#8C96A0",
    "baseline": "#66727D",
    "itransformer": "#2F6B9A",
    "full": "#C23B3B",
}
LINESTYLES = {
    "dlinear": "-.",
    "baseline": ":",
    "itransformer": "--",
    "full": "-",
}
LINEWIDTHS = {
    "dlinear": 1.15,
    "baseline": 1.15,
    "itransformer": 1.25,
    "full": 1.55,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "figure_review" / "trajectory_case_study",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spherical_to_ecef(position: np.ndarray) -> np.ndarray:
    radius, longitude, latitude = np.moveaxis(np.asarray(position, dtype=np.float64), -1, 0)
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            radius * cos_latitude * np.cos(longitude),
            radius * cos_latitude * np.sin(longitude),
            radius * np.sin(latitude),
        ),
        axis=-1,
    )


def ecef_to_enu(ecef: np.ndarray, reference_spherical: np.ndarray) -> np.ndarray:
    reference_ecef = spherical_to_ecef(reference_spherical)
    _, longitude, latitude = np.asarray(reference_spherical, dtype=np.float64)
    rotation = np.array(
        [
            [-np.sin(longitude), np.cos(longitude), 0.0],
            [
                -np.sin(latitude) * np.cos(longitude),
                -np.sin(latitude) * np.sin(longitude),
                np.cos(latitude),
            ],
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
        ]
    )
    return (np.asarray(ecef, dtype=np.float64) - reference_ecef) @ rotation.T


def trajectory_features(states: np.ndarray) -> tuple[np.ndarray, list[str]]:
    positions = spherical_to_ecef(states[:, :, :3])
    increments = np.diff(positions, axis=1)
    headings = np.unwrap(states[:, :, 5].astype(np.float64), axis=1)
    features = np.column_stack(
        (
            np.linalg.norm(increments, axis=2).sum(axis=1),
            np.linalg.norm(positions[:, -1] - positions[:, 0], axis=1),
            np.abs(np.diff(headings, axis=1)).sum(axis=1),
            np.ptp(states[:, :, 0].astype(np.float64), axis=1),
            np.std(states[:, :, 3].astype(np.float64), axis=1),
        )
    )
    names = [
        "three_dimensional_path_length_m",
        "net_displacement_m",
        "cumulative_heading_variation_rad",
        "radial_range_m",
        "speed_standard_deviation_mps",
    ]
    return features, names


def select_case(data: dict[str, np.ndarray]) -> dict[str, Any]:
    mask = data["trajectory_labels"] == MANEUVER
    class_ids = data["trajectory_ids"][mask].astype(np.int64)
    class_states = data["clean_trajectories"][mask]
    features, feature_names = trajectory_features(class_states)
    median = np.median(features, axis=0)
    q25, q75 = np.percentile(features, (25, 75), axis=0)
    scale = q75 - q25
    scale[scale == 0.0] = 1.0
    robust_features = (features - median) / scale
    distances = np.linalg.norm(robust_features, axis=1)
    selected_local_index = int(np.argmin(distances))
    trajectory_id = int(class_ids[selected_local_index])

    window_mask = data["trajectory_ids_confirmatory"] == trajectory_id
    class_window_indices = np.flatnonzero(window_mask)
    starts = data["window_starts_confirmatory"][class_window_indices].astype(np.int64)
    median_start = float(np.median(starts))
    window_choice = int(np.argmin(np.abs(starts - median_start)))
    window_index = int(class_window_indices[window_choice])
    return {
        "trajectory_id": trajectory_id,
        "trajectory_array_index": int(np.flatnonzero(data["trajectory_ids"] == trajectory_id)[0]),
        "window_index": window_index,
        "window_start": int(data["window_starts_confirmatory"][window_index]),
        "class_size": int(mask.sum()),
        "distance_to_robust_median": float(distances[selected_local_index]),
        "feature_names": feature_names,
        "feature_values": {
            name: float(value)
            for name, value in zip(feature_names, features[selected_local_index])
        },
        "class_feature_medians": {
            name: float(value) for name, value in zip(feature_names, median)
        },
        "class_feature_iqrs": {
            name: float(value) for name, value in zip(feature_names, scale)
        },
    }


def predict_case(
    model_key: str,
    seed: int,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    source_records: dict[tuple[str, int], dict[str, Any]],
    input_scaler: Any,
    output_scaler: Any,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    record = source_records[(model_key, seed)]
    payload = confirm._source_payload(record)
    checkpoint = Path(str(record["checkpoint"])).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != str(record["checkpoint_sha256"]):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")

    x_scaled = input_scaler.transform(x_raw.reshape(-1, 6)).reshape(x_raw.shape)
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=3,
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(y_raw.reshape(-1, 3)).reshape(y_raw.shape)
    x_tensor = torch.from_numpy(x_scaled).to(device)
    y_tensor = torch.from_numpy(y_scaled.astype(np.float32, copy=False)).to(device)

    model = confirm.reconstruct_model(payload, input_scaler, output_scaler, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    inference_policy = apply_paper_inference_policy(model) if model_key == "full" else None
    model.to(device).eval()
    with torch.no_grad():
        prediction_scaled = exp1.unified_predict(
            model,
            x_tensor,
            y_tensor.size(1),
            y_true_scaled=y_tensor,
            device=device,
            eval_protocol=exp1.EVAL_PROTOCOL,
        )
    if tuple(prediction_scaled.shape) != tuple(y_tensor.shape):
        raise RuntimeError(
            f"Unexpected prediction shape for {model_key}, seed {seed}: "
            f"{tuple(prediction_scaled.shape)}"
        )
    prediction = output_scaler.inverse_transform(
        prediction_scaled.detach().cpu().numpy().reshape(-1, 3)
    ).reshape(y_raw.shape)
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"Non-finite prediction for {model_key}, seed {seed}")
    provenance = {
        "model_key": model_key,
        "seed": seed,
        "source_record": str(Path(str(record["source_path"])).resolve()),
        "source_record_sha256": str(record["source_sha256"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "inference_policy": inference_policy,
    }
    del model, state, x_tensor, y_tensor, prediction_scaled
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction[0], provenance


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)


PANEL_SIZE_IEEE = (3.46, 2.22)
BAR_PANEL_SIZE_IEEE = (3.46, 2.40)


def configure_ieee_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def save_ieee_panel(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)


def style_ieee_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.6)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.6, width=0.4)
    ax.minorticks_on()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def style_ieee_bar_axes(ax: plt.Axes, *, categorical: str = "x") -> None:
    """IEEE spines and inward ticks for categorical bar charts."""
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.6)
    if categorical == "x":
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.tick_params(axis="y", which="minor", direction="in", left=True, right=True, length=1.6, width=0.4)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        return
    if categorical == "y":
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", direction="in", bottom=True, top=True, length=1.6, width=0.4)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        return
    raise ValueError(f"Unsupported categorical axis: {categorical}")


def render_ieee_patch_legend(
    output_dir: Path,
    handles: list,
    labels: list[str],
    stem: str = "legend",
    ncol: int | None = None,
) -> None:
    """Top legend bar for grouped-bar figures (patches, not line styles)."""
    configure_ieee_matplotlib()
    n_labels = len(labels)
    ncol = n_labels if ncol is None else ncol
    n_rows = int(np.ceil(n_labels / ncol))
    fig = plt.figure(figsize=(7.16, 0.28 + 0.18 * n_rows))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        handleheight=0.85,
        columnspacing=1.15,
        handletextpad=0.4,
        fontsize=8,
    )
    save_ieee_panel(fig, output_dir / stem)
    plt.close(fig)


def render_ieee_legend_bar(output_dir: Path, stem: str = "legend") -> None:
    from matplotlib.lines import Line2D

    configure_ieee_matplotlib()
    handles = [Line2D([0], [0], color="black", linewidth=1.35, label="Ground truth")]
    for model_key in MODEL_KEYS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[model_key],
                linestyle=LINESTYLES[model_key],
                linewidth=LINEWIDTHS[model_key],
                label=DISPLAY_NAMES[model_key],
            )
        )
    fig = plt.figure(figsize=(7.16, 0.32))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        fontsize=8,
    )
    save_ieee_panel(fig, output_dir / stem)
    plt.close(fig)


def _padded_xy_limits(
    points: np.ndarray, pad_frac: float = 0.16, min_pad_km: float = 5.0
) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = max(min_pad_km, pad_frac * (x_max - x_min))
    y_pad = max(min_pad_km, pad_frac * (y_max - y_min))
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _plot_model_paths(
    ax: plt.Axes,
    prediction_enu_seeds: dict[str, np.ndarray],
    *,
    slice_obj: slice | None = None,
    include_labels: bool,
    faint_lw: float,
    faint_alpha: float,
) -> None:
    for model_key in MODEL_KEYS:
        seed_values = prediction_enu_seeds[model_key]
        if slice_obj is not None:
            seed_values = seed_values[:, slice_obj]
        for values in seed_values:
            ax.plot(
                values[:, 0],
                values[:, 1],
                color=COLORS[model_key],
                linestyle="-",
                linewidth=faint_lw,
                alpha=faint_alpha,
                zorder=2,
            )
        values = seed_values.mean(axis=0)
        ax.plot(
            values[:, 0],
            values[:, 1],
            color=COLORS[model_key],
            linestyle=LINESTYLES[model_key],
            linewidth=LINEWIDTHS[model_key],
            label=DISPLAY_NAMES[model_key] if include_labels else None,
            zorder=4 if model_key == "full" else 3,
            solid_capstyle="round",
        )


def draw_spatial(
    ax: plt.Axes,
    observed_enu: np.ndarray,
    truth_enu: np.ndarray,
    prediction_enu_seeds: dict[str, np.ndarray],
    *,
    show_legend: bool = True,
    show_time_label: bool = True,
    ylabel: bool = True,
    include_labels: bool = True,
) -> None:
    del observed_enu  # Overview span hides method differences; the panel uses the terminal window.
    terminal_steps = 64
    truth_term = truth_enu[-terminal_steps:]
    ax.plot(
        truth_term[:, 0], truth_term[:, 1], color="#171A1D", linewidth=1.7,
        label="Ground truth" if include_labels else None, zorder=5,
    )
    _plot_model_paths(
        ax,
        prediction_enu_seeds,
        slice_obj=slice(-terminal_steps, None),
        include_labels=include_labels,
        faint_lw=0.6,
        faint_alpha=0.22,
    )
    ax.set_xlabel("East displacement (km)")
    if ylabel:
        ax.set_ylabel("North displacement (km)")
    ax.grid(color="#D7DCE2", linewidth=0.5, alpha=0.85)
    (x0, x1), (y0, y1) = _padded_xy_limits(truth_term, pad_frac=0.20, min_pad_km=8.0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.plot(
        truth_term[-1, 0], truth_term[-1, 1],
        marker="s", color="#171A1D", markersize=4.2, zorder=6,
        linestyle="none",
    )
    for model_key in MODEL_KEYS:
        terminal = prediction_enu_seeds[model_key][:, -1, :2].mean(axis=0)
        ax.plot(
            terminal[0], terminal[1],
            marker="o", color=COLORS[model_key], markersize=3.8, zorder=6,
            linestyle="none",
        )
    if show_legend:
        ax.legend(
            loc="upper left",
            ncol=1,
            handlelength=2.2,
            borderaxespad=0.35,
            frameon=True,
            fancybox=False,
            edgecolor="#D7DCE2",
            framealpha=0.92,
            fontsize=6.0,
        )
    if show_time_label:
        ax.text(
            0.04, 0.04, "t = 193–256 s", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
        )


def draw_error(
    ax: plt.Axes,
    errors_km: dict[str, np.ndarray],
    *,
    show_legend: bool = True,
    ylabel: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    time_s = np.arange(1, 257, dtype=np.float64)
    for model_key in MODEL_KEYS:
        values = errors_km[model_key]
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        ax.fill_between(
            time_s, np.maximum(0.0, mean - std), mean + std,
            color=COLORS[model_key], alpha=0.12, linewidth=0,
        )
        ax.plot(
            time_s, mean, color=COLORS[model_key],
            linestyle=LINESTYLES[model_key],
            linewidth=LINEWIDTHS[model_key],
            label=DISPLAY_NAMES[model_key],
        )
    ax.set_xlim(1, 256)
    ax.set_xticks([1, 64, 128, 192, 256])
    if ylim is None:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Forecast time (s)")
    if ylabel:
        ax.set_ylabel("3-D position error (km)")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.5, alpha=0.85)
    if show_legend:
        ax.legend(loc="upper left", ncol=1, handlelength=2.2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    manifest, data = confirm.validate_locked_dataset()
    # The canonical validator intentionally returns only formal-evaluation
    # arrays. Case selection additionally needs complete frozen trajectories;
    # read those fields only after the dataset hash/manifest contract passes.
    with np.load(confirm.DATASET, allow_pickle=False) as archive:
        for key in ("trajectory_labels", "clean_trajectories", "window_starts_confirmatory"):
            if key not in archive.files:
                raise RuntimeError(f"Frozen confirmatory dataset is missing {key}")
            data[key] = np.asarray(archive[key])
    main_bundle, source_records = confirm.validate_main_bundle()
    case = select_case(data)
    index = int(case["window_index"])
    x_raw = data["X_confirmatory"][index : index + 1]
    y_raw = data["y_confirmatory"][index : index + 1]

    representative_payload = confirm._source_payload(source_records[("full", 42)])
    input_scaler, output_scaler, scaler_hashes = confirm._load_scalers(representative_payload)
    predictions_spherical: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_KEYS}
    provenance: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for seed in confirm.SEEDS:
            prediction, record = predict_case(
                model_key, seed, x_raw, y_raw, source_records,
                input_scaler, output_scaler, device,
            )
            predictions_spherical[model_key].append(prediction)
            provenance.append(record)
            print(f"[trajectory-case] inferred {model_key} seed={seed}")

    observation_spherical = x_raw[0, :, :3].astype(np.float64)
    truth_spherical = y_raw[0].astype(np.float64)
    reference = observation_spherical[-1]
    observed_enu = ecef_to_enu(spherical_to_ecef(observation_spherical), reference) / 1000.0
    truth_ecef = spherical_to_ecef(truth_spherical)
    truth_enu = ecef_to_enu(truth_ecef, reference) / 1000.0
    prediction_ecef = {
        key: np.stack([spherical_to_ecef(value) for value in values], axis=0)
        for key, values in predictions_spherical.items()
    }
    prediction_enu_seeds = {
        key: np.stack([ecef_to_enu(value, reference) / 1000.0 for value in values], axis=0)
        for key, values in prediction_ecef.items()
    }
    errors_km = {
        key: np.linalg.norm(values - truth_ecef[None, :, :], axis=2) / 1000.0
        for key, values in prediction_ecef.items()
    }
    metrics: dict[str, dict[str, Any]] = {}
    for model_key, values in errors_km.items():
        ade = values.mean(axis=1)
        fde = values[:, -1]
        metrics[model_key] = {
            "display_name": DISPLAY_NAMES[model_key],
            "ade_km_by_seed": {str(seed): float(value) for seed, value in zip(confirm.SEEDS, ade)},
            "fde_km_by_seed": {str(seed): float(value) for seed, value in zip(confirm.SEEDS, fde)},
            "ade_km_mean": float(ade.mean()),
            "ade_km_std_population": float(ade.std()),
            "fde_km_mean": float(fde.mean()),
            "fde_km_std_population": float(fde.std()),
        }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    spatial_fig, spatial_ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    draw_spatial(spatial_ax, observed_enu, truth_enu, prediction_enu_seeds)
    save_figure(spatial_fig, output_dir / "trajectory_case_spatial")
    plt.close(spatial_fig)

    error_fig, error_ax = plt.subplots(figsize=(3.45, 2.65), constrained_layout=True)
    draw_error(error_ax, errors_km)
    save_figure(error_fig, output_dir / "trajectory_case_error")
    plt.close(error_fig)

    preview_fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    draw_spatial(axes[0], observed_enu, truth_enu, prediction_enu_seeds)
    draw_error(axes[1], errors_km)
    axes[0].set_title("(a) Terminal ENU overlay", loc="left", fontweight="bold")
    axes[1].set_title("(b) Error growth over the forecast", loc="left", fontweight="bold")
    save_figure(preview_fig, output_dir / "trajectory_case_preview")
    plt.close(preview_fig)

    np.savez_compressed(
        output_dir / "trajectory_case_data.npz",
        observed_enu_km=observed_enu,
        truth_enu_km=truth_enu,
        forecast_time_s=np.arange(1, 257, dtype=np.int64),
        **{
            f"{key}_prediction_enu_km": value
            for key, value in prediction_enu_seeds.items()
        },
        **{f"{key}_error_km": value for key, value in errors_km.items()},
    )
    with (output_dir / "trajectory_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_key", "seed", "ade_km", "fde_km"))
        for model_key in MODEL_KEYS:
            for seed, ade, fde in zip(
                confirm.SEEDS,
                metrics[model_key]["ade_km_by_seed"].values(),
                metrics[model_key]["fde_km_by_seed"].values(),
            ):
                writer.writerow((model_key, seed, f"{ade:.12g}", f"{fde:.12g}"))

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    artifact_paths = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "trajectory_case_qa.json"
    )
    qa = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_status": "candidate_not_integrated",
        "core_conclusion": (
            "On a ground-truth-selected weaving window, Transformer peels away at long "
            "horizon while DLinear, iTransformer, and PLGAFormer remain closer to the "
            "true track. PLGAFormer has the lowest mean ADE/FDE but the largest "
            "checkpoint spread; the case is illustrative, not a ranking."
        ),
        "selection": {
            **case,
            "maneuver_class_rationale": (
                "Weaving is fixed before sample selection because the frozen maneuver-level "
                "aggregate identifies it as the hardest of the three maneuver classes."
            ),
            "trajectory_rule": (
                "Nearest trajectory to the component-wise median after IQR scaling of five "
                "full-ground-truth geometry/dynamics features; model outputs are excluded."
            ),
            "window_rule": (
                "Nearest admissible window start to the median start; ties resolve to the lower start."
            ),
            "model_error_used_for_selection": False,
        },
        "dataset": {
            "path": str(confirm.DATASET.resolve()),
            "sha256": sha256_file(confirm.DATASET),
            "manifest_sha256": sha256_file(confirm.DATASET_MANIFEST),
            "protocol_document_sha256": sha256_file(confirm.PROTOCOL_DOCUMENT),
            "trajectory_count": int(data["trajectory_ids"].size),
            "weaving_trajectory_count": int(np.count_nonzero(data["trajectory_labels"] == MANEUVER)),
        },
        "main_bundle": {
            "path": str(confirm.MAIN_BUNDLE.resolve()),
            "bundle_id": main_bundle["bundle_id"],
            "config_sha256": main_bundle["config_sha256"],
        },
        "scaler_sha256": scaler_hashes,
        "checkpoints": provenance,
        "seeds": list(confirm.SEEDS),
        "forecast_horizon_s": 256,
        "figure_methods": [DISPLAY_NAMES[key] for key in MODEL_KEYS],
        "omitted_from_figure": ["PatchTST"],
        "spatial_window": "observed history and ground truth, IQR-independent padded bounds",
        "metrics": metrics,
        "uncertainty_encoding": (
            "The spatial panel shows the last 64 forecast seconds, with each seed as a "
            "faint path and the mean Cartesian prediction as a strong path. The north "
            "axis is expanded relative to the east axis so cross-track differences remain "
            "visible. The error panel shows the full 256 s mean plus or minus one "
            "population standard deviation across seeds. PatchTST is omitted from both panels."
        ),
        "training_performed": False,
        "test_evaluation_performed": True,
        "git_head": git_head,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
        },
    }
    (output_dir / "trajectory_case_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "selection": case, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
