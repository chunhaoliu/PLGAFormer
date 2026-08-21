#!/usr/bin/env python3
"""Generate signed, non-cherry-picked trajectory and gate evidence for TAES."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "taes_submission_artifacts" / "generated" / "formal_v3"
MANEUVER_ORDER = ("longitudinal", "turning", "weaving")
DISPLAY_NAMES = {
    "longitudinal": "Longitudinal",
    "turning": "Turning",
    "weaving": "Weaving",
}
COLORS = {
    "truth": "#111111",
    "observation": "#6B7280",
    "plgaformer": "#0072B2",
    "transformer": "#D55E00",
    "rotating_3dof": "#009E73",
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
from utils.inference_protocol import predict_by_eval_protocol  # noqa: E402
from utils.formal_evidence import bundle_as_registry, load_evidence_bundle  # noqa: E402
from utils.seq2seq_protocol import align_source_position_scale  # noqa: E402
from utils.final_plgaformer import (  # noqa: E402
    final_evidence_signature,
    final_plgaformer_kwargs,
    resolve_final_plgaformer,
)

EARTH_RADIUS_M = float(HGVConfig.get_physical_constraints()["earth_radius"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--signature", default=os.getenv("HGV_EXP1_RUN_SIGNATURE", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tslib-root", type=Path, default=None)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path, requested_signature: str) -> tuple[dict, str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("bundle_kind") == "main":
            registry, signature = bundle_as_registry(path)
            if requested_signature.strip() and requested_signature.strip() != signature:
                raise RuntimeError("Requested signature does not match the formal-v3 Main bundle.")
            return registry, signature
        raise RuntimeError("Prediction analysis requires a formal-v3 Main Results manifest.")
    raise RuntimeError("Legacy Exp1 registry input is diagnostic-only and not accepted in paper mode.")


def find_run(registry: dict, signature: str, seed: int, model_type: str) -> dict:
    matches = [
        record
        for record in registry.get("runs", {}).values()
        if record.get("run_signature") == signature
        and int(record.get("seed", -1)) == int(seed)
        and record.get("model_type") == model_type
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one signed run for model={model_type}, seed={seed}; found {len(matches)}."
        )
    return matches[0]


def select_representative_windows(
    trajectory_ids: np.ndarray,
    maneuver_labels: np.ndarray,
    trajectory_metrics: dict[str, dict],
) -> dict[str, dict[str, float | int]]:
    """Select the median-error trajectory and its central test window per maneuver."""
    selected: dict[str, dict[str, float | int]] = {}
    for maneuver in MANEUVER_ORDER:
        maneuver_ids = sorted(
            set(int(value) for value in trajectory_ids[maneuver_labels == maneuver])
        )
        candidates = [
            (trajectory_id, float(trajectory_metrics[str(trajectory_id)]["ade"]))
            for trajectory_id in maneuver_ids
            if str(trajectory_id) in trajectory_metrics
        ]
        if not candidates:
            raise RuntimeError(f"No trajectory-level metrics are available for {maneuver}.")
        median_ade = float(np.median([value for _, value in candidates]))
        trajectory_id, trajectory_ade = min(
            candidates,
            key=lambda item: (abs(item[1] - median_ade), item[0]),
        )
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        window_index = int(indices[len(indices) // 2])
        selected[maneuver] = {
            "trajectory_id": trajectory_id,
            "window_index": window_index,
            "trajectory_ade_m": trajectory_ade,
            "maneuver_median_ade_m": median_ade,
            "windows_for_trajectory": int(indices.size),
        }
    return selected


def physical_kwargs(input_scaler, output_scaler) -> dict:
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


def load_selected_data(selection: dict[str, dict[str, float | int]]) -> tuple[dict, object, object]:
    dataset_path = get_dataset_npz_path(PROJECT_ROOT)
    with np.load(dataset_path, allow_pickle=False) as data:
        indices = np.asarray(
            [selection[maneuver]["window_index"] for maneuver in MANEUVER_ORDER],
            dtype=np.int64,
        )
        x_physical = np.asarray(data["X_test"][indices], dtype=np.float32)
        y_physical = np.asarray(data["y_test"][indices], dtype=np.float32)
        window_starts = np.asarray(data["window_starts_test"][indices], dtype=np.int64)

    input_scaler = joblib.load(get_input_scaler_path(PROJECT_ROOT))
    output_scaler = joblib.load(get_output_scaler_path(PROJECT_ROOT))
    x_scaled = input_scaler.transform(x_physical.reshape(-1, x_physical.shape[-1])).reshape(
        x_physical.shape
    )
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=y_physical.shape[-1],
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(y_physical.reshape(-1, y_physical.shape[-1])).reshape(
        y_physical.shape
    ).astype(np.float32, copy=False)
    return {
        "dataset_path": dataset_path,
        "x_physical": x_physical,
        "y_physical": y_physical,
        "x_scaled": x_scaled,
        "y_scaled": y_scaled,
        "window_starts": window_starts,
    }, input_scaler, output_scaler


def predict_models(
    data: dict,
    registry: dict,
    signature: str,
    seed: int,
    device: torch.device,
    input_scaler,
    output_scaler,
    bundle_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    x = torch.from_numpy(data["x_scaled"]).to(device)
    y = torch.from_numpy(data["y_scaled"]).to(device)
    label_len = int(HGVConfig.get_train_config().get("label_len", 128))
    predictions: dict[str, np.ndarray] = {}
    checkpoint_audit: dict[str, dict] = {}
    captured_gate: list[torch.Tensor] = []

    for model_type in ("plgaformer", "transformer", "rotating_3dof"):
        kwargs = physical_kwargs(input_scaler, output_scaler) if model_type in {
            "plgaformer",
            "rotating_3dof",
        } else None
        if model_type == "transformer":
            tslib_root = os.getenv("HGV_TSLIB_ROOT", "").strip()
            if not tslib_root:
                raise RuntimeError(
                    "Prediction analysis reconstruction of Transformer requires "
                    "--tslib-root or HGV_TSLIB_ROOT."
                )
            kwargs = {"source": "tslib", "tslib_root": tslib_root}
        if model_type == "plgaformer":
            kwargs.update(final_plgaformer_kwargs())
        model = create_registered_model(
            model_type=model_type,
            input_dim=6,
            device=device,
            plgaformer_kwargs=kwargs,
        )
        if model_type in {"plgaformer", "transformer"}:
            record = find_run(registry, signature, seed, model_type)
            if model_type == "plgaformer":
                checkpoint, _, final_audit = resolve_final_plgaformer(seed, bundle_path=bundle_path)
            else:
                checkpoint = Path(record.get("checkpoint_path", "")).resolve()
                final_audit = {}
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing signed checkpoint: {checkpoint}")
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            checkpoint_audit[model_type] = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "completed_at": record.get("completed_at"),
                **final_audit,
            }
        else:
            checkpoint_audit[model_type] = {"path": None, "source": "parameter-free analytical model"}

        if model_type == "plgaformer":
            original = model._scheduled_prior_weight

            def capture_weight(kinematic_prior, output, decoder_output, future_mask):
                weight = original(kinematic_prior, output, decoder_output, future_mask)
                captured_gate.append(weight.detach().cpu())
                return weight

            model._scheduled_prior_weight = capture_weight

        model.eval()
        with torch.no_grad():
            prediction = predict_by_eval_protocol(
                model=model,
                x=x,
                y_true_scaled=y,
                pred_length=y.size(1),
                device=device,
                eval_protocol="source_context_decoder",
                eval_ar_seed_mode="zero",
                label_len=label_len,
            )
        predictions[model_type] = output_scaler.inverse_transform(
            prediction.detach().cpu().numpy().reshape(-1, 3)
        ).reshape(prediction.shape)
        del model

    if len(captured_gate) != 1:
        raise RuntimeError(f"Expected one PLGAFormer gate capture; found {len(captured_gate)}.")
    gate = captured_gate[0][:, -data["y_scaled"].shape[1] :, :].numpy()
    return predictions, gate, checkpoint_audit


def spherical_to_ecef(values: np.ndarray) -> np.ndarray:
    radius, longitude, latitude = np.moveaxis(values[..., :3], -1, 0)
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            radius * cos_latitude * np.cos(longitude),
            radius * cos_latitude * np.sin(longitude),
            radius * np.sin(latitude),
        ),
        axis=-1,
    )


def to_local_enu(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_radius, ref_lon, ref_lat = reference[:3]
    ref_ecef = spherical_to_ecef(reference[None, :3])[0]
    delta = spherical_to_ecef(values) - ref_ecef
    rotation = np.asarray(
        [
            [-np.sin(ref_lon), np.cos(ref_lon), 0.0],
            [
                -np.sin(ref_lat) * np.cos(ref_lon),
                -np.sin(ref_lat) * np.sin(ref_lon),
                np.cos(ref_lat),
            ],
            [
                np.cos(ref_lat) * np.cos(ref_lon),
                np.cos(ref_lat) * np.sin(ref_lon),
                np.sin(ref_lat),
            ],
        ]
    )
    return delta @ rotation.T


def _style_axis(axis) -> None:
    axis.grid(True, color="#D1D5DB", linewidth=0.45, alpha=0.75)
    axis.tick_params(labelsize=7)


def plot_trajectory_figure(
    output_path: Path,
    data: dict,
    predictions: dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(7.15, 6.15))
    model_labels = {
        "plgaformer": "PLGAFormer",
        "transformer": "Transformer",
        "rotating_3dof": "Identified 3-DOF",
    }
    for row, maneuver in enumerate(MANEUVER_ORDER):
        reference = data["x_physical"][row, -1, :3]
        observation = to_local_enu(data["x_physical"][row, :, :3], reference) / 1000.0
        truth = to_local_enu(data["y_physical"][row], reference) / 1000.0
        local_predictions = {
            name: to_local_enu(values[row], reference) / 1000.0
            for name, values in predictions.items()
        }

        axis_3d = fig.add_subplot(3, 3, row * 3 + 1, projection="3d")
        axis_3d.plot(*observation.T, color=COLORS["observation"], linewidth=1.2)
        axis_3d.plot(*truth.T, color=COLORS["truth"], linewidth=1.6)
        for name, values in local_predictions.items():
            axis_3d.plot(*values.T, color=COLORS[name], linewidth=1.0)
        axis_3d.set_xlabel("E (km)", fontsize=7, labelpad=0)
        axis_3d.set_ylabel("N (km)", fontsize=7, labelpad=0)
        axis_3d.set_zlabel("U (km)", fontsize=7, labelpad=0)
        axis_3d.set_title(f"({chr(97 + row * 3)}) {DISPLAY_NAMES[maneuver]}: 3-D", fontsize=8)
        axis_3d.tick_params(labelsize=6, pad=0)
        axis_3d.view_init(elev=24, azim=-57)

        axis_ground = fig.add_subplot(3, 3, row * 3 + 2)
        axis_ground.plot(observation[:, 0], observation[:, 1], color=COLORS["observation"], linewidth=1.2)
        axis_ground.plot(truth[:, 0], truth[:, 1], color=COLORS["truth"], linewidth=1.6)
        for name, values in local_predictions.items():
            axis_ground.plot(values[:, 0], values[:, 1], color=COLORS[name], linewidth=1.0)
        axis_ground.set_xlabel("East (km)", fontsize=7)
        axis_ground.set_ylabel("North (km)", fontsize=7)
        axis_ground.set_aspect("equal", adjustable="datalim")
        axis_ground.set_title(f"({chr(98 + row * 3)}) Ground track", fontsize=8)
        _style_axis(axis_ground)

        axis_altitude = fig.add_subplot(3, 3, row * 3 + 3)
        observed_time = np.arange(-data["x_physical"].shape[1] + 1, 1)
        future_time = np.arange(1, data["y_physical"].shape[1] + 1)
        axis_altitude.plot(
            observed_time,
            (data["x_physical"][row, :, 0] - EARTH_RADIUS_M) / 1000.0,
            color=COLORS["observation"],
            linewidth=1.2,
        )
        axis_altitude.plot(
            future_time,
            (data["y_physical"][row, :, 0] - EARTH_RADIUS_M) / 1000.0,
            color=COLORS["truth"],
            linewidth=1.6,
        )
        for name, values in predictions.items():
            axis_altitude.plot(
                future_time,
                (values[row, :, 0] - EARTH_RADIUS_M) / 1000.0,
                color=COLORS[name],
                linewidth=1.0,
            )
        axis_altitude.axvline(0, color="#9CA3AF", linewidth=0.7, linestyle=":")
        axis_altitude.set_xlabel("Time from forecast origin (s)", fontsize=7)
        axis_altitude.set_ylabel("Altitude (km)", fontsize=7)
        axis_altitude.set_title(f"({chr(99 + row * 3)}) Altitude", fontsize=8)
        _style_axis(axis_altitude)

    handles = [
        plt.Line2D([0], [0], color=COLORS["observation"], lw=1.5, label="Observation"),
        plt.Line2D([0], [0], color=COLORS["truth"], lw=1.8, label="Truth"),
    ]
    handles.extend(
        plt.Line2D([0], [0], color=COLORS[name], lw=1.3, label=label)
        for name, label in model_labels.items()
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.08, wspace=0.40, hspace=0.48)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_error_gate_figure(
    output_path: Path,
    data: dict,
    predictions: dict[str, np.ndarray],
    gate: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 3.65), sharex="col")
    future_time = np.arange(1, data["y_physical"].shape[1] + 1)
    model_labels = {
        "plgaformer": "PLGAFormer",
        "transformer": "Transformer",
        "rotating_3dof": "Identified 3-DOF",
    }
    true_ecef = spherical_to_ecef(data["y_physical"])
    for col, maneuver in enumerate(MANEUVER_ORDER):
        axis_error = axes[0, col]
        for name, values in predictions.items():
            error = np.linalg.norm(spherical_to_ecef(values)[col] - true_ecef[col], axis=-1)
            axis_error.plot(future_time, error / 1000.0, color=COLORS[name], linewidth=1.2)
        axis_error.set_title(f"({chr(97 + col)}) {DISPLAY_NAMES[maneuver]}", fontsize=8)
        axis_error.set_ylabel("Displacement error (km)" if col == 0 else "", fontsize=7)
        _style_axis(axis_error)

        axis_gate = axes[1, col]
        axis_gate.fill_between(
            future_time,
            gate[col].min(axis=-1),
            gate[col].max(axis=-1),
            color=COLORS["plgaformer"],
            alpha=0.16,
            linewidth=0,
        )
        axis_gate.plot(
            future_time,
            gate[col].mean(axis=-1),
            color=COLORS["plgaformer"],
            linewidth=1.4,
        )
        axis_gate.set_ylim(0.0, 1.0)
        axis_gate.set_xlabel("Forecast horizon (s)", fontsize=7)
        axis_gate.set_ylabel("Prior weight" if col == 0 else "", fontsize=7)
        axis_gate.set_title(f"({chr(100 + col)}) Gate mean and channel range", fontsize=8)
        _style_axis(axis_gate)

    handles = [
        plt.Line2D([0], [0], color=COLORS[name], lw=1.4, label=label)
        for name, label in model_labels.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.17, wspace=0.30, hspace=0.38)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_diagnostics(
    output_path: Path,
    registry: dict,
    signature: str,
    bundle_path: Path,
) -> None:
    histories = [
        resolve_final_plgaformer(seed, bundle_path=bundle_path)[1].get("training_history", {})
        for seed in (42, 123, 456)
    ]
    if any(not history.get("train_losses") or not history.get("val_losses") for history in histories):
        raise RuntimeError("Formal-v3 PLGAFormer records lack complete training histories.")
    gate_history = [history.get("physics_gate_history", []) for history in histories]
    if not all(gate_history):
        raise RuntimeError("Formal-v3 PLGAFormer histories lack physics gate diagnostics.")
    common_epochs = min(
        min(len(history["train_losses"]), len(history["val_losses"]))
        for history in histories
    )
    common_epochs = min(common_epochs, min(len(history) for history in gate_history))
    if common_epochs <= 0:
        raise RuntimeError("Formal-v3 PLGAFormer records have empty training histories.")
    epochs = np.arange(1, common_epochs + 1)
    train_loss = np.asarray([history["train_losses"][:common_epochs] for history in histories], dtype=float)
    val_loss = np.asarray([history["val_losses"][:common_epochs] for history in histories], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(5.25, 2.20))
    for values, label, color in (
        (train_loss, "Training", "#6B7280"),
        (val_loss, "Validation", COLORS["plgaformer"]),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axes[0].plot(epochs, mean, color=color, linewidth=1.3, label=label)
        axes[0].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch", fontsize=7)
    axes[0].set_ylabel("Objective", fontsize=7)
    axes[0].set_title("(a) Optimization", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    _style_axis(axes[0])

    gate_fields = (
        ("nominal_prior_weight_32", "32 s", "#009E73"),
        ("nominal_prior_weight_128", "128 s", "#E69F00"),
        ("nominal_prior_weight_256", "256 s", "#CC79A7"),
    )
    for field, label, color in gate_fields:
        values = np.asarray(
            [[epoch[field] for epoch in history[:common_epochs]] for history in gate_history],
            dtype=float,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axes[1].plot(epochs, mean, color=color, linewidth=1.3, label=label)
        axes[1].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14)
    axes[1].set_ylim(0.68, 1.01)
    axes[1].set_xlabel("Epoch", fontsize=7)
    axes[1].set_ylabel("Zero-disagreement weight", fontsize=7)
    axes[1].set_title("(b) Nominal prior schedule", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    _style_axis(axes[1])

    fig.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.22, wspace=0.38)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tslib_root is not None:
        tslib_root = args.tslib_root.expanduser().resolve()
        if not tslib_root.is_dir():
            raise FileNotFoundError(f"TSLib checkout is missing: {tslib_root}")
        os.environ["HGV_TSLIB_ROOT"] = str(tslib_root)
    bundle = load_evidence_bundle(args.registry.resolve())
    registry, signature = load_registry(args.registry.resolve(), args.signature)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT / "hgv_multiregime_state_v2_1" / signature
    _, final_payload, _ = resolve_final_plgaformer(args.seed, bundle_path=args.registry)
    trajectory_metrics = final_payload["eval_results"]["256"]["trajectory_metrics"]
    final_signature, final_signature_audit = final_evidence_signature(signature, bundle_path=args.registry)

    dataset_path = get_dataset_npz_path(PROJECT_ROOT)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        trajectory_ids = np.asarray(dataset["trajectory_ids_test"], dtype=np.int64)
        maneuver_labels = np.asarray(dataset["maneuver_labels_test"]).astype(str)
    selection = select_representative_windows(
        trajectory_ids,
        maneuver_labels,
        trajectory_metrics,
    )
    data, input_scaler, output_scaler = load_selected_data(selection)
    device = torch.device(args.device)
    predictions, gate, checkpoint_audit = predict_models(
        data,
        registry,
        signature,
        args.seed,
        device,
        input_scaler,
        output_scaler,
        args.registry,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "fig_prediction_analysis.pdf"
    gate_path = args.output_dir / "fig_error_gate_analysis.pdf"
    training_path = args.output_dir / "fig_training_diagnostics.pdf"
    plot_trajectory_figure(trajectory_path, data, predictions)
    plot_error_gate_figure(gate_path, data, predictions, gate)
    plot_training_diagnostics(training_path, registry, signature, args.registry)

    for index, maneuver in enumerate(MANEUVER_ORDER):
        selection[maneuver]["window_start_s"] = int(data["window_starts"][index])
        selection[maneuver]["gate_mean_32"] = float(gate[index, 31].mean())
        selection[maneuver]["gate_mean_128"] = float(gate[index, 127].mean())
        selection[maneuver]["gate_mean_256"] = float(gate[index, 255].mean())
    manifest = {
        "artifact_kind": "formal_v3_taes_prediction_analysis",
        "bundle_id": bundle.get("bundle_id"),
        "config_sha256": bundle.get("config_sha256"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "run_signature": final_signature,
        "base_exp1_signature": signature,
        "final_signature_audit": final_signature_audit,
        "seed": args.seed,
        "selection_rule": (
            "Within each maneuver family, choose the held-out source trajectory whose "
            "seed-42 PLGAFormer trajectory-level 256-s ADE is closest to that family's "
            "median, then choose the central extracted window of that trajectory."
        ),
        "dataset": str(data["dataset_path"].resolve()),
        "selection": selection,
        "checkpoints": checkpoint_audit,
        "outputs": [
            str(trajectory_path.resolve()),
            str(gate_path.resolve()),
            str(training_path.resolve()),
        ],
    }
    manifest_path = args.output_dir / "prediction_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {trajectory_path}")
    print(f"Wrote {gate_path}")
    print(f"Wrote {training_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
