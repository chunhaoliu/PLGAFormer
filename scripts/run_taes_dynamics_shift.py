#!/usr/bin/env python3
"""Evaluate frozen formal models under controlled simulator-parameter shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp5_ood_dynamics"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
)
MANUSCRIPT_DIR = PROJECT_ROOT / "experiments" / "taes_submission_artifacts" / "generated"
REGISTRY_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "main_run_set_manifest.json"
)
MODEL_KEY_TO_TYPE = {
    "baseline": "transformer",
    "full": "plgaformer",
    "rotating_3dof": "rotating_3dof",
}
DISPLAY_NAMES = {
    "transformer": "Transformer",
    "plgaformer": "PLGAFormer",
    "rotating_3dof": "Rotating-Earth 3-DOF",
}
CONDITION_LABELS = {
    "nominal": "Nominal",
    "aero_shift": "$C_L-10\\%$, $C_D+10\\%$",
    "ballistic_shift": "$m+15\\%$, $S-10\\%$",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_generator import DCBNN_HGV_Simulator  # noqa: E402
from data_generation.data_paths import (  # noqa: E402
    get_dataset_npz_path,
    get_input_scaler_path,
    get_output_scaler_path,
    get_raw_trajectories_npz_path,
)
from models import HGVConfig  # noqa: E402
from models.model_factory import create_registered_model  # noqa: E402
from scripts.generate_taes_prediction_analysis import find_run, physical_kwargs, sha256_file  # noqa: E402
from utils.inference_protocol import predict_by_eval_protocol  # noqa: E402
from utils.formal_evidence import (  # noqa: E402
    bundle_as_registry,
    load_formal_config,
    validate_evidence_bundle,
)
from utils.seq2seq_protocol import align_source_position_scale  # noqa: E402
from utils.final_plgaformer import (  # noqa: E402
    final_evidence_signature,
    final_plgaformer_kwargs,
    resolve_final_plgaformer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "formal_v3.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--manuscript-dir", type=Path, default=None)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args(argv)


def _formal_robustness_values(config) -> tuple[dict[str, dict[str, float]], tuple[str, ...], tuple[int, ...]]:
    study = config["studies"]["robustness"]
    conditions = {}
    for item in study.get("dynamics_shift_conditions", []):
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("Every formal robustness dynamics condition needs a name and scales.")
        name = str(item["name"])
        conditions[name] = {
            key: float(item[key])
            for key in ("mass_scale", "area_scale", "cl_scale", "cd_scale")
        }
    if not conditions or "nominal" not in conditions:
        raise ValueError("Formal robustness config must include a nominal condition.")
    model_types = tuple(
        MODEL_KEY_TO_TYPE[str(key)]
        for key in study.get("model_keys", [])
        if str(key) in MODEL_KEY_TO_TYPE
    )
    model_seeds = tuple(int(seed) for seed in study.get("model_seeds", []))
    if not model_types or not model_seeds:
        raise ValueError("Formal robustness config must declare model keys and seeds.")
    return conditions, model_types, model_seeds


def central_window_rows(trajectory_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one central row and its trajectory ID for each held-out source trajectory."""
    rows = []
    ids = []
    for trajectory_id in sorted(np.unique(trajectory_ids)):
        candidates = np.flatnonzero(trajectory_ids == trajectory_id)
        rows.append(int(candidates[len(candidates) // 2]))
        ids.append(int(trajectory_id))
    return np.asarray(rows, dtype=np.int64), np.asarray(ids, dtype=np.int64)


def _sha256_json(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _simulate_shifted_trajectory(
    initial_state: np.ndarray,
    maneuver: str,
    parameters: dict[str, float],
    *,
    points_per_trajectory: int,
    sampling_interval_s: float,
) -> np.ndarray:
    simulator = DCBNN_HGV_Simulator()
    simulator.m *= float(parameters["mass_scale"])
    simulator.S *= float(parameters["area_scale"])
    base_aerodynamics = simulator.aerodynamic_coefficients

    def shifted_aerodynamics(alpha, bank, mach):
        cl, cd, cy = base_aerodynamics(alpha, bank, mach)
        return (
            float(parameters["cl_scale"]) * cl,
            float(parameters["cd_scale"]) * cd,
            cy,
        )

    simulator.aerodynamic_coefficients = shifted_aerodynamics
    initial = {
        "r": float(initial_state[0]),
        "lambda": float(initial_state[1]),
        "phi": float(initial_state[2]),
        "V": float(initial_state[3]),
        "gamma": float(initial_state[4]),
        "psi": float(initial_state[5]),
    }
    _, trajectory = simulator.simulate_trajectory(
        initial,
        maneuver,
        duration=float(points_per_trajectory - 1) * sampling_interval_s,
        sampling_interval_s=sampling_interval_s,
        num_points=points_per_trajectory,
    )
    if trajectory is None or trajectory.shape != (points_per_trajectory, 6):
        raise RuntimeError(f"Shifted simulation failed for maneuver={maneuver}.")
    return np.asarray(trajectory, dtype=np.float32)


def load_or_generate_shifted_trajectories(
    cache_path: Path,
    force_regenerate: bool,
    *,
    config,
    conditions: dict[str, dict[str, float]],
    seq_len: int,
    pred_len: int,
) -> tuple[dict[str, np.ndarray], dict]:
    raw_path = get_raw_trajectories_npz_path(PROJECT_ROOT)
    dataset_path = get_dataset_npz_path(PROJECT_ROOT)
    with np.load(dataset_path, allow_pickle=False) as dataset:
        central_rows, test_ids = central_window_rows(dataset["trajectory_ids_test"])
        window_starts = np.asarray(dataset["window_starts_test"][central_rows], dtype=np.int64)
        processed_x = np.asarray(dataset["X_test"][central_rows], dtype=np.float32)
        processed_y = np.asarray(dataset["y_test"][central_rows], dtype=np.float32)
    with np.load(raw_path, allow_pickle=False) as raw:
        raw_trajectories = np.asarray(raw["trajectories"], dtype=np.float32)
        initial_states = np.asarray(raw["initial_states"], dtype=np.float32)
        labels = np.asarray(raw["trajectory_labels"]).astype(str)
        raw_ids = np.asarray(raw["trajectory_ids"], dtype=np.int64)

    if not np.array_equal(raw_ids, np.arange(len(raw_ids), dtype=np.int64)):
        raise RuntimeError("Raw trajectory IDs are not direct row indices.")
    nominal_x, nominal_y = build_windows(
        raw_trajectories[test_ids], window_starts, seq_len=seq_len, pred_len=pred_len
    )
    if not np.array_equal(nominal_x, processed_x) or not np.array_equal(nominal_y, processed_y):
        raise RuntimeError(
            "Raw trajectories do not reconstruct the processed central test windows exactly."
        )
    cache_signature = _sha256_json(
        {
            "conditions": conditions,
            "test_ids": test_ids.tolist(),
            "window_starts": window_starts.tolist(),
            "raw_sha256": sha256_file(raw_path),
        }
    )
    if cache_path.is_file() and not force_regenerate:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["cache_signature"].item()) == cache_signature:
                trajectories = {
                    condition: np.asarray(cached[condition], dtype=np.float32)
                    for condition in conditions
                }
                metadata = {
                    "cache_signature": cache_signature,
                    "test_ids": test_ids,
                    "window_starts": window_starts,
                    "raw_path": raw_path,
                    "dataset_path": dataset_path,
                    "cache_reused": True,
                }
                return trajectories, metadata

    trajectories = {"nominal": raw_trajectories[test_ids]}
    for condition, parameters in conditions.items():
        if condition == "nominal":
            continue
        generated = []
        print(f"[dynamics-shift] generating {condition} for {len(test_ids)} trajectories", flush=True)
        for index, trajectory_id in enumerate(test_ids, 1):
            generated.append(
                _simulate_shifted_trajectory(
                    initial_states[trajectory_id],
                    labels[trajectory_id],
                    parameters,
                    points_per_trajectory=int(config["dataset"]["points_per_trajectory"]),
                    sampling_interval_s=float(config["dataset"]["sampling_interval_s"]),
                )
            )
            if index % 10 == 0 or index == len(test_ids):
                print(f"  {condition}: {index}/{len(test_ids)}", flush=True)
        trajectories[condition] = np.stack(generated)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_signature=np.asarray(cache_signature),
        test_ids=test_ids,
        window_starts=window_starts,
        **trajectories,
    )
    metadata = {
        "cache_signature": cache_signature,
        "test_ids": test_ids,
        "window_starts": window_starts,
        "raw_path": raw_path,
        "dataset_path": dataset_path,
        "cache_reused": False,
    }
    return trajectories, metadata


def build_windows(
    trajectories: np.ndarray,
    window_starts: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [trajectory[start : start + seq_len, :6] for trajectory, start in zip(trajectories, window_starts)]
    )
    y = np.stack(
        [
            trajectory[start + seq_len : start + seq_len + pred_len, :3]
            for trajectory, start in zip(trajectories, window_starts)
        ]
    )
    if x.shape[1:] != (seq_len, 6) or y.shape[1:] != (pred_len, 3):
        raise RuntimeError(f"Invalid shifted window shapes: X={x.shape}, y={y.shape}.")
    return x.astype(np.float32), y.astype(np.float32)


def scale_windows(x: np.ndarray, y: np.ndarray, input_scaler, output_scaler):
    x_scaled = input_scaler.transform(x.reshape(-1, 6)).reshape(x.shape)
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=3,
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(y.reshape(-1, 3)).reshape(y.shape).astype(
        np.float32, copy=False
    )
    return x_scaled, y_scaled


def spherical_to_ecef(values: np.ndarray) -> np.ndarray:
    radius = values[..., 0]
    longitude = values[..., 1]
    latitude = values[..., 2]
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            radius * cos_latitude * np.cos(longitude),
            radius * cos_latitude * np.sin(longitude),
            radius * np.sin(latitude),
        ),
        axis=-1,
    )


def evaluate_prediction(prediction: np.ndarray, truth: np.ndarray) -> dict:
    errors = np.linalg.norm(spherical_to_ecef(prediction) - spherical_to_ecef(truth), axis=-1)
    trajectory_ade = errors.mean(axis=1)
    trajectory_fde = errors[:, -1]
    return {
        "ade_m": float(trajectory_ade.mean()),
        "fde_m": float(trajectory_fde.mean()),
        "trajectory_count": int(len(trajectory_ade)),
        "trajectory_ade_m": trajectory_ade.tolist(),
        "trajectory_fde_m": trajectory_fde.tolist(),
    }


def predict(
    model,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    output_scaler,
    device: torch.device,
    batch_size: int,
    *,
    pred_len: int,
    eval_protocol: str,
    eval_ar_seed_mode: str,
    label_len: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_scaled)),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = predict_by_eval_protocol(
                model=model,
                x=x_batch,
                y_true_scaled=y_batch,
                pred_length=pred_len,
                device=device,
                eval_protocol=eval_protocol,
                eval_ar_seed_mode=eval_ar_seed_mode,
                label_len=label_len,
            )
            rows.append(prediction.detach().cpu().numpy())
    scaled = np.concatenate(rows, axis=0)
    return output_scaler.inverse_transform(scaled.reshape(-1, 3)).reshape(scaled.shape)


def aggregate_seed_metrics(seed_metrics: list[tuple[int, dict]]) -> dict:
    trajectory_ade = np.mean(
        np.asarray([row["trajectory_ade_m"] for _, row in seed_metrics], dtype=float),
        axis=0,
    )
    trajectory_fde = np.mean(
        np.asarray([row["trajectory_fde_m"] for _, row in seed_metrics], dtype=float),
        axis=0,
    )
    seed_ade = np.asarray([row["ade_m"] for _, row in seed_metrics], dtype=float)
    seed_fde = np.asarray([row["fde_m"] for _, row in seed_metrics], dtype=float)
    return {
        "model_seeds": [seed for seed, _ in seed_metrics],
        "ade_m": float(seed_ade.mean()),
        "ade_std_m": float(seed_ade.std(ddof=1)) if len(seed_ade) > 1 else 0.0,
        "fde_m": float(seed_fde.mean()),
        "fde_std_m": float(seed_fde.std(ddof=1)) if len(seed_fde) > 1 else 0.0,
        "trajectory_count": int(len(trajectory_ade)),
        "trajectory_ade_m": trajectory_ade.tolist(),
        "trajectory_fde_m": trajectory_fde.tolist(),
    }


def render_table(results: dict, conditions: dict[str, dict[str, float]], model_types: tuple[str, ...], pred_len: int) -> str:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{Frozen-model {pred_len} s errors under simulator-parameter shifts. "
        r"Values are mean$\pm$sample standard deviation across three model seeds in "
        r"kilometers; the analytical baseline is deterministic.}",
        r"\label{tab:dynamics_shift}",
        r"\scriptsize",
        r"\resizebox{0.72\textwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Simulator condition & Method & ADE & FDE \\",
        r"\midrule",
    ]
    for condition_index, condition in enumerate(conditions):
        if condition_index:
            lines.append(r"\addlinespace[1pt]")
        for model_type in model_types:
            row = results[condition][model_type]
            label = DISPLAY_NAMES[model_type]
            if model_type == "plgaformer":
                label = rf"\textbf{{{label}}}"
            condition_label = CONDITION_LABELS.get(condition, condition.replace("_", " ").title())
            lines.append(
                f"{condition_label} & {label} & "
                f"{row['ade_m'] / 1000:.3f}$\\pm${row['ade_std_m'] / 1000:.3f} & "
                f"{row['fde_m'] / 1000:.3f}$\\pm${row['fde_std_m'] / 1000:.3f} "
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_formal_config(args.config)
    verification = validate_evidence_bundle(args.registry, config, expected_kind="main")
    if not verification["passed"]:
        codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
        raise RuntimeError("Formal-v3 Main Results bundle failed verification: " + ", ".join(codes))
    bundle = verification["bundle"]
    conditions, model_types, model_seeds = _formal_robustness_values(config)
    registry, signature = bundle_as_registry(args.registry)
    if args.manuscript_dir is None:
        args.manuscript_dir = MANUSCRIPT_DIR / "formal_v3" / "hgv_multiregime_state_v2_1" / str(bundle["bundle_id"])

    cache_path = args.results_dir / "shifted_test_trajectories.npz"
    seq_len = int(config["task"]["seq_len"])
    pred_len = int(config["task"]["pred_len"])
    trajectories, simulation_metadata = load_or_generate_shifted_trajectories(
        cache_path,
        args.force_regenerate,
        config=config,
        conditions=conditions,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    input_scaler = joblib.load(get_input_scaler_path(PROJECT_ROOT))
    output_scaler = joblib.load(get_output_scaler_path(PROJECT_ROOT))
    device = torch.device(args.device)
    results = {}
    checkpoint_audits = {}

    for condition, condition_trajectories in trajectories.items():
        print(f"[dynamics-shift] evaluating {condition}", flush=True)
        x, y = build_windows(
            condition_trajectories,
            simulation_metadata["window_starts"],
            seq_len=seq_len,
            pred_len=pred_len,
        )
        x_scaled, y_scaled = scale_windows(x, y, input_scaler, output_scaler)
        results[condition] = {}
        for model_type in model_types:
            seeds = model_seeds if model_type != "rotating_3dof" else (model_seeds[0],)
            seed_metrics = []
            for seed in seeds:
                kwargs = physical_kwargs(input_scaler, output_scaler) if model_type in {
                    "plgaformer",
                    "rotating_3dof",
                } else None
                if model_type == "plgaformer":
                    kwargs.update(final_plgaformer_kwargs())
                model = create_registered_model(
                    model_type=model_type,
                    input_dim=6,
                    device=device,
                    plgaformer_kwargs=kwargs,
                )
                if model_type != "rotating_3dof":
                    record = find_run(registry, signature, seed, model_type)
                    if model_type == "plgaformer":
                        checkpoint, _, final_audit = resolve_final_plgaformer(seed, bundle_path=args.registry)
                    else:
                        checkpoint = Path(record["checkpoint_path"]).resolve()
                        final_audit = {}
                    model.load_state_dict(
                        torch.load(checkpoint, map_location=device, weights_only=True),
                        strict=True,
                    )
                    checkpoint_audits.setdefault(model_type, {})[str(seed)] = {
                        "path": str(checkpoint),
                        "sha256": sha256_file(checkpoint),
                        "run_signature": signature,
                        **final_audit,
                    }
                prediction = predict(
                    model,
                    x_scaled,
                    y_scaled,
                    output_scaler,
                    device,
                    args.batch_size,
                    pred_len=pred_len,
                    eval_protocol=str(config["task"]["eval_protocol"]),
                    eval_ar_seed_mode=str(config["task"]["eval_ar_seed_mode"]),
                    label_len=int(config["task"]["label_len"]),
                )
                seed_metrics.append((seed, evaluate_prediction(prediction, y)))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results[condition][model_type] = aggregate_seed_metrics(seed_metrics)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    final_signature, final_signature_audit = final_evidence_signature(signature, bundle_path=args.registry)
    payload = {
        "artifact_kind": "formal_v3_taes_dynamics_shift",
        "bundle_id": bundle.get("bundle_id"),
        "config_sha256": bundle.get("config_sha256"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "experiment": str(config["studies"]["robustness"]["study_id"]),
        "study_id": str(config["studies"]["robustness"]["study_id"]),
        "run_signature": final_signature,
        "base_exp1_signature": signature,
        "final_signature_audit": final_signature_audit,
        "conditions": conditions,
        "model_types": list(model_types),
        "model_seeds": list(model_seeds),
        "seq_len": seq_len,
        "pred_len": pred_len,
        "test_trajectory_count": int(len(simulation_metadata["test_ids"])),
        "window_protocol": f"one central {seq_len}/{pred_len} window per held-out source trajectory",
        "test_trajectory_ids": simulation_metadata["test_ids"].tolist(),
        "window_starts": simulation_metadata["window_starts"].tolist(),
        "simulation_cache_signature": simulation_metadata["cache_signature"],
        "simulation_cache_reused": simulation_metadata["cache_reused"],
        "checkpoint_audits": checkpoint_audits,
        "results": results,
    }
    result_path = args.results_dir / "dynamics_shift_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.manuscript_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.manuscript_dir / "table_dynamics_shift.tex"
    table_path.write_text(render_table(results, conditions, model_types, pred_len), encoding="utf-8")
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
