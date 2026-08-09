#!/usr/bin/env python3
"""Generate a separate PIT-aligned radar-tracking HGV dataset.

The default command is a nine-trajectory pilot.  Passing ``--formal`` selects
the planned 1,000-trajectory protocol.  Neither mode overwrites the existing
``trajectory_level_v1`` dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.data_generator import DCBNN_HGV_Simulator
from data_generation.pit_radar_protocol import (
    PIT_RADAR_PROTOCOL,
    RadarConfig,
    TrackingConfig,
    assemble_pit_radar_dataset,
    track_noisy_radar_states,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "pit_aligned_radar_pilot.npz"
)
FORMAL_OUTPUT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "pit_aligned_radar_dataset.npz"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="Generate 1,000 trajectories.")
    parser.add_argument("--force", action="store_true", help="Replace an existing candidate output.")
    parser.add_argument("--resume", action="store_true", help="Resume a deterministic generation checkpoint.")
    parser.add_argument(
        "--max-new-trajectories",
        type=int,
        default=None,
        help="Stop after this many newly accepted trajectories and save a checkpoint.",
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--sampling-interval-s", type=float, default=2.0)
    parser.add_argument("--integration-method", choices=("DOP853", "RK45"), default="DOP853")
    parser.add_argument("--integration-rtol", type=float, default=1e-8)
    parser.add_argument("--integration-atol", type=float, default=1e-10)
    parser.add_argument("--internal-max-step-s", type=float, default=4.0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--pred-len", type=int, default=256)
    parser.add_argument("--tracking-burn-in-steps", type=int, default=32)
    parser.add_argument("--train-origins-per-trajectory", type=int, default=16)
    parser.add_argument("--eval-window-stride", type=int, default=5)
    parser.add_argument("--max-attempt-factor", type=int, default=8)
    parser.add_argument("--initial-speed-min-mps", type=float, default=6_300.0)
    parser.add_argument("--initial-speed-max-mps", type=float, default=7_000.0)
    parser.add_argument("--min-glide-altitude-m", type=float, default=30_000.0)
    parser.add_argument("--max-glide-altitude-m", type=float, default=80_000.0)
    parser.add_argument("--min-glide-speed-mps", type=float, default=3_000.0)
    return parser.parse_args(argv)


def _balanced_labels(count: int) -> list[str]:
    maneuvers = ("longitudinal", "turning", "weaving")
    return [maneuvers[index % len(maneuvers)] for index in range(count)]


def _sample_initial_conditions(
    rng: np.random.Generator,
    *,
    initial_speed_min_mps: float,
    initial_speed_max_mps: float,
) -> dict[str, float]:
    """Sample entry states compatible with a complete 1,998 s glide record."""
    return {
        "h": float(rng.uniform(60_000.0, 72_000.0)),
        "lambda": float(rng.uniform(np.deg2rad(-0.5), np.deg2rad(0.5))),
        "phi": float(rng.uniform(np.deg2rad(-0.5), np.deg2rad(0.5))),
        "V": float(rng.uniform(initial_speed_min_mps, initial_speed_max_mps)),
        "gamma": float(rng.uniform(np.deg2rad(-1.0), np.deg2rad(1.0))),
        "psi": float(rng.uniform(np.deg2rad(90.0), np.deg2rad(110.0))),
    }


def _simulate_one(
    simulator: DCBNN_HGV_Simulator,
    initial_conditions: dict[str, float],
    maneuver: str,
    *,
    points: int,
    sampling_interval_s: float,
    internal_max_step_s: float,
    integration_method: str,
    integration_rtol: float,
    integration_atol: float,
    min_glide_altitude_m: float,
    max_glide_altitude_m: float,
    min_glide_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    radius = simulator.R_earth + initial_conditions["h"]
    initial_state = np.asarray(
        [
            radius,
            initial_conditions["lambda"],
            initial_conditions["phi"],
            initial_conditions["V"],
            initial_conditions["gamma"],
            initial_conditions["psi"],
        ],
        dtype=np.float64,
    )
    sample_times = np.arange(points, dtype=np.float64) * sampling_interval_s
    try:
        solution = solve_ivp(
            lambda t, y: simulator.hgv_dynamics(t, y, maneuver),
            (0.0, float(sample_times[-1])),
            initial_state,
            t_eval=sample_times,
            method=integration_method,
            rtol=integration_rtol,
            atol=integration_atol,
            max_step=internal_max_step_s,
            first_step=min(0.1, internal_max_step_s),
        )
    except (FloatingPointError, RuntimeError, ValueError):
        # An infeasible random entry state may descend below the surface or
        # otherwise leave the simulator domain. The caller resamples it while
        # retaining the deterministic RNG sequence and attempt count.
        return None
    if not solution.success:
        return None
    states = solution.y.T
    if states.shape != (points, 6) or not np.all(np.isfinite(states)):
        return None
    try:
        simulator_valid = simulator.validate_trajectory(states, maneuver)
    except UnicodeError:
        return None
    if not simulator_valid:
        return None
    altitudes = states[:, 0] - simulator.R_earth
    if (
        float(np.min(altitudes)) < float(min_glide_altitude_m)
        or float(np.max(altitudes)) > float(max_glide_altitude_m)
        or float(np.min(states[:, 3])) < float(min_glide_speed_mps)
    ):
        return None
    return sample_times, states


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _checkpoint_config(args: argparse.Namespace, count: int) -> dict[str, Any]:
    ignored = {"force", "resume", "max_new_trajectories", "output"}
    config = {
        key: _json_value(value)
        for key, value in vars(args).items()
        if key not in ignored
    }
    config["resolved_count"] = int(count)
    return config


def _save_checkpoint(
    path: Path,
    *,
    clean_rows: list[np.ndarray],
    tracked_rows: list[np.ndarray],
    radar_rows: list[np.ndarray],
    labels: list[str],
    initial_states: list[np.ndarray],
    attempts: int,
    cumulative_elapsed_s: float,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The checkpoint is temporary and rewritten after every chunk. Keeping it
    # uncompressed avoids repeatedly spending CPU time recompressing all prior
    # trajectories; the completed formal artifact is still compressed.
    np.savez(
        path,
        clean_trajectories=np.stack(clean_rows).astype(np.float32),
        tracked_trajectories=np.stack(tracked_rows).astype(np.float32),
        noisy_radar_razel=np.stack(radar_rows).astype(np.float32),
        trajectory_labels=np.asarray(labels),
        initial_states=np.stack(initial_states).astype(np.float32),
        generation_attempts=np.asarray(attempts, dtype=np.int64),
        cumulative_elapsed_s=np.asarray(cumulative_elapsed_s, dtype=np.float64),
        rng_state_json=np.asarray(json.dumps(rng.bit_generator.state)),
        checkpoint_config_json=np.asarray(
            json.dumps(config, sort_keys=True, ensure_ascii=False)
        ),
    )


def _load_checkpoint(
    path: Path,
    *,
    rng: np.random.Generator,
    expected_config: dict[str, Any],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    list[np.ndarray],
    int,
    float,
]:
    loaded = np.load(path, allow_pickle=False)
    try:
        stored_config = json.loads(str(loaded["checkpoint_config_json"].item()))
        if stored_config != expected_config:
            raise ValueError(
                "Generation checkpoint configuration does not match the requested protocol."
            )
        rng.bit_generator.state = json.loads(str(loaded["rng_state_json"].item()))
        clean_rows = [row.astype(np.float64) for row in loaded["clean_trajectories"]]
        tracked_rows = [row.astype(np.float64) for row in loaded["tracked_trajectories"]]
        radar_rows = [row.astype(np.float64) for row in loaded["noisy_radar_razel"]]
        labels = loaded["trajectory_labels"].astype(str).tolist()
        initial_states = [row.astype(np.float64) for row in loaded["initial_states"]]
        attempts = int(loaded["generation_attempts"])
        cumulative_elapsed_s = float(loaded["cumulative_elapsed_s"])
    finally:
        loaded.close()
    return (
        clean_rows,
        tracked_rows,
        radar_rows,
        labels,
        initial_states,
        attempts,
        cumulative_elapsed_s,
    )


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.points < args.seq_len + args.pred_len:
        raise ValueError("points must be at least seq_len + pred_len.")
    if args.internal_max_step_s <= 0.0:
        raise ValueError("internal-max-step-s must be positive.")
    if args.integration_rtol <= 0.0 or args.integration_atol <= 0.0:
        raise ValueError("integration tolerances must be positive.")
    if args.initial_speed_min_mps >= args.initial_speed_max_mps:
        raise ValueError("initial-speed-min-mps must be less than initial-speed-max-mps.")
    count = int(args.count if args.count is not None else (1000 if args.formal else 9))
    if count < 9:
        raise ValueError("count must be at least 9 so every maneuver appears in every split.")
    output = Path(args.output) if args.output is not None else (
        FORMAL_OUTPUT if args.formal else DEFAULT_OUTPUT
    )
    output = output.expanduser().resolve()
    if output.name == "hgv_trajectory_dataset.npz":
        raise ValueError("Refusing to overwrite the existing trajectory_level_v1 dataset.")
    checkpoint_path = output.with_suffix(".generation_checkpoint.npz")
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Candidate output already exists: {output}. Pass --force to replace it."
        )
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"Generation checkpoint not found: {checkpoint_path}")
    if checkpoint_path.exists() and not (args.resume or args.force):
        raise FileExistsError(
            f"Generation checkpoint already exists: {checkpoint_path}. "
            "Pass --resume to continue or --force to restart."
        )
    if args.max_new_trajectories is not None and args.max_new_trajectories <= 0:
        raise ValueError("max-new-trajectories must be positive.")

    rng = np.random.default_rng(args.seed)
    simulator = DCBNN_HGV_Simulator()
    simulator.sampling_interval_s = float(args.sampling_interval_s)
    simulator.points_per_trajectory = int(args.points)
    radar = RadarConfig()
    tracking = TrackingConfig()
    labels_requested = _balanced_labels(count)
    checkpoint_config = _checkpoint_config(args, count)
    if args.resume:
        (
            clean_rows,
            tracked_rows,
            radar_rows,
            labels,
            initial_states,
            attempts,
            cumulative_elapsed_s,
        ) = _load_checkpoint(
            checkpoint_path,
            rng=rng,
            expected_config=checkpoint_config,
        )
        print(
            f"resumed {len(clean_rows)}/{count} accepted trajectories "
            f"after {attempts} attempts"
        )
    else:
        clean_rows = []
        tracked_rows = []
        radar_rows = []
        labels = []
        initial_states = []
        attempts = 0
        cumulative_elapsed_s = 0.0
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    max_attempts = count * int(args.max_attempt_factor)
    started = time.perf_counter()
    stop_after = count
    if args.max_new_trajectories is not None:
        stop_after = min(count, len(clean_rows) + int(args.max_new_trajectories))

    while len(clean_rows) < stop_after and attempts < max_attempts:
        maneuver = labels_requested[len(clean_rows)]
        initial = _sample_initial_conditions(
            rng,
            initial_speed_min_mps=args.initial_speed_min_mps,
            initial_speed_max_mps=args.initial_speed_max_mps,
        )
        attempts += 1
        simulated = _simulate_one(
            simulator,
            initial,
            maneuver,
            points=args.points,
            sampling_interval_s=args.sampling_interval_s,
            internal_max_step_s=args.internal_max_step_s,
            integration_method=args.integration_method,
            integration_rtol=args.integration_rtol,
            integration_atol=args.integration_atol,
            min_glide_altitude_m=args.min_glide_altitude_m,
            max_glide_altitude_m=args.max_glide_altitude_m,
            min_glide_speed_mps=args.min_glide_speed_mps,
        )
        if simulated is None:
            continue
        _, clean_states = simulated
        tracked = track_noisy_radar_states(
            clean_states,
            sampling_interval_s=args.sampling_interval_s,
            rng=rng,
            radar=radar,
            tracking=tracking,
            earth_radius_m=simulator.R_earth,
        )
        clean_rows.append(clean_states)
        tracked_rows.append(tracked["tracked_states"])
        radar_rows.append(tracked["noisy_razel"])
        labels.append(maneuver)
        initial_states.append(clean_states[0])
        print(
            f"accepted {len(clean_rows):4d}/{count} "
            f"({maneuver}, attempt {attempts}, elapsed {time.perf_counter() - started:.1f}s)"
        )

    if len(clean_rows) < stop_after:
        raise RuntimeError(
            f"Generated only {len(clean_rows)}/{stop_after} requested trajectories "
            f"after {attempts} attempts."
        )
    if len(clean_rows) < count:
        cumulative_elapsed_s += time.perf_counter() - started
        _save_checkpoint(
            checkpoint_path,
            clean_rows=clean_rows,
            tracked_rows=tracked_rows,
            radar_rows=radar_rows,
            labels=labels,
            initial_states=initial_states,
            attempts=attempts,
            cumulative_elapsed_s=cumulative_elapsed_s,
            rng=rng,
            config=checkpoint_config,
        )
        print(
            json.dumps(
                {
                    "status": "checkpoint_saved",
                    "checkpoint": str(checkpoint_path),
                    "accepted": len(clean_rows),
                    "target": count,
                    "attempts": attempts,
                    "elapsed_total_s": cumulative_elapsed_s,
                },
                indent=2,
            )
        )
        return checkpoint_path, checkpoint_path

    payload = assemble_pit_radar_dataset(
        np.stack(clean_rows),
        np.stack(tracked_rows),
        np.stack(radar_rows),
        labels,
        sampling_interval_s=args.sampling_interval_s,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        tracking_burn_in_steps=args.tracking_burn_in_steps,
        train_origins_per_trajectory=args.train_origins_per_trajectory,
        eval_window_stride=args.eval_window_stride,
        split_seed=args.split_seed,
        generation_seed=args.seed,
        radar=radar,
        tracking=tracking,
    )
    payload["initial_states"] = np.stack(initial_states).astype(np.float32)
    payload["integration_max_step_s"] = np.asarray(
        args.internal_max_step_s,
        dtype=np.float64,
    )
    payload["integration_method"] = np.asarray(args.integration_method)
    payload["integration_rtol"] = np.asarray(args.integration_rtol, dtype=np.float64)
    payload["integration_atol"] = np.asarray(args.integration_atol, dtype=np.float64)
    payload["generation_attempts"] = np.asarray(attempts, dtype=np.int64)
    payload["acceptance_min_glide_altitude_m"] = np.asarray(
        args.min_glide_altitude_m,
        dtype=np.float64,
    )
    payload["acceptance_max_glide_altitude_m"] = np.asarray(
        args.max_glide_altitude_m,
        dtype=np.float64,
    )
    payload["acceptance_min_glide_speed_mps"] = np.asarray(
        args.min_glide_speed_mps,
        dtype=np.float64,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "artifact": str(output),
        "sha256": _sha256(output),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_protocol": PIT_RADAR_PROTOCOL,
        "formal_requested": bool(args.formal),
        "complete_trajectory_count": count,
        "maneuver_counts": {
            label: int(np.count_nonzero(np.asarray(labels) == label))
            for label in sorted(set(labels))
        },
        "window_counts": {
            split: int(payload[f"X_{split}"].shape[0])
            for split in ("train", "val", "test")
        },
        "complete_split_counts": {
            split: int(payload[f"complete_trajectory_ids_{split}"].shape[0])
            for split in ("train", "val", "test")
        },
        "elapsed_s": cumulative_elapsed_s + time.perf_counter() - started,
        "generation_attempts": attempts,
        "protocol_arguments": checkpoint_config,
        "completion_invocation": {
            key: _json_value(value)
            for key, value in vars(args).items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return output, manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
