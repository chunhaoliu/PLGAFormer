#!/usr/bin/env python3
"""Generate the leakage-controlled multi-regime HGV v2.1 dataset.

The default command generates a 60-trajectory pilot. ``--formal`` selects the
balanced 1,800-trajectory protocol (300 trajectories in each of the six joint
vertical/lateral strata).  Latin-hypercube samples are generated independently
inside every split and joint stratum.  The artifact uses clean simulator
states; it does not claim that one ground radar observes an intercontinental
trajectory.
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
from scipy.signal import find_peaks
from scipy.stats import qmc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.multiregime_protocol import (
    CONTROL_PARAMETER_NAMES,
    MANEUVER_TYPES,
    MULTIREGIME_DATASET_PROTOCOL,
    VERTICAL_REGIMES,
    MultiregimeHGVSimulator,
    sample_maneuver_profile,
)
from utils.trajectory_protocol import build_windows_for_trajectories


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_pilot_v2_1.npz"
)
FORMAL_OUTPUT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_dataset_v2_1.npz"
)
STATE_FEATURE_NAMES = ("r", "lambda", "phi", "V", "gamma", "psi")
TARGET_FEATURE_NAMES = ("r", "lambda", "phi")
DESIGN_DIMENSION = 21
FORMAL_TRAJECTORY_COUNT = 1800


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--sampling-interval-s", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--pred-len", type=int, default=256)
    parser.add_argument("--train-origins-per-trajectory", type=int, default=16)
    parser.add_argument("--eval-window-stride", type=int, default=5)
    parser.add_argument("--integration-method", choices=("DOP853", "RK45"), default="DOP853")
    parser.add_argument("--integration-rtol", type=float, default=1e-7)
    parser.add_argument("--integration-atol", type=float, default=1e-9)
    parser.add_argument("--integration-max-step-s", type=float, default=4.0)
    parser.add_argument("--max-attempt-factor", type=int, default=6)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _joint_strata(count: int) -> list[tuple[str, str]]:
    strata = [
        (vertical, maneuver)
        for vertical in VERTICAL_REGIMES
        for maneuver in MANEUVER_TYPES
    ]
    if count % len(strata) != 0:
        raise ValueError(
            f"Trajectory count must be divisible by {len(strata)} for exact balance."
        )
    return [
        stratum
        for _ in range(count // len(strata))
        for stratum in strata
    ]


def _split_counts_per_stratum(count_per_stratum: int) -> dict[str, int]:
    """Return a deterministic 70/10/20 split within each joint stratum."""
    if count_per_stratum < 5:
        raise ValueError("Each joint stratum needs at least five trajectories.")
    n_val = max(1, int(round(0.10 * count_per_stratum)))
    n_test = max(1, int(round(0.20 * count_per_stratum)))
    n_train = count_per_stratum - n_val - n_test
    if n_train < 1:
        raise ValueError("The requested count is too small for three nonempty splits.")
    return {"train": n_train, "val": n_val, "test": n_test}


def _space_filling_design(
    count: int,
    *,
    seed: int,
    split_seed: int,
    split_index: int,
    vertical_index: int,
    maneuver_index: int,
) -> np.ndarray:
    derived_seed = int(
        np.random.SeedSequence(
            [seed, split_seed, split_index, vertical_index, maneuver_index]
        ).generate_state(1)[0]
    )
    sampler = qmc.LatinHypercube(
        d=DESIGN_DIMENSION,
        scramble=True,
        seed=derived_seed,
    )
    return np.asarray(sampler.random(count), dtype=np.float64)


def _design_rows(
    count: int,
    *,
    seed: int,
    split_seed: int,
) -> list[tuple[str, str, str, np.ndarray]]:
    count_per_stratum = count // (len(VERTICAL_REGIMES) * len(MANEUVER_TYPES))
    split_counts = _split_counts_per_stratum(count_per_stratum)
    rows: list[tuple[str, str, str, np.ndarray]] = []
    for split_index, split_name in enumerate(("train", "val", "test")):
        for vertical_index, vertical in enumerate(VERTICAL_REGIMES):
            for maneuver_index, maneuver in enumerate(MANEUVER_TYPES):
                design = _space_filling_design(
                    split_counts[split_name],
                    seed=seed,
                    split_seed=split_seed,
                    split_index=split_index,
                    vertical_index=vertical_index,
                    maneuver_index=maneuver_index,
                )
                rows.extend(
                    (split_name, vertical, maneuver, unit_sample)
                    for unit_sample in design
                )
    return rows


def _sample_initial_state(
    simulator: MultiregimeHGVSimulator,
    *,
    vertical_regime: str,
    qeg_gamma_reference_deg: float,
    unit_sample: np.ndarray,
) -> np.ndarray:
    unit = np.asarray(unit_sample, dtype=np.float64)
    if unit.shape != (6,) or np.any((unit < 0.0) | (unit > 1.0)):
        raise ValueError("Initial-state unit sample must contain six values in [0, 1].")

    def scale(index: int, low: float, high: float) -> float:
        return float(low + unit[index] * (high - low))

    altitude_m = scale(0, 55_000.0, 70_000.0)
    velocity_mps = scale(1, 6_000.0, 7_000.0)
    if vertical_regime == "quasi_equilibrium":
        gamma_deg = qeg_gamma_reference_deg + scale(5, -0.03, 0.03)
    else:
        gamma_deg = scale(5, -1.0, 1.0)
    return np.asarray(
        [
            simulator.R_earth + altitude_m,
            np.deg2rad(scale(2, -20.0, 20.0)),
            np.deg2rad(scale(3, -25.0, 25.0)),
            velocity_mps,
            np.deg2rad(gamma_deg),
            np.deg2rad(scale(4, 75.0, 105.0)),
        ],
        dtype=np.float64,
    )


def _vertical_extrema_count(altitude_km: np.ndarray) -> int:
    smooth = np.convolve(altitude_km, np.ones(11) / 11.0, mode="same")[5:-5]
    peaks, _ = find_peaks(smooth, prominence=1.0, distance=25)
    valleys, _ = find_peaks(-smooth, prominence=1.0, distance=25)
    return int(len(peaks) + len(valleys))


def _simulate_candidate(
    simulator: MultiregimeHGVSimulator,
    initial_state: np.ndarray,
    profile: Any,
    *,
    points: int,
    sampling_interval_s: float,
    integration_method: str,
    integration_rtol: float,
    integration_atol: float,
    integration_max_step_s: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], int] | None:
    times = np.arange(points, dtype=np.float64) * sampling_interval_s
    try:
        solution = solve_ivp(
            lambda time_s, state: simulator.hgv_dynamics_profile(
                time_s, state, profile
            ),
            (0.0, float(times[-1])),
            initial_state,
            t_eval=times,
            method=integration_method,
            rtol=integration_rtol,
            atol=integration_atol,
            max_step=integration_max_step_s,
            first_step=min(0.1, integration_max_step_s),
        )
    except (FloatingPointError, RuntimeError, ValueError):
        return None
    if not solution.success or solution.y.shape != (6, points):
        return None
    states = solution.y.T
    if not np.all(np.isfinite(states)):
        return None

    altitude_km = (states[:, 0] - simulator.R_earth) / 1000.0
    speed_km_s = states[:, 3] / 1000.0
    if (
        float(np.min(altitude_km)) < 30.0
        or float(np.max(altitude_km)) > 80.0
        or float(np.min(speed_km_s)) < 3.0
        or float(np.max(speed_km_s)) > 7.2
    ):
        return None

    diagnostics = simulator.trajectory_diagnostics(times, states, profile)
    if (
        float(np.max(diagnostics["dynamic_pressure_pa"])) > 100_000.0
        or float(np.max(diagnostics["load_factor_g"])) > 5.0
    ):
        return None

    extrema_count = _vertical_extrema_count(altitude_km)
    if profile.vertical_regime == "quasi_equilibrium" and extrema_count > 1:
        return None
    if profile.vertical_regime == "skip_glide" and extrema_count < 2:
        return None
    return states, diagnostics, extrema_count


def _select_even_origins(max_start: int, count: int) -> np.ndarray:
    if max_start < 0:
        return np.empty(0, dtype=np.int64)
    available = max_start + 1
    count = min(int(count), available)
    return np.unique(
        np.rint(np.linspace(0, max_start, count)).astype(np.int64)
    )


def _build_split(
    states: np.ndarray,
    maneuver_labels: np.ndarray,
    vertical_labels: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    seq_len: int,
    pred_len: int,
    stride: int,
    selected_origins: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    windows = build_windows_for_trajectories(
        states,
        maneuver_labels,
        trajectory_ids,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
    )
    if selected_origins is not None:
        keep = np.isin(windows["window_starts"], selected_origins)
        windows = {key: value[keep] for key, value in windows.items()}
    vertical_by_id = {
        int(trajectory_id): str(label)
        for trajectory_id, label in zip(trajectory_ids, vertical_labels)
    }
    windows["vertical_regimes"] = np.asarray(
        [vertical_by_id[int(value)] for value in windows["trajectory_ids"]]
    )
    windows["joint_strata"] = np.char.add(
        np.char.add(windows["vertical_regimes"].astype(str), "__"),
        windows["maneuver_labels"].astype(str),
    )
    return windows


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    count = int(
        args.count
        if args.count is not None
        else (FORMAL_TRAJECTORY_COUNT if args.formal else 60)
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else (FORMAL_OUTPUT if args.formal else DEFAULT_OUTPUT)
    )
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace existing dataset without --force: {output}")

    _joint_strata(count)
    requested_design = _design_rows(
        count,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    rng = np.random.default_rng(args.seed)
    simulator = MultiregimeHGVSimulator()
    states_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    bank_command_rows: list[np.ndarray] = []
    bank_achieved_rows: list[np.ndarray] = []
    dynamic_pressure_rows: list[np.ndarray] = []
    load_factor_rows: list[np.ndarray] = []
    heat_proxy_rows: list[np.ndarray] = []
    maneuver_labels: list[str] = []
    vertical_labels: list[str] = []
    joint_labels: list[str] = []
    control_rows: list[np.ndarray] = []
    initial_rows: list[np.ndarray] = []
    extrema_counts: list[int] = []
    split_assignments: list[str] = []

    attempts = 0
    maximum_attempts = count * int(args.max_attempt_factor)
    started = time.perf_counter()
    for split_name, vertical_regime, maneuver_type, design_sample in requested_design:
        accepted = False
        local_attempt = 0
        while not accepted and attempts < maximum_attempts:
            attempts += 1
            local_attempt += 1
            candidate_sample = (
                design_sample
                if local_attempt == 1
                else rng.random(DESIGN_DIMENSION)
            )
            profile = sample_maneuver_profile(
                rng,
                vertical_regime=vertical_regime,
                maneuver_type=maneuver_type,
                unit_sample=candidate_sample[:15],
            )
            initial_state = _sample_initial_state(
                simulator,
                vertical_regime=vertical_regime,
                qeg_gamma_reference_deg=profile.qeg_gamma_reference_deg,
                unit_sample=candidate_sample[15:],
            )
            candidate = _simulate_candidate(
                simulator,
                initial_state,
                profile,
                points=args.points,
                sampling_interval_s=args.sampling_interval_s,
                integration_method=args.integration_method,
                integration_rtol=args.integration_rtol,
                integration_atol=args.integration_atol,
                integration_max_step_s=args.integration_max_step_s,
            )
            if candidate is None:
                continue
            states, diagnostics, extrema_count = candidate
            states_rows.append(states.astype(np.float32))
            alpha_rows.append(diagnostics["alpha"].astype(np.float32))
            bank_command_rows.append(
                diagnostics["bank_command"].astype(np.float32)
            )
            bank_achieved_rows.append(diagnostics["bank"].astype(np.float32))
            dynamic_pressure_rows.append(
                diagnostics["dynamic_pressure_pa"].astype(np.float32)
            )
            load_factor_rows.append(diagnostics["load_factor_g"].astype(np.float32))
            heat_proxy_rows.append(diagnostics["heat_rate_proxy"].astype(np.float32))
            maneuver_labels.append(maneuver_type)
            vertical_labels.append(vertical_regime)
            joint_labels.append(f"{vertical_regime}__{maneuver_type}")
            control_rows.append(profile.parameter_vector())
            initial_rows.append(initial_state)
            extrema_counts.append(extrema_count)
            split_assignments.append(split_name)
            accepted = True
        if not accepted:
            raise RuntimeError(
                f"Accepted {len(states_rows)}/{count} trajectories after "
                f"{attempts} attempts; increase --max-attempt-factor only after review."
            )

    states = np.stack(states_rows)
    maneuver_arr = np.asarray(maneuver_labels)
    vertical_arr = np.asarray(vertical_labels)
    joint_arr = np.asarray(joint_labels)
    split_assignment_arr = np.asarray(split_assignments)
    split_ids = {
        split_name: np.where(split_assignment_arr == split_name)[0].astype(np.int64)
        for split_name in ("train", "val", "test")
    }
    max_start = args.points - args.seq_len - args.pred_len
    train_origins = _select_even_origins(
        max_start, args.train_origins_per_trajectory
    )
    split_payload: dict[str, np.ndarray] = {}
    for split_name in ("train", "val", "test"):
        ids = split_ids[split_name]
        selected = train_origins if split_name == "train" else None
        stride = 1 if split_name == "train" else args.eval_window_stride
        windows = _build_split(
            states[ids],
            maneuver_arr[ids],
            vertical_arr[ids],
            ids,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=stride,
            selected_origins=selected,
        )
        split_payload[f"complete_trajectory_ids_{split_name}"] = ids
        split_payload[f"X_{split_name}"] = windows["X"]
        split_payload[f"y_{split_name}"] = windows["y"]
        split_payload[f"maneuver_labels_{split_name}"] = windows["maneuver_labels"]
        split_payload[f"vertical_regimes_{split_name}"] = windows["vertical_regimes"]
        split_payload[f"joint_strata_{split_name}"] = windows["joint_strata"]
        split_payload[f"trajectory_ids_{split_name}"] = windows["trajectory_ids"]
        split_payload[f"window_starts_{split_name}"] = windows["window_starts"]

    elapsed_s = time.perf_counter() - started
    payload = {
        "dataset_protocol": np.asarray(MULTIREGIME_DATASET_PROTOCOL),
        "observation_protocol": np.asarray("complete_simulator_state_v2_1"),
        "input_source": np.asarray("clean_simulated_hgv_state"),
        "target_source": np.asarray("clean_simulated_hgv_position"),
        "state_feature_names": np.asarray(STATE_FEATURE_NAMES),
        "target_feature_names": np.asarray(TARGET_FEATURE_NAMES),
        "maneuver_taxonomy": np.asarray(MANEUVER_TYPES),
        "vertical_regime_taxonomy": np.asarray(VERTICAL_REGIMES),
        "control_parameter_names": np.asarray(CONTROL_PARAMETER_NAMES),
        "sampling_interval_s": np.asarray(args.sampling_interval_s),
        "points_per_trajectory": np.asarray(args.points),
        "trajectory_duration_s": np.asarray(
            (args.points - 1) * args.sampling_interval_s
        ),
        "seq_len": np.asarray(args.seq_len),
        "pred_len": np.asarray(args.pred_len),
        "window_stride": np.asarray(args.eval_window_stride),
        "train_origin_policy": np.asarray("evenly_spaced_fixed_origins"),
        "train_origins_per_trajectory": np.asarray(len(train_origins)),
        "train_window_origins": train_origins,
        "generation_seed": np.asarray(args.seed),
        "split_seed": np.asarray(args.split_seed),
        "sampling_design": np.asarray(
            "split_and_joint_stratum_latin_hypercube_v1"
        ),
        "split_ratios": np.asarray([0.70, 0.10, 0.20], dtype=np.float64),
        "trajectory_ids": np.arange(count, dtype=np.int64),
        "trajectory_labels": maneuver_arr,
        "vertical_regimes": vertical_arr,
        "joint_strata": joint_arr,
        "clean_trajectories": states,
        "alpha_commands": np.stack(alpha_rows),
        "bank_commands": np.stack(bank_command_rows),
        "bank_achieved": np.stack(bank_achieved_rows),
        "dynamic_pressure_pa": np.stack(dynamic_pressure_rows),
        "load_factor_g": np.stack(load_factor_rows),
        "heat_rate_proxy": np.stack(heat_proxy_rows),
        "control_parameters": np.stack(control_rows),
        "initial_states": np.stack(initial_rows).astype(np.float32),
        "vertical_extrema_counts": np.asarray(extrema_counts, dtype=np.int64),
        "integration_method": np.asarray(args.integration_method),
        "integration_rtol": np.asarray(args.integration_rtol),
        "integration_atol": np.asarray(args.integration_atol),
        "integration_max_step_s": np.asarray(args.integration_max_step_s),
        "generation_attempts": np.asarray(attempts),
        **split_payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    manifest = {
        "artifact": str(output),
        "sha256": _sha256(output),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_protocol": MULTIREGIME_DATASET_PROTOCOL,
        "trajectory_count": count,
        "joint_stratum_count": int(count // 6),
        "sampling_design": "split_and_joint_stratum_latin_hypercube_v1",
        "points_per_trajectory": int(args.points),
        "sampling_interval_s": float(args.sampling_interval_s),
        "trajectory_duration_s": float((args.points - 1) * args.sampling_interval_s),
        "complete_trajectory_split_counts": {
            key: int(len(value)) for key, value in split_ids.items()
        },
        "window_counts": {
            split: int(len(split_payload[f"X_{split}"]))
            for split in ("train", "val", "test")
        },
        "generation_attempts": attempts,
        "acceptance_rate": count / attempts,
        "elapsed_s": elapsed_s,
        "arguments": {
            key: _json_value(value) for key, value in vars(args).items()
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
