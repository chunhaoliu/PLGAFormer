#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw long-trajectory artifact helpers for generated HGV data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


RAW_TRAJECTORY_PROTOCOL = "raw_trajectory_v1"
STATE_FEATURE_NAMES = ("r", "lambda", "phi", "V", "gamma", "psi")
CONTROL_FEATURE_NAMES = ("alpha", "bank")
MANEUVER_TAXONOMY = ("longitudinal", "turning", "weaving")

_LAMBDA = "\u03bb"
_PHI = "\u03c6"
_GAMMA = "\u03b3"
_PSI = "\u03c8"


def _read_field(trajectory: dict[str, Any], aliases: tuple[str, ...], *, required: bool = True) -> np.ndarray | None:
    for key in aliases:
        if key in trajectory:
            return np.asarray(trajectory[key])
    if required:
        raise KeyError(f"Trajectory is missing required field; tried aliases={aliases!r}")
    return None


def _as_1d_float(values: np.ndarray, field_name: str, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"Field '{field_name}' length {len(arr)} does not match expected {expected_len}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Field '{field_name}' contains non-finite values.")
    return arr


def trajectories_to_raw_arrays(
    trajectories: list[dict[str, Any]],
    *,
    sampling_interval_s: float,
    points_per_trajectory: int | None = None,
    trajectory_duration_s: float | None = None,
    generation_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Convert simulator trajectory dictionaries to a dense raw long-trajectory artifact."""
    if not trajectories:
        raise ValueError("At least one trajectory is required to build a raw artifact.")

    state_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    bank_rows: list[np.ndarray] = []
    velocity_rows: list[np.ndarray] = []
    labels: list[str] = []
    expected_points: int | None = points_per_trajectory

    for idx, trajectory in enumerate(trajectories):
        r = _as_1d_float(_read_field(trajectory, ("r",)), "r")
        n_points = len(r)
        if expected_points is None:
            expected_points = n_points
        if n_points != expected_points:
            raise ValueError(
                f"Trajectory {idx} has {n_points} points; expected {expected_points}."
            )

        lon = _as_1d_float(_read_field(trajectory, (_LAMBDA, "lambda", "lon")), "lambda", n_points)
        lat = _as_1d_float(_read_field(trajectory, (_PHI, "phi", "lat")), "phi", n_points)
        velocity = _as_1d_float(_read_field(trajectory, ("V", "velocity")), "V", n_points)
        gamma = _as_1d_float(_read_field(trajectory, (_GAMMA, "gamma")), "gamma", n_points)
        psi = _as_1d_float(_read_field(trajectory, (_PSI, "psi")), "psi", n_points)
        time = _as_1d_float(_read_field(trajectory, ("time",)), "time", n_points)
        alpha = _as_1d_float(_read_field(trajectory, ("alpha",)), "alpha", n_points)
        bank = _as_1d_float(_read_field(trajectory, ("bank",)), "bank", n_points)
        stored_velocity = _read_field(trajectory, ("velocity",), required=False)
        stored_velocity = velocity if stored_velocity is None else _as_1d_float(stored_velocity, "velocity", n_points)

        state_rows.append(np.column_stack([r, lon, lat, velocity, gamma, psi]).astype(np.float32))
        time_rows.append(time.astype(np.float64))
        alpha_rows.append(alpha.astype(np.float32))
        bank_rows.append(bank.astype(np.float32))
        velocity_rows.append(stored_velocity.astype(np.float32))
        labels.append(str(trajectory.get("maneuver_type", "")))

    if trajectory_duration_s is None:
        trajectory_duration_s = float(expected_points - 1) * float(sampling_interval_s)

    raw_trajectories = np.stack(state_rows, axis=0).astype(np.float32)
    payload = {
        "raw_protocol": np.asarray(RAW_TRAJECTORY_PROTOCOL),
        "trajectories": raw_trajectories,
        "time": np.stack(time_rows, axis=0).astype(np.float64),
        "alpha": np.stack(alpha_rows, axis=0).astype(np.float32),
        "bank": np.stack(bank_rows, axis=0).astype(np.float32),
        "velocity": np.stack(velocity_rows, axis=0).astype(np.float32),
        "initial_states": raw_trajectories[:, 0, :],
        "trajectory_ids": np.arange(len(trajectories), dtype=np.int64),
        "trajectory_labels": np.asarray(labels),
        "maneuver_taxonomy": np.asarray(MANEUVER_TAXONOMY),
        "state_feature_names": np.asarray(STATE_FEATURE_NAMES),
        "control_feature_names": np.asarray(CONTROL_FEATURE_NAMES),
        "sampling_interval_s": np.asarray(float(sampling_interval_s), dtype=np.float64),
        "points_per_trajectory": np.asarray(int(expected_points), dtype=np.int64),
        "trajectory_duration_s": np.asarray(float(trajectory_duration_s), dtype=np.float64),
    }
    if generation_seed is not None:
        payload["generation_seed"] = np.asarray(int(generation_seed), dtype=np.int64)
    return payload


def save_raw_trajectories(
    trajectories: list[dict[str, Any]],
    path: str | Path,
    *,
    sampling_interval_s: float,
    points_per_trajectory: int | None = None,
    trajectory_duration_s: float | None = None,
    generation_seed: int | None = None,
) -> Path:
    """Save complete generated trajectories before window construction."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = trajectories_to_raw_arrays(
        trajectories,
        sampling_interval_s=sampling_interval_s,
        points_per_trajectory=points_per_trajectory,
        trajectory_duration_s=trajectory_duration_s,
        generation_seed=generation_seed,
    )
    np.savez_compressed(output_path, **payload)
    return output_path


def load_raw_trajectory_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a raw long-trajectory NPZ artifact."""
    raw_path = Path(path)
    loaded = np.load(raw_path, allow_pickle=False)
    try:
        data = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()

    required = {
        "raw_protocol",
        "trajectories",
        "time",
        "alpha",
        "bank",
        "trajectory_ids",
        "trajectory_labels",
        "sampling_interval_s",
        "points_per_trajectory",
        "trajectory_duration_s",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Raw trajectory artifact {raw_path} is missing keys: {missing}")
    if str(data["raw_protocol"].item()) != RAW_TRAJECTORY_PROTOCOL:
        raise ValueError(
            f"Raw trajectory protocol must be {RAW_TRAJECTORY_PROTOCOL}; "
            f"got {data['raw_protocol'].item()}."
        )
    trajectories = np.asarray(data["trajectories"])
    if trajectories.ndim != 3 or trajectories.shape[-1] != len(STATE_FEATURE_NAMES):
        raise ValueError(
            "Raw trajectories must have shape [n_trajectories, n_points, 6]."
        )
    n_trajectories, n_points, _ = trajectories.shape
    if data["time"].shape != (n_trajectories, n_points):
        raise ValueError("Raw time array shape does not match trajectories.")
    if data["alpha"].shape != (n_trajectories, n_points):
        raise ValueError("Raw alpha array shape does not match trajectories.")
    if data["bank"].shape != (n_trajectories, n_points):
        raise ValueError("Raw bank array shape does not match trajectories.")
    return data


def load_raw_trajectories_as_dicts(path: str | Path) -> list[dict[str, np.ndarray | str]]:
    """Return raw trajectories in the simulator dictionary format used by plotters."""
    data = load_raw_trajectory_arrays(path)
    states = data["trajectories"]
    stored_velocity = data.get("velocity")
    output: list[dict[str, np.ndarray | str]] = []
    for idx, label in enumerate(data["trajectory_labels"].astype(str).tolist()):
        state = states[idx]
        output.append(
            {
                "r": state[:, 0],
                _LAMBDA: state[:, 1],
                _PHI: state[:, 2],
                "V": state[:, 3],
                _GAMMA: state[:, 4],
                _PSI: state[:, 5],
                "time": data["time"][idx],
                "alpha": data["alpha"][idx],
                "bank": data["bank"][idx],
                "velocity": stored_velocity[idx] if stored_velocity is not None else state[:, 3],
                "maneuver_type": label,
            }
        )
    return output
