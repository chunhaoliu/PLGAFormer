#!/usr/bin/env python3
"""Reintegrate a stratified subset under tighter tolerances.

This audit compares the stored float32 trajectories with an independent
DOP853 solve using rtol=1e-10, atol=1e-12, and max_step=1 s.  It is a numerical
convergence check, not a validation against flight data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.multiregime_protocol import (
    MULTIREGIME_DATASET_PROTOCOL,
    MultiregimeHGVSimulator,
    maneuver_profile_from_vector,
)


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_pilot_v2_1.npz"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--samples-per-stratum", type=int, default=1)
    parser.add_argument("--max-ecef-error-m", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ecef(states: np.ndarray) -> np.ndarray:
    radius = states[:, 0]
    longitude = states[:, 1]
    latitude = states[:, 2]
    cos_lat = np.cos(latitude)
    return np.column_stack(
        [
            radius * cos_lat * np.cos(longitude),
            radius * cos_lat * np.sin(longitude),
            radius * np.sin(latitude),
        ]
    )


def _select_ids(joint: np.ndarray, samples_per_stratum: int) -> np.ndarray:
    selected: list[int] = []
    for stratum in sorted(set(joint.astype(str))):
        ids = np.where(joint.astype(str) == stratum)[0]
        count = min(samples_per_stratum, len(ids))
        indices = np.unique(
            np.rint(np.linspace(0, len(ids) - 1, count)).astype(np.int64)
        )
        selected.extend(ids[indices].tolist())
    return np.asarray(selected, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    with np.load(dataset, allow_pickle=False) as loaded:
        protocol = str(np.asarray(loaded["dataset_protocol"]).item())
        if protocol != MULTIREGIME_DATASET_PROTOCOL:
            raise ValueError(f"Expected {MULTIREGIME_DATASET_PROTOCOL}, got {protocol}.")
        states = np.asarray(loaded["clean_trajectories"], dtype=np.float64)
        initial_states = np.asarray(loaded["initial_states"], dtype=np.float64)
        controls = np.asarray(loaded["control_parameters"], dtype=np.float64)
        vertical = np.asarray(loaded["vertical_regimes"]).astype(str)
        maneuver = np.asarray(loaded["trajectory_labels"]).astype(str)
        joint = np.asarray(loaded["joint_strata"]).astype(str)
        dt = float(np.asarray(loaded["sampling_interval_s"]).item())

    selected_ids = _select_ids(joint, int(args.samples_per_stratum))
    times = np.arange(states.shape[1], dtype=np.float64) * dt
    simulator = MultiregimeHGVSimulator()
    records: list[dict[str, object]] = []
    for trajectory_id in selected_ids:
        profile = maneuver_profile_from_vector(
            controls[trajectory_id],
            vertical_regime=vertical[trajectory_id],
            maneuver_type=maneuver[trajectory_id],
        )
        solution = solve_ivp(
            lambda time_s, state: simulator.hgv_dynamics_profile(
                time_s, state, profile
            ),
            (0.0, float(times[-1])),
            initial_states[trajectory_id],
            t_eval=times,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
            max_step=1.0,
            first_step=0.05,
        )
        if not solution.success or solution.y.shape != (6, len(times)):
            raise RuntimeError(
                f"Strict reintegration failed for trajectory {trajectory_id}: "
                f"{solution.message}"
            )
        strict_states = solution.y.T
        ecef_error = np.linalg.norm(
            _ecef(strict_states) - _ecef(states[trajectory_id]),
            axis=1,
        )
        records.append(
            {
                "trajectory_id": int(trajectory_id),
                "joint_stratum": str(joint[trajectory_id]),
                "max_ecef_error_m": float(np.max(ecef_error)),
                "median_ecef_error_m": float(np.median(ecef_error)),
                "terminal_ecef_error_m": float(ecef_error[-1]),
            }
        )

    overall_max = max(float(item["max_ecef_error_m"]) for item in records)
    passed = overall_max <= float(args.max_ecef_error_m)
    report = {
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "dataset_protocol": protocol,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_solver": {
            "method": "DOP853",
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_step_s": 1.0,
        },
        "samples_per_stratum": int(args.samples_per_stratum),
        "sample_count": int(len(records)),
        "maximum_allowed_ecef_error_m": float(args.max_ecef_error_m),
        "maximum_observed_ecef_error_m": overall_max,
        "passed": passed,
        "records": records,
        "scope": (
            "Numerical convergence against a tighter integration only; "
            "not flight-data or high-fidelity-model validation."
        ),
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset.with_name(dataset.stem + ".convergence.json")
    )
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
