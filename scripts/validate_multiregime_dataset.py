#!/usr/bin/env python3
"""Audit the multi-regime HGV artifact before any formal training is allowed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.multiregime_protocol import (
    CONTROL_PARAMETER_NAMES,
    MANEUVER_TYPES,
    MULTIREGIME_DATASET_PROTOCOL,
    VERTICAL_REGIMES,
)
from utils.trajectory_protocol import validate_disjoint_trajectory_splits


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_pilot_v2_1.npz"
)
EARTH_RADIUS_M = 6_378_000.0
FORMAL_TRAJECTORY_COUNT = 1800


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true", help="Print the report without updating the audit JSON.")
    parser.add_argument("--require-formal-count", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.0, 0.05, 0.5, 0.95, 1.0)
    quantiles = np.quantile(np.asarray(values, dtype=np.float64), levels)
    return {
        name: float(value)
        for name, value in zip(("min", "p05", "median", "p95", "max"), quantiles)
    }


def _counts(values: np.ndarray) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values.astype(str).tolist()).items())
    }


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    with np.load(dataset, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    required = {
        "dataset_protocol",
        "clean_trajectories",
        "trajectory_labels",
        "vertical_regimes",
        "joint_strata",
        "alpha_commands",
        "bank_commands",
        "bank_achieved",
        "dynamic_pressure_pa",
        "load_factor_g",
        "control_parameters",
        "vertical_extrema_counts",
        "complete_trajectory_ids_train",
        "complete_trajectory_ids_val",
        "complete_trajectory_ids_test",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Dataset is missing required keys: {missing}")

    states = np.asarray(data["clean_trajectories"], dtype=np.float64)
    labels = np.asarray(data["trajectory_labels"]).astype(str)
    vertical = np.asarray(data["vertical_regimes"]).astype(str)
    joint = np.asarray(data["joint_strata"]).astype(str)
    alpha = np.asarray(data["alpha_commands"], dtype=np.float64)
    bank_command = np.asarray(data["bank_commands"], dtype=np.float64)
    bank = np.asarray(data["bank_achieved"], dtype=np.float64)
    pressure = np.asarray(data["dynamic_pressure_pa"], dtype=np.float64)
    load_factor = np.asarray(data["load_factor_g"], dtype=np.float64)
    control_parameters = np.asarray(data["control_parameters"], dtype=np.float64)
    extrema = np.asarray(data["vertical_extrema_counts"], dtype=np.int64)
    n_trajectories, n_points, n_features = states.shape
    altitude_km = (states[:, :, 0] - EARTH_RADIUS_M) / 1000.0
    speed_km_s = states[:, :, 3] / 1000.0

    checks: dict[str, dict[str, Any]] = {}
    protocol = str(np.asarray(data["dataset_protocol"]).item())
    _check(
        checks,
        "protocol_identity",
        protocol == MULTIREGIME_DATASET_PROTOCOL,
        protocol,
    )
    _check(
        checks,
        "array_shape",
        n_features == 6
        and alpha.shape == (n_trajectories, n_points)
        and bank.shape == (n_trajectories, n_points)
        and pressure.shape == (n_trajectories, n_points)
        and load_factor.shape == (n_trajectories, n_points),
        {
            "states": list(states.shape),
            "alpha": list(alpha.shape),
            "bank": list(bank.shape),
        },
    )
    _check(
        checks,
        "formal_count",
        (n_trajectories == FORMAL_TRAJECTORY_COUNT)
        if args.require_formal_count
        else (n_trajectories % 6 == 0),
        n_trajectories,
    )
    joint_counts = _counts(joint)
    _check(
        checks,
        "joint_stratum_balance",
        len(joint_counts) == 6 and len(set(joint_counts.values())) == 1,
        joint_counts,
    )
    _check(
        checks,
        "taxonomy",
        set(labels) == set(MANEUVER_TYPES)
        and set(vertical) == set(VERTICAL_REGIMES),
        {"maneuver": _counts(labels), "vertical": _counts(vertical)},
    )
    split_report = validate_disjoint_trajectory_splits(
        data["complete_trajectory_ids_train"],
        data["complete_trajectory_ids_val"],
        data["complete_trajectory_ids_test"],
    )
    _check(checks, "trajectory_split_disjoint", split_report["is_disjoint"], split_report)

    split_strata: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        ids = np.asarray(data[f"complete_trajectory_ids_{split}"], dtype=np.int64)
        split_strata[split] = _counts(joint[ids])
    _check(
        checks,
        "split_stratification",
        all(
            len(counts) == 6 and len(set(counts.values())) == 1
            for counts in split_strata.values()
        ),
        split_strata,
    )
    _check(
        checks,
        "finite_values",
        all(
            np.all(np.isfinite(array))
            for array in (
                states,
                alpha,
                bank_command,
                bank,
                pressure,
                load_factor,
                control_parameters,
            )
        ),
        "states, controls, and load diagnostics",
    )
    physical_detail = {
        "altitude_km": _quantiles(altitude_km),
        "speed_km_s": _quantiles(speed_km_s),
        "dynamic_pressure_kpa": _quantiles(pressure / 1000.0),
        "load_factor_g": _quantiles(load_factor),
    }
    _check(
        checks,
        "physical_envelope",
        float(np.min(altitude_km)) >= 30.0
        and float(np.max(altitude_km)) <= 80.0
        and float(np.min(speed_km_s)) >= 3.0
        and float(np.max(speed_km_s)) <= 7.2
        and float(np.max(pressure)) <= 100_000.0
        and float(np.max(load_factor)) <= 5.0,
        physical_detail,
    )
    qeg = vertical == "quasi_equilibrium"
    skip = vertical == "skip_glide"
    _check(
        checks,
        "vertical_regime_separation",
        bool(np.all(extrema[qeg] <= 1) and np.all(extrema[skip] >= 2)),
        {
            "quasi_equilibrium_extrema": _quantiles(extrema[qeg]),
            "skip_glide_extrema": _quantiles(extrema[skip]),
        },
    )
    bank_command_deg = np.rad2deg(bank_command)
    bank_deg = np.rad2deg(bank)
    longitudinal = labels == "longitudinal"
    turning = labels == "turning"
    weaving = labels == "weaving"
    bank_detail = {
        "longitudinal_abs_max_deg": float(
            np.max(np.abs(bank_command_deg[longitudinal]))
        ),
        "turning_abs_max_deg": _quantiles(
            np.max(np.abs(bank_command_deg[turning]), axis=1)
        ),
        "weaving_abs_max_deg": _quantiles(
            np.max(np.abs(bank_command_deg[weaving]), axis=1)
        ),
        "weaving_has_both_signs_fraction": float(
            np.mean(
                (np.min(bank_command_deg[weaving], axis=1) < 0.0)
                & (np.max(bank_command_deg[weaving], axis=1) > 0.0)
            )
        ),
        "achieved_max_rate_deg_s": float(
            np.max(np.abs(np.diff(bank_deg, axis=1)))
            / float(np.asarray(data["sampling_interval_s"]).item())
        ),
        "command_achieved_are_distinct": bool(
            np.max(np.abs(bank_command - bank)) > np.deg2rad(1.0)
        ),
    }
    _check(
        checks,
        "maneuver_control_semantics",
        bank_detail["longitudinal_abs_max_deg"] == 0.0
        and bank_detail["turning_abs_max_deg"]["min"] >= 20.0
        and bank_detail["turning_abs_max_deg"]["max"] <= 35.0
        and bank_detail["weaving_abs_max_deg"]["min"] >= 20.0
        and bank_detail["weaving_abs_max_deg"]["max"] <= 35.0
        and bank_detail["weaving_has_both_signs_fraction"] == 1.0,
        bank_detail,
    )
    _check(
        checks,
        "smooth_bank_actuation",
        bank_detail["achieved_max_rate_deg_s"] <= 3.05
        and bank_detail["command_achieved_are_distinct"],
        bank_detail,
    )
    parameter_std = np.std(control_parameters, axis=0)
    diversity_detail = {
        name: float(value)
        for name, value in zip(CONTROL_PARAMETER_NAMES, parameter_std)
    }
    randomized_names = (
        "alpha_ld_deg",
        "alpha_max_deg",
        "velocity_threshold_low_mps",
        "velocity_threshold_high_mps",
        "qeg_gamma_reference_deg",
    )
    randomized_indices = [CONTROL_PARAMETER_NAMES.index(name) for name in randomized_names]
    _check(
        checks,
        "control_parameter_diversity",
        bool(np.all(parameter_std[randomized_indices] > 0.0)),
        diversity_detail,
    )

    low_index = CONTROL_PARAMETER_NAMES.index("velocity_threshold_low_mps")
    high_index = CONTROL_PARAMETER_NAMES.index("velocity_threshold_high_mps")
    skip_ids = np.where(skip)[0]
    skip_speed = states[skip_ids, :, 3]
    low = control_parameters[skip_ids, low_index][:, None]
    high = control_parameters[skip_ids, high_index][:, None]
    branch_fractions = {
        "low": float(np.mean(skip_speed <= low)),
        "transition": float(np.mean((skip_speed > low) & (skip_speed < high))),
        "high": float(np.mean(skip_speed >= high)),
    }
    _check(
        checks,
        "skip_attack_law_branch_coverage",
        min(branch_fractions.values()) >= 0.10,
        branch_fractions,
    )

    initial_longitude_deg = np.rad2deg(states[:, 0, 1])
    initial_latitude_deg = np.rad2deg(states[:, 0, 2])
    initial_heading_deg = np.rad2deg(states[:, 0, 5])
    geographic_detail = {
        "initial_longitude_deg": _quantiles(initial_longitude_deg),
        "initial_latitude_deg": _quantiles(initial_latitude_deg),
        "initial_heading_deg": _quantiles(initial_heading_deg),
    }
    _check(
        checks,
        "initial_condition_geographic_diversity",
        float(np.ptp(initial_longitude_deg)) >= 35.0
        and float(np.ptp(initial_latitude_deg)) >= 45.0
        and float(np.ptp(initial_heading_deg)) >= 25.0,
        geographic_detail,
    )

    balance_variables = np.column_stack(
        [
            (states[:, 0, 0] - EARTH_RADIUS_M) / 1000.0,
            states[:, 0, 3] / 1000.0,
            initial_longitude_deg,
            initial_latitude_deg,
            initial_heading_deg,
            control_parameters[:, :5],
        ]
    )
    max_smd = 0.0
    smd_detail: dict[str, float] = {}
    for stratum in sorted(set(joint)):
        stratum_ids = np.where(joint == stratum)[0]
        train_ids = np.intersect1d(
            stratum_ids,
            np.asarray(data["complete_trajectory_ids_train"], dtype=np.int64),
        )
        for split_name in ("val", "test"):
            comparison_ids = np.intersect1d(
                stratum_ids,
                np.asarray(
                    data[f"complete_trajectory_ids_{split_name}"], dtype=np.int64
                ),
            )
            pooled = np.std(balance_variables[stratum_ids], axis=0, ddof=1)
            usable = pooled > 1e-12
            smd = np.zeros(balance_variables.shape[1], dtype=np.float64)
            smd[usable] = np.abs(
                np.mean(balance_variables[train_ids], axis=0)[usable]
                - np.mean(balance_variables[comparison_ids], axis=0)[usable]
            ) / pooled[usable]
            value = float(np.max(smd))
            smd_detail[f"{stratum}:train_vs_{split_name}"] = value
            max_smd = max(max_smd, value)
    # A 60-track pilot has only one validation trajectory per stratum, so its
    # split-level SMD is a diagnostic guard only.  The formal 30-per-stratum
    # validation blocks use the publication gate of 0.25.
    balance_limit = 0.25 if args.require_formal_count else 2.0
    _check(
        checks,
        "continuous_covariate_split_balance",
        max_smd <= balance_limit,
        {"maximum_standardized_mean_difference": max_smd, "limit": balance_limit, **smd_detail},
    )

    class_summaries: dict[str, Any] = {}
    for stratum in sorted(set(joint)):
        mask = joint == stratum
        class_summaries[stratum] = {
            "count": int(np.sum(mask)),
            "initial_altitude_km": _quantiles(altitude_km[mask, 0]),
            "terminal_altitude_km": _quantiles(altitude_km[mask, -1]),
            "initial_speed_km_s": _quantiles(speed_km_s[mask, 0]),
            "terminal_speed_km_s": _quantiles(speed_km_s[mask, -1]),
            "vertical_extrema": _quantiles(extrema[mask]),
        }

    manifest_path = dataset.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    digest = _sha256(dataset)
    _check(
        checks,
        "manifest_hash",
        bool(manifest) and manifest.get("sha256") == digest,
        {"computed": digest, "manifest": manifest.get("sha256")},
    )
    passed = all(item["passed"] for item in checks.values())
    report = {
        "dataset": str(dataset),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_protocol": protocol,
        "passed": passed,
        "checks": checks,
        "class_summaries": class_summaries,
        "paper_dataset_evidence_allowed": bool(
            passed and n_trajectories == FORMAL_TRAJECTORY_COUNT
        ),
        "data_gate_passed_for_training": bool(
            passed and n_trajectories == FORMAL_TRAJECTORY_COUNT
        ),
        "note": (
            "The formal data gate passed. Manuscript readiness and experiment "
            "authorization are tracked separately from this data-only audit."
            if passed and n_trajectories == FORMAL_TRAJECTORY_COUNT
            else "A passing pilot validates protocol mechanics only. Formal paper "
            "evidence requires the independently audited 1,800-trajectory artifact."
        ),
    }
    if not args.no_write:
        output = (
            args.output.resolve()
            if args.output is not None
            else dataset.with_suffix(".audit.json")
        )
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
