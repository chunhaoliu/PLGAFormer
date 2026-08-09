#!/usr/bin/env python3
"""Validate a PIT-aligned radar-tracking dataset and write an audit report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.pit_radar_protocol import (
    PIT_RADAR_PROTOCOL,
    RadarConfig,
    radar_razel_to_ecef,
    spherical_to_ecef,
)
from data_provider.hgv_data import load_hgv_dataset
from data_provider.validation import validate_hgv_dataset


DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "pit_aligned_radar_pilot.npz"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(data: dict[str, np.ndarray]) -> RadarConfig:
    keys = np.asarray(data["radar_config_keys"]).astype(str).tolist()
    values = np.asarray(data["radar_config_values"], dtype=np.float64).tolist()
    return RadarConfig(**dict(zip(keys, values)))


def _paired_window_integrity(data: dict[str, np.ndarray]) -> dict[str, object]:
    clean = np.asarray(data["clean_trajectories"])
    tracked = np.asarray(data["tracked_trajectories"])
    seq_len = int(data["seq_len"])
    pred_len = int(data["pred_len"])
    maximum_x_error = 0.0
    maximum_y_error = 0.0
    checked_windows = 0
    for split in ("train", "val", "test"):
        ids = np.asarray(data[f"trajectory_ids_{split}"], dtype=np.int64)
        starts = np.asarray(data[f"window_starts_{split}"], dtype=np.int64)
        stored_x = np.asarray(data[f"X_{split}"])
        stored_y = np.asarray(data[f"y_{split}"])
        for row, (trajectory_id, start) in enumerate(zip(ids, starts)):
            expected_x = tracked[trajectory_id, start:start + seq_len, :6]
            target_start = start + seq_len
            expected_y = clean[
                trajectory_id,
                target_start:target_start + pred_len,
                :3,
            ]
            maximum_x_error = max(
                maximum_x_error,
                float(np.max(np.abs(stored_x[row] - expected_x))),
            )
            maximum_y_error = max(
                maximum_y_error,
                float(np.max(np.abs(stored_y[row] - expected_y))),
            )
            checked_windows += 1
    passed = maximum_x_error == 0.0 and maximum_y_error == 0.0
    return {
        "passed": passed,
        "checked_windows": checked_windows,
        "maximum_input_copy_error": maximum_x_error,
        "maximum_target_copy_error": maximum_y_error,
    }


def audit_dataset(path: Path) -> dict[str, object]:
    dataset_path = path.expanduser().resolve()
    dataset_sha256 = _sha256(dataset_path)
    manifest_path = dataset_path.with_suffix(".manifest.json")
    manifest: dict[str, object] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_passed = bool(
        manifest is not None
        and manifest.get("sha256") == dataset_sha256
        and manifest.get("dataset_protocol") == PIT_RADAR_PROTOCOL
    )
    bundle = load_hgv_dataset(dataset_path, require_trajectory_level=True)
    data = bundle.raw
    generic = validate_hgv_dataset(dataset_path, require_trajectory_level=True)
    clean = np.asarray(data["clean_trajectories"], dtype=np.float64)
    tracked = np.asarray(data["tracked_trajectories"], dtype=np.float64)
    noisy_razel = np.asarray(data["noisy_radar_razel"], dtype=np.float64)
    burn_in = int(data["tracking_burn_in_steps"])
    earth_radius_m = 6_378_000.0

    truth_ecef = spherical_to_ecef(clean)
    tracked_ecef = spherical_to_ecef(tracked)
    measured_ecef = radar_razel_to_ecef(noisy_razel, _read_config(data))
    raw_errors = np.linalg.norm(
        measured_ecef[:, burn_in:] - truth_ecef[:, burn_in:],
        axis=-1,
    )
    tracked_errors = np.linalg.norm(
        tracked_ecef[:, burn_in:] - truth_ecef[:, burn_in:],
        axis=-1,
    )
    altitudes = clean[:, :, 0] - earth_radius_m
    speeds = clean[:, :, 3]
    min_altitude = float(np.min(altitudes))
    max_altitude = float(np.max(altitudes))
    min_speed = float(np.min(speeds))
    min_allowed_altitude = float(data["acceptance_min_glide_altitude_m"])
    max_allowed_altitude = float(data["acceptance_max_glide_altitude_m"])
    min_allowed_speed = float(data["acceptance_min_glide_speed_mps"])
    envelope_passed = (
        min_altitude >= min_allowed_altitude
        and max_altitude <= max_allowed_altitude
        and min_speed >= min_allowed_speed
    )
    filter_passed = (
        float(np.mean(tracked_errors)) < float(np.mean(raw_errors))
        and float(np.percentile(tracked_errors, 95))
        < float(np.percentile(raw_errors, 95))
    )
    paired_integrity = _paired_window_integrity(data)
    protocol_passed = bundle.protocol == PIT_RADAR_PROTOCOL

    report: dict[str, object] = {
        "dataset": str(dataset_path),
        "sha256": dataset_sha256,
        "dataset_protocol": bundle.protocol,
        "complete_trajectory_count": int(clean.shape[0]),
        "points_per_trajectory": int(clean.shape[1]),
        "time_metadata": bundle.time_metadata,
        "window_counts": {
            split: int(data[f"X_{split}"].shape[0])
            for split in ("train", "val", "test")
        },
        "complete_split_counts": {
            split: int(data[f"complete_trajectory_ids_{split}"].shape[0])
            for split in ("train", "val", "test")
        },
        "physical_envelope": {
            "passed": envelope_passed,
            "altitude_km": [min_altitude / 1000.0, max_altitude / 1000.0],
            "speed_kmps": [min_speed / 1000.0, float(np.max(speeds)) / 1000.0],
            "required_altitude_km": [
                min_allowed_altitude / 1000.0,
                max_allowed_altitude / 1000.0,
            ],
            "required_min_speed_kmps": min_allowed_speed / 1000.0,
        },
        "radar_tracking": {
            "passed": filter_passed,
            "burn_in_steps": burn_in,
            "burn_in_s": burn_in * float(data["sampling_interval_s"]),
            "raw_position_error_m": {
                "mean": float(np.mean(raw_errors)),
                "p95": float(np.percentile(raw_errors, 95)),
            },
            "tracked_position_error_m": {
                "mean": float(np.mean(tracked_errors)),
                "p95": float(np.percentile(tracked_errors, 95)),
            },
        },
        "paired_window_integrity": paired_integrity,
        "generic_validation": {
            "passed": bool(generic["passed"]),
            "split": generic["split"],
            "taxonomy": generic["taxonomy"],
        },
        "manifest": {
            "passed": manifest_passed,
            "path": str(manifest_path),
            "recorded_sha256": None if manifest is None else manifest.get("sha256"),
        },
    }
    report["passed"] = bool(
        protocol_passed
        and generic["passed"]
        and envelope_passed
        and filter_passed
        and paired_integrity["passed"]
        and manifest_passed
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_dataset(args.dataset)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.dataset.expanduser().resolve().with_suffix(".audit.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"audit_report={output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
