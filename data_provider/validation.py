#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset quality checks for generated HGV trajectory data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hgv_data import load_hgv_dataset
from utils.trajectory_protocol import DEFAULT_MANEUVERS


EARTH_RADIUS_M = 6_378_000.0
MIN_HEIGHT_M = 0.0
MAX_HEIGHT_M = 120_000.0
MIN_VELOCITY_MPS = 100.0
MAX_VELOCITY_MPS = 15_000.0


def _finite_report(X: np.ndarray, y: np.ndarray) -> dict[str, int]:
    return {
        "X_nonfinite": int(np.size(X) - np.isfinite(X).sum()),
        "y_nonfinite": int(np.size(y) - np.isfinite(y).sum()),
    }


def _physical_report(split_name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x_height = X[:, :, 0] - EARTH_RADIUS_M
    y_height = y[:, :, 0] - EARTH_RADIUS_M
    velocity = X[:, :, 3] if X.shape[-1] > 3 else np.asarray([], dtype=np.float32)

    report = {
        "split": split_name,
        "height_below_min": int(np.count_nonzero(np.concatenate([x_height.reshape(-1), y_height.reshape(-1)]) < MIN_HEIGHT_M)),
        "height_above_max": int(np.count_nonzero(np.concatenate([x_height.reshape(-1), y_height.reshape(-1)]) > MAX_HEIGHT_M)),
        "velocity_below_min": int(np.count_nonzero(velocity < MIN_VELOCITY_MPS)) if velocity.size else 0,
        "velocity_above_max": int(np.count_nonzero(velocity > MAX_VELOCITY_MPS)) if velocity.size else 0,
    }
    return report


def _shape_report(split_name: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    return {
        "split": split_name,
        "X_shape": tuple(int(v) for v in X.shape),
        "y_shape": tuple(int(v) for v in y.shape),
        "sample_count_match": int(X.shape[0]) == int(y.shape[0]),
        "input_dim_ok": X.ndim == 3 and X.shape[-1] == 6,
        "output_dim_ok": y.ndim == 3 and y.shape[-1] == 3,
    }


def _metadata_report(split_name: str, split) -> dict[str, Any]:
    starts = np.asarray(split.window_starts)
    return {
        "split": split_name,
        "metadata_count_match": (
            len(split.X)
            == len(split.y)
            == len(split.maneuver_labels)
            == len(split.trajectory_ids)
            == len(split.window_starts)
        ),
        "unique_trajectories": int(len(np.unique(split.trajectory_ids))),
        "negative_window_starts": int(np.count_nonzero(starts < 0)),
    }


def _taxonomy_report(raw: Any) -> dict[str, Any]:
    labels = []
    if "trajectory_labels" in raw:
        labels = np.asarray(raw["trajectory_labels"]).astype(str).tolist()
    elif "maneuver_labels" in raw:
        labels = np.asarray(raw["maneuver_labels"]).astype(str).tolist()
    else:
        split_labels = []
        for split in ("train", "val", "test"):
            key = f"maneuver_labels_{split}"
            if key in raw:
                split_labels.extend(np.asarray(raw[key]).astype(str).tolist())
        labels = split_labels
    unique = sorted(set(labels))
    expected = sorted(DEFAULT_MANEUVERS)
    return {
        "expected_maneuvers": expected,
        "observed_maneuvers": unique,
        "uses_expected_three_class_taxonomy": unique == expected,
    }


def _section_passes(section: dict[str, Any]) -> bool:
    if "sample_count_match" in section:
        return bool(section["sample_count_match"] and section["input_dim_ok"] and section["output_dim_ok"])
    if "metadata_count_match" in section:
        return bool(section["metadata_count_match"] and section["negative_window_starts"] == 0)
    if "height_below_min" in section:
        return all(int(section[key]) == 0 for key in [
            "height_below_min",
            "height_above_max",
            "velocity_below_min",
            "velocity_above_max",
        ])
    if "X_nonfinite" in section:
        return int(section["X_nonfinite"]) == 0 and int(section["y_nonfinite"]) == 0
    return True


def validate_hgv_dataset(
    path: str | Path | None = None,
    *,
    require_trajectory_level: bool = False,
) -> dict[str, Any]:
    """Return a structured quality report for an HGV dataset npz."""
    bundle = load_hgv_dataset(path, require_trajectory_level=require_trajectory_level)
    report: dict[str, Any] = {
        "path": str(bundle.path),
        "protocol": bundle.protocol,
        "time_metadata": bundle.time_metadata,
        "split": bundle.split_report,
        "taxonomy": _taxonomy_report(bundle.raw),
        "shape": {},
        "finite": {},
        "physical": {},
        "metadata": {},
    }

    checks_pass = bool(bundle.split_report is None or bundle.split_report.get("is_disjoint", False))
    checks_pass = checks_pass and bool(report["taxonomy"]["uses_expected_three_class_taxonomy"])
    for split_name, split in bundle.splits.items():
        shape = _shape_report(split_name, split.X, split.y)
        finite = _finite_report(split.X, split.y)
        physical = _physical_report(split_name, split.X, split.y)
        metadata = _metadata_report(split_name, split)

        report["shape"][split_name] = shape
        report["finite"][split_name] = finite
        report["physical"][split_name] = physical
        report["metadata"][split_name] = metadata
        checks_pass = checks_pass and all(
            _section_passes(section)
            for section in (shape, finite, physical, metadata)
        )

    report["passed"] = bool(checks_pass)
    return report
