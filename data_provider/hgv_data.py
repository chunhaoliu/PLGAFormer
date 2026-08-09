#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict HGV trajectory dataset loader.

This module is the top-level data-provider boundary for experiments. It follows
the Autoformer/FEDformer habit of centralizing dataset reads while enforcing the
HGV-specific rule that complete trajectory ids are split before windowing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.data_paths import get_dataset_npz_path
from utils.trajectory_protocol import (
    TRAJECTORY_LEVEL_PROTOCOL,
    TRAJECTORY_LEVEL_PROTOCOLS,
    dataset_time_metadata,
    dataset_protocol_name,
    dataset_uses_trajectory_level_split,
    get_split_maneuver_labels,
    validate_npz_trajectory_splits,
)


@dataclass(frozen=True)
class HGVSplit:
    """One train/val/test split plus its window metadata."""

    name: str
    X: np.ndarray
    y: np.ndarray
    maneuver_labels: np.ndarray
    trajectory_ids: np.ndarray
    window_starts: np.ndarray


@dataclass(frozen=True)
class HGVArrayBundle:
    """Loaded HGV npz arrays with protocol diagnostics."""

    path: Path
    protocol: str
    time_metadata: dict[str, Any]
    splits: dict[str, HGVSplit]
    split_report: dict[str, Any] | None
    raw: Any


def _require_keys(data: Any, keys: list[str], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Dataset {path} is missing required keys: {missing}")


def _read_split(data: Any, split: str, path: Path) -> HGVSplit:
    x_key = f"X_{split}"
    y_key = f"y_{split}"
    id_key = f"trajectory_ids_{split}"
    start_key = f"window_starts_{split}"
    _require_keys(data, [x_key, y_key, id_key, start_key], path)

    labels = get_split_maneuver_labels(data, split)
    if labels is None:
        raise ValueError(f"Dataset {path} is missing maneuver labels for split '{split}'.")

    X = np.asarray(data[x_key])
    y = np.asarray(data[y_key])
    trajectory_ids = np.asarray(data[id_key], dtype=np.int64)
    window_starts = np.asarray(data[start_key], dtype=np.int64)
    labels = np.asarray(labels)
    n = len(X)
    metadata_lengths = {
        "y": len(y),
        "maneuver_labels": len(labels),
        "trajectory_ids": len(trajectory_ids),
        "window_starts": len(window_starts),
    }
    bad = {key: value for key, value in metadata_lengths.items() if value != n}
    if bad:
        raise ValueError(f"Split '{split}' metadata length mismatch in {path}: X={n}, {bad}")

    return HGVSplit(
        name=split,
        X=X,
        y=y,
        maneuver_labels=labels,
        trajectory_ids=trajectory_ids,
        window_starts=window_starts,
    )


def load_hgv_dataset(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    require_trajectory_level: bool = True,
    mmap_mode: str | None = None,
) -> HGVArrayBundle:
    """Load the HGV dataset npz and enforce the selected dataset protocol."""
    dataset_path = Path(path) if path is not None else get_dataset_npz_path(
        Path(project_root) if project_root is not None else None
    )
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"HGV dataset not found: {dataset_path}")

    loaded = np.load(dataset_path, allow_pickle=True, mmap_mode=mmap_mode)
    try:
        data = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    protocol = dataset_protocol_name(data)
    if require_trajectory_level and not dataset_uses_trajectory_level_split(data):
        raise ValueError(
            f"Dataset protocol must be one of {sorted(TRAJECTORY_LEVEL_PROTOCOLS)}; "
            f"got {protocol}. "
            "Regenerate data with the trajectory-level generator before running formal experiments."
        )

    _require_keys(data, ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"], dataset_path)
    splits = {
        split: _read_split(data, split, dataset_path)
        for split in ("train", "val", "test")
    }
    split_report = validate_npz_trajectory_splits(data)
    if require_trajectory_level and (split_report is None or not split_report.get("is_disjoint", False)):
        raise ValueError(f"Trajectory split overlap detected in {dataset_path}: {split_report}")

    return HGVArrayBundle(
        path=dataset_path,
        protocol=protocol,
        time_metadata=dataset_time_metadata(data),
        splits=splits,
        split_report=split_report,
        raw=data,
    )
