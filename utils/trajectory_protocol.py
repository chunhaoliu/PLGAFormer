#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory-level dataset protocol helpers.

The project task is long HGV trajectory prediction. Splits therefore happen on
complete trajectory ids first; sliding windows are generated only inside each
split. This avoids putting overlapping windows from the same trajectory into
both training and evaluation sets.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_LEVEL_PROTOCOL = "trajectory_level_v1"
PIT_RADAR_TRAJECTORY_LEVEL_PROTOCOL = "pit_aligned_radar_v1"
LEGACY_MULTIREGIME_TRAJECTORY_LEVEL_PROTOCOL = "hgv_multiregime_state_v2"
MULTIREGIME_TRAJECTORY_LEVEL_PROTOCOL = "hgv_multiregime_state_v2_1"
TRAJECTORY_LEVEL_PROTOCOLS = frozenset(
    {
        TRAJECTORY_LEVEL_PROTOCOL,
        PIT_RADAR_TRAJECTORY_LEVEL_PROTOCOL,
        LEGACY_MULTIREGIME_TRAJECTORY_LEVEL_PROTOCOL,
        MULTIREGIME_TRAJECTORY_LEVEL_PROTOCOL,
    }
)
LEGACY_WINDOW_PROTOCOL = "legacy_window_random_split"
DEFAULT_TRAJECTORY_COUNT = 1000
DEFAULT_POINTS_PER_TRAJECTORY = 1000
DEFAULT_SAMPLING_INTERVAL_S = 1.0
DEFAULT_SEQ_LEN = 256
DEFAULT_PRED_LEN = 256
DEFAULT_WINDOW_STRIDE = 5
DEFAULT_PREDICTION_HORIZONS = (32, 64, 128, 256)
DEFAULT_MANEUVERS = ("longitudinal", "turning", "weaving")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def steps_to_seconds(steps: int | float, sampling_interval_s: int | float = DEFAULT_SAMPLING_INTERVAL_S) -> float:
    """Convert forecast steps to physical seconds using dataset sampling metadata."""
    return float(steps) * float(sampling_interval_s)


def horizon_label(
    steps: int,
    sampling_interval_s: int | float = DEFAULT_SAMPLING_INTERVAL_S,
    *,
    compact: bool = False,
) -> str:
    """Return an unambiguous horizon label for tables and figures."""
    seconds = steps_to_seconds(steps, sampling_interval_s)
    if abs(seconds - int(seconds)) < 1e-9:
        seconds_text = str(int(seconds))
    else:
        seconds_text = f"{seconds:g}"
    if compact:
        return f"{int(steps)} step/{seconds_text} s"
    return f"{int(steps)} steps ({seconds_text} s)"


def read_scalar(data: Any, key: str, default: Any = None) -> Any:
    """Read a scalar value from an npz-like object or mapping."""
    try:
        value = data[key]
    except Exception:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def dataset_time_metadata(data: Any) -> dict[str, Any]:
    """Return normalized dataset timing/window metadata with safe defaults."""
    sampling_interval_s = float(read_scalar(data, "sampling_interval_s", DEFAULT_SAMPLING_INTERVAL_S))
    seq_len = int(read_scalar(data, "seq_len", DEFAULT_SEQ_LEN))
    pred_len = int(read_scalar(data, "pred_len", DEFAULT_PRED_LEN))
    window_stride = int(read_scalar(data, "window_stride", DEFAULT_WINDOW_STRIDE))
    points = read_scalar(data, "points_per_trajectory", None)
    duration = read_scalar(data, "trajectory_duration_s", None)
    generation_seed = read_scalar(data, "generation_seed", None)
    return {
        "sampling_interval_s": sampling_interval_s,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "window_stride": window_stride,
        "points_per_trajectory": None if points is None else int(points),
        "trajectory_duration_s": None if duration is None else float(duration),
        "generation_seed": None if generation_seed is None else int(generation_seed),
        "input_duration_s": steps_to_seconds(seq_len, sampling_interval_s),
        "prediction_duration_s": steps_to_seconds(pred_len, sampling_interval_s),
    }


def dataset_file_identity(path: str | Path) -> dict[str, Any]:
    """Read the immutable identity and timing contract of a dataset artifact."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset artifact not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        identity = {
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_protocol": dataset_protocol_name(data),
            **dataset_time_metadata(data),
        }
    return identity


def _as_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_string(value.item(), default=default)
        if value.size == 1:
            return _as_string(value.reshape(-1)[0], default=default)
    return str(value)


def _normalize_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    total = float(ratios.sum())
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    ratios = ratios / total
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def _counts_for_class(n_items: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 0, 1

    n_train = int(np.floor(n_items * train_ratio))
    n_val = int(np.floor(n_items * val_ratio))
    n_train = max(1, min(n_train, n_items - 2))
    n_val = max(1, min(n_val, n_items - n_train - 1))
    n_test = n_items - n_train - n_val
    if n_test < 1:
        n_train = max(1, n_train - (1 - n_test))
        n_test = n_items - n_train - n_val
    return n_train, n_val, n_test


def split_trajectory_ids(
    labels: Sequence[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified split of complete trajectory ids.

    Args:
        labels: One maneuver label per complete trajectory.
        train_ratio: Relative train split size.
        val_ratio: Relative validation split size.
        test_ratio: Relative test split size.
        seed: Random seed for deterministic shuffling.
    """
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be a 1D sequence with one entry per trajectory.")

    train_ratio, val_ratio, test_ratio = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []

    for label in np.unique(labels_arr):
        class_ids = np.where(labels_arr == label)[0]
        class_ids = rng.permutation(class_ids)
        n_train, n_val, _ = _counts_for_class(len(class_ids), train_ratio, val_ratio)
        train_ids.extend(class_ids[:n_train].tolist())
        val_ids.extend(class_ids[n_train:n_train + n_val].tolist())
        test_ids.extend(class_ids[n_train + n_val:].tolist())

    return {
        "train": np.sort(np.asarray(train_ids, dtype=np.int64)),
        "val": np.sort(np.asarray(val_ids, dtype=np.int64)),
        "test": np.sort(np.asarray(test_ids, dtype=np.int64)),
    }


def validate_disjoint_trajectory_splits(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
) -> dict[str, Any]:
    """Return overlap diagnostics for trajectory-level splits."""
    train_set = set(np.asarray(train_ids, dtype=np.int64).tolist())
    val_set = set(np.asarray(val_ids, dtype=np.int64).tolist())
    test_set = set(np.asarray(test_ids, dtype=np.int64).tolist())
    train_val = sorted(train_set & val_set)
    train_test = sorted(train_set & test_set)
    val_test = sorted(val_set & test_set)
    return {
        "is_disjoint": not train_val and not train_test and not val_test,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "train_val_overlap": train_val,
        "train_test_overlap": train_test,
        "val_test_overlap": val_test,
    }


def build_windows_for_trajectories(
    trajectories: Sequence[np.ndarray],
    labels: Sequence[Any],
    trajectory_ids: Sequence[int] | None = None,
    seq_len: int = 64,
    pred_len: int = 128,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """Build fixed-length windows within already-selected complete trajectories."""
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if len(trajectories) != len(labels):
        raise ValueError("trajectories and labels must have the same length.")
    if trajectory_ids is None:
        trajectory_ids = np.arange(len(trajectories), dtype=np.int64)
    if len(trajectory_ids) != len(trajectories):
        raise ValueError("trajectory_ids must match trajectories length.")

    X: list[np.ndarray] = []
    y: list[np.ndarray] = []
    maneuver_labels: list[Any] = []
    source_ids: list[int] = []
    window_starts: list[int] = []

    for traj, label, trajectory_id in zip(trajectories, labels, trajectory_ids):
        traj_arr = np.asarray(traj)
        if traj_arr.ndim != 2 or traj_arr.shape[1] < 3:
            raise ValueError("Each trajectory must have shape [time, features>=3].")
        max_start = traj_arr.shape[0] - seq_len - pred_len
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride):
            split = start + seq_len
            end = split + pred_len
            X.append(traj_arr[start:split, :6])
            y.append(traj_arr[split:end, :3])
            maneuver_labels.append(label)
            source_ids.append(int(trajectory_id))
            window_starts.append(int(start))

    input_dim = 6
    return {
        "X": np.asarray(X, dtype=np.float32).reshape(-1, seq_len, input_dim),
        "y": np.asarray(y, dtype=np.float32).reshape(-1, pred_len, 3),
        "maneuver_labels": np.asarray(maneuver_labels),
        "trajectory_ids": np.asarray(source_ids, dtype=np.int64),
        "window_starts": np.asarray(window_starts, dtype=np.int64),
    }


def dataset_protocol_name(data: Any) -> str:
    """Read the dataset protocol string from an npz-like object."""
    try:
        return _as_string(data["dataset_protocol"], default=LEGACY_WINDOW_PROTOCOL)
    except Exception:
        return LEGACY_WINDOW_PROTOCOL


def dataset_uses_trajectory_level_split(data: Any) -> bool:
    return dataset_protocol_name(data) in TRAJECTORY_LEVEL_PROTOCOLS


def validate_npz_trajectory_splits(data: Any) -> dict[str, Any] | None:
    required = ["trajectory_ids_train", "trajectory_ids_val", "trajectory_ids_test"]
    if not all(key in data for key in required):
        return None
    return validate_disjoint_trajectory_splits(
        np.unique(data["trajectory_ids_train"]),
        np.unique(data["trajectory_ids_val"]),
        np.unique(data["trajectory_ids_test"]),
    )


def get_split_maneuver_labels(data: Any, split: str) -> np.ndarray | None:
    """Return maneuver labels for a split, supporting new and legacy datasets."""
    split = split.lower().strip()
    split_key = f"maneuver_labels_{split}"
    if split_key in data:
        return np.asarray(data[split_key])
    if "maneuver_labels" not in data:
        return None
    all_labels = np.asarray(data["maneuver_labels"])
    index_key = f"split_{split}_indices"
    if index_key in data:
        return all_labels[np.asarray(data[index_key], dtype=np.int64)]
    return None
