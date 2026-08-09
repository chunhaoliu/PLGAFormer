#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive an alternative supervised-window view from frozen v2.1 trajectories.

This script never re-simulates an HGV trajectory and never changes trajectory
membership.  It rebuilds only the train/validation/test windows from the
complete trajectories stored in the audited parent artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.multiregime_protocol import MULTIREGIME_DATASET_PROTOCOL
from data_provider.hgv_data import load_hgv_dataset
from utils.trajectory_protocol import build_windows_for_trajectories


DEFAULT_PARENT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_multiregime_dataset_v2_1.npz"
)
DEFAULT_VIEW_DIR = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "derived_window_views"
)
DERIVED_WINDOW_PROTOCOL = "complete_trajectory_window_view_v1"

_SPLIT_WINDOW_PREFIXES = (
    "X_",
    "y_",
    "maneuver_labels_",
    "vertical_regimes_",
    "joint_strata_",
    "trajectory_ids_",
    "window_starts_",
    "complete_trajectory_ids_",
)
_REPLACED_METADATA_KEYS = {
    "seq_len",
    "pred_len",
    "window_stride",
    "train_origin_policy",
    "train_origins_per_trajectory",
    "train_window_origins",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def select_even_origins(max_start: int, count: int) -> np.ndarray:
    """Select the same deterministic training-origin policy as v2.1."""
    if max_start < 0:
        return np.empty(0, dtype=np.int64)
    available = max_start + 1
    count = min(int(count), available)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, max_start, count)).astype(np.int64)
    )


def build_split_window_view(
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
    """Build one split while retaining trajectory-level label provenance."""
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


def derive_window_view(
    parent_path: str | Path,
    output_path: str | Path,
    *,
    seq_len: int,
    pred_len: int,
    train_origins_per_trajectory: int = 16,
    eval_window_stride: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Create and validate one derived window artifact."""
    parent = Path(parent_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not parent.is_file():
        raise FileNotFoundError(f"Parent dataset not found: {parent}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Refusing to replace existing window view without --force: {output}"
        )
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive.")
    if train_origins_per_trajectory <= 0 or eval_window_stride <= 0:
        raise ValueError("Origin count and evaluation stride must be positive.")

    parent_sha256 = _sha256(parent)
    with np.load(parent, allow_pickle=False) as source:
        required = {
            "dataset_protocol",
            "clean_trajectories",
            "trajectory_labels",
            "vertical_regimes",
            *(f"complete_trajectory_ids_{split}" for split in ("train", "val", "test")),
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"Parent dataset is missing required keys: {missing}")

        parent_protocol = _scalar_text(source["dataset_protocol"])
        if parent_protocol != MULTIREGIME_DATASET_PROTOCOL:
            raise ValueError(
                "Window views must derive from the frozen multiregime v2.1 "
                f"protocol; got {parent_protocol!r}."
            )

        states = np.asarray(source["clean_trajectories"])
        maneuver_labels = np.asarray(source["trajectory_labels"])
        vertical_labels = np.asarray(source["vertical_regimes"])
        if states.ndim != 3 or states.shape[2] < 6:
            raise ValueError(
                "clean_trajectories must have shape [trajectory, time, features>=6]."
            )
        if len(maneuver_labels) != len(states) or len(vertical_labels) != len(states):
            raise ValueError("Complete-trajectory labels do not match trajectory count.")

        max_start = states.shape[1] - seq_len - pred_len
        if max_start < 0:
            raise ValueError(
                f"seq_len + pred_len exceeds {states.shape[1]} stored time points."
            )
        train_origins = select_even_origins(
            max_start, train_origins_per_trajectory
        )

        # Preserve every non-window parent field, including complete trajectories
        # and simulator diagnostics, so the view remains self-contained.
        payload = {
            key: np.asarray(source[key])
            for key in source.files
            if not key.startswith(_SPLIT_WINDOW_PREFIXES)
            and key not in _REPLACED_METADATA_KEYS
        }
        split_payload: dict[str, np.ndarray] = {}
        split_counts: dict[str, int] = {}
        complete_counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            ids = np.asarray(
                source[f"complete_trajectory_ids_{split}"], dtype=np.int64
            )
            if np.any(ids < 0) or np.any(ids >= len(states)):
                raise ValueError(f"Invalid complete trajectory id in split {split}.")
            selected = train_origins if split == "train" else None
            stride = 1 if split == "train" else eval_window_stride
            windows = build_split_window_view(
                states[ids],
                maneuver_labels[ids],
                vertical_labels[ids],
                ids,
                seq_len=seq_len,
                pred_len=pred_len,
                stride=stride,
                selected_origins=selected,
            )
            split_payload[f"complete_trajectory_ids_{split}"] = ids
            split_payload[f"X_{split}"] = windows["X"]
            split_payload[f"y_{split}"] = windows["y"]
            split_payload[f"maneuver_labels_{split}"] = windows[
                "maneuver_labels"
            ]
            split_payload[f"vertical_regimes_{split}"] = windows[
                "vertical_regimes"
            ]
            split_payload[f"joint_strata_{split}"] = windows["joint_strata"]
            split_payload[f"trajectory_ids_{split}"] = windows["trajectory_ids"]
            split_payload[f"window_starts_{split}"] = windows["window_starts"]
            split_counts[split] = int(len(windows["X"]))
            complete_counts[split] = int(len(ids))

        view_id = (
            f"obs{seq_len}_pred{pred_len}_"
            f"train{len(train_origins)}_evalstride{eval_window_stride}"
        )
        payload.update(
            {
                "seq_len": np.asarray(seq_len),
                "pred_len": np.asarray(pred_len),
                "window_stride": np.asarray(eval_window_stride),
                "train_origin_policy": np.asarray(
                    "evenly_spaced_fixed_origins"
                ),
                "train_origins_per_trajectory": np.asarray(len(train_origins)),
                "train_window_origins": train_origins,
                "window_view_protocol": np.asarray(DERIVED_WINDOW_PROTOCOL),
                "window_view_id": np.asarray(view_id),
                "parent_dataset_protocol": np.asarray(parent_protocol),
                "parent_dataset_sha256": np.asarray(parent_sha256),
                "parent_dataset_filename": np.asarray(parent.name),
                **split_payload,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    bundle = load_hgv_dataset(output, require_trajectory_level=True)
    if bundle.time_metadata["seq_len"] != seq_len:
        raise RuntimeError("Derived view seq_len failed post-write validation.")
    if bundle.time_metadata["pred_len"] != pred_len:
        raise RuntimeError("Derived view pred_len failed post-write validation.")

    manifest = {
        "artifact": (
            str(output.relative_to(PROJECT_ROOT))
            if output.is_relative_to(PROJECT_ROOT)
            else output.name
        ),
        "sha256": _sha256(output),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_protocol": bundle.protocol,
        "window_view_protocol": DERIVED_WINDOW_PROTOCOL,
        "window_view_id": view_id,
        "parent_dataset": (
            str(parent.relative_to(PROJECT_ROOT))
            if parent.is_relative_to(PROJECT_ROOT)
            else parent.name
        ),
        "parent_dataset_sha256": parent_sha256,
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "train_origins_per_trajectory": int(len(train_origins)),
        "eval_window_stride": int(eval_window_stride),
        "complete_trajectory_split_counts": complete_counts,
        "window_counts": split_counts,
        "trajectory_split_is_disjoint": bool(
            bundle.split_report and bundle.split_report["is_disjoint"]
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--train-origins-per-trajectory", type=int, default=16)
    parser.add_argument("--eval-window-stride", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output
    if output is None:
        output = (
            DEFAULT_VIEW_DIR
            / f"hgv_multiregime_state_v2_1_obs{args.seq_len}_pred{args.pred_len}.npz"
        )
    manifest = derive_window_view(
        args.parent,
        output,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_origins_per_trajectory=args.train_origins_per_trajectory,
        eval_window_stride=args.eval_window_stride,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
