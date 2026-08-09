from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from data_provider.hgv_data import load_hgv_dataset
from scripts.derive_multiregime_window_view import (
    DERIVED_WINDOW_PROTOCOL,
    derive_window_view,
    select_even_origins,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_parent(path: Path) -> None:
    trajectory_count = 6
    points = 20
    states = np.arange(
        trajectory_count * points * 6, dtype=np.float32
    ).reshape(trajectory_count, points, 6)
    labels = np.asarray(
        ["longitudinal", "turning", "weaving"] * 2
    )
    vertical = np.asarray(["qeg"] * 3 + ["skip"] * 3)
    np.savez_compressed(
        path,
        dataset_protocol=np.asarray("hgv_multiregime_state_v2_1"),
        observation_protocol=np.asarray("complete_simulator_state_v2_1"),
        sampling_interval_s=np.asarray(1.0),
        points_per_trajectory=np.asarray(points),
        trajectory_duration_s=np.asarray(points - 1.0),
        seq_len=np.asarray(8),
        pred_len=np.asarray(8),
        window_stride=np.asarray(5),
        train_origin_policy=np.asarray("evenly_spaced_fixed_origins"),
        train_origins_per_trajectory=np.asarray(2),
        train_window_origins=np.asarray([0, 4], dtype=np.int64),
        trajectory_ids=np.arange(trajectory_count, dtype=np.int64),
        trajectory_labels=labels,
        vertical_regimes=vertical,
        joint_strata=np.char.add(
            np.char.add(vertical.astype(str), "__"), labels.astype(str)
        ),
        clean_trajectories=states,
        complete_trajectory_ids_train=np.asarray([0, 1, 2], dtype=np.int64),
        complete_trajectory_ids_val=np.asarray([3], dtype=np.int64),
        complete_trajectory_ids_test=np.asarray([4, 5], dtype=np.int64),
        # These old window arrays must be replaced, not copied.
        X_train=np.empty((0, 8, 6), dtype=np.float32),
        y_train=np.empty((0, 8, 3), dtype=np.float32),
        maneuver_labels_train=np.empty(0, dtype="<U12"),
        trajectory_ids_train=np.empty(0, dtype=np.int64),
        window_starts_train=np.empty(0, dtype=np.int64),
        X_val=np.empty((0, 8, 6), dtype=np.float32),
        y_val=np.empty((0, 8, 3), dtype=np.float32),
        maneuver_labels_val=np.empty(0, dtype="<U12"),
        trajectory_ids_val=np.empty(0, dtype=np.int64),
        window_starts_val=np.empty(0, dtype=np.int64),
        X_test=np.empty((0, 8, 6), dtype=np.float32),
        y_test=np.empty((0, 8, 3), dtype=np.float32),
        maneuver_labels_test=np.empty(0, dtype="<U12"),
        trajectory_ids_test=np.empty(0, dtype=np.int64),
        window_starts_test=np.empty(0, dtype=np.int64),
    )


def test_select_even_origins_includes_endpoints() -> None:
    assert select_even_origins(13, 3).tolist() == [0, 6, 13]


def test_derive_window_view_preserves_parent_and_splits(tmp_path: Path) -> None:
    parent = tmp_path / "parent.npz"
    output = tmp_path / "view.npz"
    _write_parent(parent)
    parent_hash = _sha256(parent)

    manifest = derive_window_view(
        parent,
        output,
        seq_len=4,
        pred_len=3,
        train_origins_per_trajectory=3,
        eval_window_stride=2,
    )
    bundle = load_hgv_dataset(output, require_trajectory_level=True)
    raw = bundle.raw

    assert bundle.protocol == "hgv_multiregime_state_v2_1"
    assert bundle.time_metadata["seq_len"] == 4
    assert bundle.time_metadata["pred_len"] == 3
    assert bundle.split_report is not None
    assert bundle.split_report["is_disjoint"]
    assert raw["window_view_protocol"].item() == DERIVED_WINDOW_PROTOCOL
    assert raw["parent_dataset_sha256"].item() == parent_hash
    assert raw["clean_trajectories"].shape == (6, 20, 6)
    assert raw["X_train"].shape == (9, 4, 6)
    assert raw["y_train"].shape == (9, 3, 3)
    assert raw["X_val"].shape == (7, 4, 6)
    assert raw["X_test"].shape == (14, 4, 6)
    assert np.array_equal(
        np.unique(raw["trajectory_ids_test"]), np.asarray([4, 5])
    )
    assert manifest["parent_dataset_sha256"] == parent_hash
    assert manifest["window_counts"] == {"train": 9, "val": 7, "test": 14}


def test_derive_window_view_refuses_overwrite(tmp_path: Path) -> None:
    parent = tmp_path / "parent.npz"
    output = tmp_path / "view.npz"
    _write_parent(parent)
    output.write_bytes(b"existing")

    try:
        derive_window_view(parent, output, seq_len=4, pred_len=3)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected overwrite refusal.")
