from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts.run_length_protocol_diagnostic import (
    build_formal_runner_args,
    read_window_shape,
)


def _write_window_file(path: Path, seq_len: int, pred_len: int) -> None:
    np.savez_compressed(
        path,
        seq_len=np.asarray(seq_len),
        pred_len=np.asarray(pred_len),
        X_train=np.zeros((2, seq_len, 6), dtype=np.float32),
        y_train=np.zeros((2, pred_len, 3), dtype=np.float32),
    )


def _args(path: Path, task_kind: str) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_path=str(path),
        task_kind=task_kind,
        models="plgaformer,transformer",
        seeds="42",
        selection_only=True,
        skip_existing=False,
    )


def test_read_window_shape_rejects_metadata_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        seq_len=np.asarray(64),
        pred_len=np.asarray(256),
        X_train=np.zeros((2, 128, 6), dtype=np.float32),
        y_train=np.zeros((2, 256, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="metadata/window shape mismatch"):
        read_window_shape(path)


def test_input_ablation_maps_to_prefix_reporting(tmp_path: Path) -> None:
    path = tmp_path / "obs64_pred256.npz"
    _write_window_file(path, 64, 256)
    forwarded = build_formal_runner_args(
        _args(path, "input-ablation"),
        ["--epochs", "2"],
    )

    assert forwarded[forwarded.index("--prediction-length") + 1] == "256"
    assert (
        forwarded[forwarded.index("--prediction-horizons") + 1]
        == "32,64,128,256"
    )
    assert forwarded[forwarded.index("--label-len") + 1] == "64"
    assert "--selection-only" in forwarded


def test_output_diagnostic_reports_only_its_trained_horizon(
    tmp_path: Path,
) -> None:
    path = tmp_path / "obs256_pred128.npz"
    _write_window_file(path, 256, 128)
    forwarded = build_formal_runner_args(
        _args(path, "output-diagnostic"),
        ["--label-len", "96"],
    )

    assert forwarded[forwarded.index("--prediction-length") + 1] == "128"
    assert forwarded[forwarded.index("--prediction-horizons") + 1] == "128"
    assert forwarded.count("--label-len") == 1
    assert forwarded[forwarded.index("--label-len") + 1] == "96"


def test_main_task_rejects_non_256_by_256_view(tmp_path: Path) -> None:
    path = tmp_path / "obs128_pred256.npz"
    _write_window_file(path, 128, 256)
    with pytest.raises(ValueError, match="main task is fixed"):
        build_formal_runner_args(_args(path, "main"), [])
