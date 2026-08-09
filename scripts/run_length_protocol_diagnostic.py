#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a length-protocol diagnostic with dataset-shape hard validation.

The main paper task remains one direct 256-step forecast.  This wrapper is for
controlled input-length ablations and separately trained output-length
diagnostics built from the same frozen complete trajectories.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_window_shape(path: str | Path) -> tuple[int, int]:
    """Read and cross-check the declared and stored supervised-window shape."""
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {"X_train", "y_train", "seq_len", "pred_len"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Dataset is missing task-shape keys: {missing}")
        seq_len = int(np.asarray(data["seq_len"]).item())
        pred_len = int(np.asarray(data["pred_len"]).item())
        stored_seq_len = int(data["X_train"].shape[1])
        stored_pred_len = int(data["y_train"].shape[1])
    if (seq_len, pred_len) != (stored_seq_len, stored_pred_len):
        raise ValueError(
            "Dataset metadata/window shape mismatch: "
            f"declared=({seq_len}, {pred_len}), "
            f"stored=({stored_seq_len}, {stored_pred_len})."
        )
    return seq_len, pred_len


def _has_option(args: list[str], option: str) -> bool:
    return option in args or any(item.startswith(f"{option}=") for item in args)


def build_formal_runner_args(
    args: argparse.Namespace, remainder: list[str]
) -> list[str]:
    """Translate one validated length task into the canonical formal runner."""
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    seq_len, pred_len = read_window_shape(dataset_path)

    if args.task_kind == "main":
        if (seq_len, pred_len) != (256, 256):
            raise ValueError(
                "The paper main task is fixed at seq_len=256, pred_len=256."
            )
        horizons = [32, 64, 128, 256]
    elif args.task_kind == "input-ablation":
        if seq_len not in {64, 128, 256} or pred_len != 256:
            raise ValueError(
                "Input-length ablation requires seq_len in {64,128,256} "
                "with pred_len=256."
            )
        horizons = [32, 64, 128, 256]
    elif args.task_kind == "output-diagnostic":
        if seq_len != 256 or pred_len not in {64, 128, 256}:
            raise ValueError(
                "Output-length diagnostic requires seq_len=256 and "
                "pred_len in {64,128,256}."
            )
        horizons = [pred_len]
    else:
        raise ValueError(f"Unsupported task kind: {args.task_kind}")

    os.environ["HGV_SEQ_LEN"] = str(seq_len)
    os.environ["HGV_PRED_LEN"] = str(pred_len)
    os.environ["HGV_PREDICTION_HORIZONS"] = ",".join(map(str, horizons))

    forwarded = [
        "--dataset-path",
        str(dataset_path),
        "--models",
        str(args.models),
        "--seeds",
        str(args.seeds),
        "--prediction-length",
        str(pred_len),
        "--prediction-horizons",
        ",".join(map(str, horizons)),
    ]
    if args.selection_only:
        forwarded.append("--selection-only")
    if args.skip_existing:
        forwarded.append("--skip-existing")
    if not _has_option(remainder, "--label-len"):
        forwarded.extend(["--label-len", str(min(128, seq_len))])
    forwarded.extend(remainder)
    return forwarded


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Any remaining options, such as --epochs 50 --batch-size 128, "
            "are forwarded to scripts/run_formal_sota_unit.py."
        ),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--task-kind",
        choices=("main", "input-ablation", "output-diagnostic"),
        required=True,
    )
    parser.add_argument("--models", default="plgaformer,transformer")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, remainder = parse_args(argv)
    runner_args = build_formal_runner_args(args, remainder)
    # Import only after shape-derived environment variables are pinned because
    # models.HGVConfig reads them during module initialization.
    from scripts.run_formal_sota_unit import main as formal_main

    return formal_main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
