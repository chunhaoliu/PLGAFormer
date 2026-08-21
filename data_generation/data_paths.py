#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Centralized dataset/scaler path helpers."""

import os
from pathlib import Path

from utils.mainline_contract import ACTIVE_DATASET_RELATIVE_PATH


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_processed_data_dir(project_root: Path | None = None) -> Path:
    override = os.getenv("HGV_PROCESSED_DIR")
    if override:
        return Path(override).expanduser().resolve()

    root = project_root if project_root is not None else get_project_root()
    return root / "data_generation" / "data" / "processed"


def get_dataset_npz_path(project_root: Path | None = None) -> Path:
    override = os.getenv("HGV_DATASET_PATH")
    if override:
        return Path(override).expanduser().resolve()
    if os.getenv("HGV_PROCESSED_DIR"):
        return get_processed_data_dir(project_root) / ACTIVE_DATASET_RELATIVE_PATH.name
    root = project_root if project_root is not None else get_project_root()
    return root / ACTIVE_DATASET_RELATIVE_PATH


def get_raw_trajectories_npz_path(project_root: Path | None = None) -> Path:
    return get_processed_data_dir(project_root) / "raw_hgv_trajectories.npz"


def get_input_scaler_path(project_root: Path | None = None) -> Path:
    override = os.getenv("HGV_INPUT_SCALER_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return get_processed_data_dir(project_root) / "scaler_hgv_trajectory.joblib"


def get_output_scaler_path(project_root: Path | None = None) -> Path:
    override = os.getenv("HGV_OUTPUT_SCALER_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return get_processed_data_dir(project_root) / "output_scaler_hgv_trajectory.joblib"


def _safe_component(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return safe or "unknown_protocol"


def configure_protocol_scaler_paths(
    dataset_identity: dict[str, object],
    *,
    force: bool = False,
    namespace: str | None = None,
) -> dict[str, Path]:
    """Pin scaler artifacts to one dataset protocol and exact dataset hash.

    Existing explicit environment overrides are respected unless ``force`` is
    true. Otherwise the paths are placed below ``processed/protocol_artifacts``.
    ``namespace`` separates consumers with different scaler-fitting contracts,
    such as Exp1 subset fitting and Exp2 full-training-split fitting.
    """
    dataset_path = Path(str(dataset_identity["dataset_path"])).resolve()
    protocol = _safe_component(dataset_identity.get("dataset_protocol", "unknown_protocol"))
    digest = _safe_component(dataset_identity.get("dataset_sha256", "unhashed"))[:16]
    artifact_dir = dataset_path.parent / "protocol_artifacts" / protocol / digest
    if namespace:
        artifact_dir = artifact_dir / _safe_component(namespace)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    defaults = {
        "HGV_INPUT_SCALER_PATH": artifact_dir / "input_scaler.joblib",
        "HGV_OUTPUT_SCALER_PATH": artifact_dir / "output_scaler.joblib",
        "HGV_SCALER_SIGNATURE_PATH": artifact_dir / "scaler_signature.json",
    }
    for name, path in defaults.items():
        if force:
            os.environ[name] = str(path)
        else:
            os.environ.setdefault(name, str(path))
    return {
        name: Path(os.environ[name]).expanduser().resolve()
        for name in defaults
    }
