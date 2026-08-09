#!/usr/bin/env python3
"""Adapters for exact external baselines whose source is not redistributed."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


AF_CILN_OFFICIAL_COMMIT = "ee39368edf833866aa472ea0ef024ebb28e92557"
AF_CILN_MODEL_SHA256 = "d6b1871e729333dbfcbdfeff8099c6602d3b4eff0d5e7c181b48d7339c6eb8af"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_git_commit(root: Path) -> str | None:
    git_dir = root / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir:"):
            return None
        git_dir = (root / marker.split(":", 1)[1].strip()).resolve()
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref:"):
        return value
    ref_name = value.split(":", 1)[1].strip()
    loose_ref = git_dir / ref_name
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref_name:
                    return commit
    return None


def resolve_af_ciln_source() -> dict[str, str]:
    """Validate and describe the exact local AF-CILN official source checkout."""
    raw_root = os.getenv("HGV_AF_CILN_ROOT", "").strip()
    if not raw_root:
        raise RuntimeError(
            "AF-CILN requires an external official checkout. Set HGV_AF_CILN_ROOT; "
            "the source is not redistributed because the upstream repository has no license file."
        )
    root = Path(raw_root).expanduser().resolve()
    model_file = root / "models" / "AF_CILN.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"AF-CILN model file not found: {model_file}")

    expected_commit = os.getenv("HGV_AF_CILN_COMMIT", AF_CILN_OFFICIAL_COMMIT).strip()
    actual_commit = _read_git_commit(root)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"AF-CILN commit mismatch: expected {expected_commit}, found {actual_commit}."
        )
    expected_hash = os.getenv("HGV_AF_CILN_SOURCE_SHA256", AF_CILN_MODEL_SHA256).strip().lower()
    actual_hash = _sha256(model_file)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"AF-CILN source hash mismatch: expected {expected_hash}, found {actual_hash}."
        )
    return {
        "root": str(root),
        "model_file": str(model_file),
        "commit": actual_commit,
        "model_sha256": actual_hash,
        "license_status": "no_license_file_in_upstream_checkout",
    }


def _load_af_ciln_class(model_file: Path):
    module_name = f"_external_af_ciln_{_sha256(model_file)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AF-CILN module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AF_CILN


class AFCILNExternalAdapter(nn.Module):
    """Exact official AF-CILN model behind a provenance and scaling adapter."""

    def __init__(
        self,
        *,
        input_dim: int = 6,
        seq_len: int = 256,
        pred_len: int = 256,
        output_dim: int = 3,
        d_model: int = 256,
        dropout: float = 0.1,
        input_scaler_mean,
        input_scaler_scale,
        output_scaler_mean,
        output_scaler_scale,
    ):
        super().__init__()
        if int(seq_len) != 256 or int(pred_len) != 256 or int(input_dim) != 6:
            raise ValueError(
                "The audited AF-CILN release is evaluated only with seq_len=256, "
                "pred_len=256, and six input channels."
            )
        source = resolve_af_ciln_source()
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            d_model=int(d_model),
            enc_in=int(input_dim),
            dropout=float(dropout),
        )
        external_class = _load_af_ciln_class(Path(source["model_file"]))
        self.external_model = external_class(config)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.output_dim = int(output_dim)
        self.source_audit = source

        self.register_buffer(
            "input_position_mean",
            torch.as_tensor(np.asarray(input_scaler_mean)[:output_dim], dtype=torch.float32),
        )
        self.register_buffer(
            "input_position_scale",
            torch.as_tensor(np.asarray(input_scaler_scale)[:output_dim], dtype=torch.float32),
        )
        self.register_buffer(
            "output_position_mean",
            torch.as_tensor(np.asarray(output_scaler_mean)[:output_dim], dtype=torch.float32),
        )
        self.register_buffer(
            "output_position_scale",
            torch.as_tensor(np.asarray(output_scaler_scale)[:output_dim], dtype=torch.float32),
        )

    def _align_positions_to_output_scale(self, x: torch.Tensor) -> torch.Tensor:
        aligned = x.clone()
        raw = (
            aligned[..., : self.output_dim] * self.input_position_scale
            + self.input_position_mean
        )
        aligned[..., : self.output_dim] = (
            raw - self.output_position_mean
        ) / self.output_position_scale
        return aligned

    def forward(
        self,
        x: torch.Tensor,
        target_length: int | None = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del decoder_context
        if x.size(1) != self.seq_len or x.size(-1) != 6:
            raise ValueError(f"AF-CILN expected [B, {self.seq_len}, 6], got {tuple(x.shape)}")
        target_length = self.pred_len if target_length is None else int(target_length)
        if not 1 <= target_length <= self.pred_len:
            raise ValueError(f"target_length must be in [1, {self.pred_len}]")
        aligned = self._align_positions_to_output_scale(x)
        mask = torch.ones_like(aligned)
        prediction = self.external_model(aligned, mask)
        if prediction.shape != (x.size(0), self.pred_len, self.output_dim):
            raise RuntimeError(f"Unexpected AF-CILN output shape: {tuple(prediction.shape)}")
        return prediction[:, :target_length, :]


__all__ = [
    "AFCILNExternalAdapter",
    "AF_CILN_OFFICIAL_COMMIT",
    "AF_CILN_MODEL_SHA256",
    "resolve_af_ciln_source",
]
