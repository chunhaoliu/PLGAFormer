#!/usr/bin/env python3
"""Adapters for the official Time-Series-Library forecasting models.

The paper-facing HGV comparison uses the public TSLib implementations for the
standard Transformer, DLinear, PatchTST, and iTransformer baselines.  TSLib is
kept as an external checkout so the HGV repository does not silently maintain
another copy of its model and layer code.  The adapter only translates the
HGV ``[B, 256, 6] -> [B, 256, 3]`` interface and records the exact source
identity used for a run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn


TSLIB_OFFICIAL_REPOSITORY = "https://github.com/thuml/Time-Series-Library"
TSLIB_OFFICIAL_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"
TSLIB_LICENSE_SHA256 = "3892254202fc16573828d34ec901ab765d8eb73a03d174b131a49771adcd8d64"

TSLIB_MODEL_FILES = {
    "transformer": "models/Transformer.py",
    "dlinear": "models/DLinear.py",
    "patchtst": "models/PatchTST.py",
    "itransformer": "models/iTransformer.py",
}

TSLIB_FILE_SHA256 = {
    "models/Transformer.py": "8325b0fd814888c8001992eeea43aca2ec998ffede1dd88fa18c9a8924fcbf8c",
    "models/DLinear.py": "ca5dd503827fb4cc49ca4cf76ba5fba26ffe22a4e765c2c494590741359f33d5",
    "models/PatchTST.py": "e677cd698a9a14f036fbbecc3395146a6407a04ab790d52d433f44e3402e0616",
    "models/iTransformer.py": "6de6b88dc0bfcaa36599c12626d611205acca7aa4d572f8da5a769ef7afee232",
    "layers/Autoformer_EncDec.py": "f69c563c6f7a59ed77769fa26380b583fe63e17a8deb5ee6573a923cccc7d8e7",
    "layers/Transformer_EncDec.py": "031b26a6ed774442bfb4cf316d4a605f7642cd652a4c2b82f5e3d1fcdc47b098",
    "layers/SelfAttention_Family.py": "0fea961b640b1a7db471df2e389abfb843ce8e5767aca5af141ee22dd43be66f",
    "layers/Embed.py": "0a29593e3796b5a3798bd3d37c1150fc2256c559abdee36e83eecd1a05787247",
    "utils/masking.py": "c22de70ad076f4eeecc857aba984741da05da54375333399ff2742b041544986",
}

# ``utils/masking.py`` is imported by TSLib's common attention module.  The
# dependency hashes are checked together with the model file so an apparently
# public baseline cannot silently resolve to a different local layer stack.
PUBLIC_TSLIB_MODEL_TYPES = frozenset(TSLIB_MODEL_FILES)

# Keep the public baseline constructor profile explicit and auditable. The
# upstream implementation remains unchanged; these values only define the
# HGV task adapter configuration and follow the official TSLib depth/factor
# defaults more closely than the historical HGV Transformer capacity.
TSLIB_ADAPTER_DEFAULTS = {
    "transformer": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 1,
        "dropout": 0.1,
    },
    "dlinear": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 1,
        "dropout": 0.1,
    },
    "patchtst": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 1,
        "dropout": 0.1,
    },
    "itransformer": {
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 1,
        "dropout": 0.1,
    },
}


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


def _normalise_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower().replace("-", "_")
    if key == "itransformer":
        return key
    if key not in PUBLIC_TSLIB_MODEL_TYPES:
        raise ValueError(
            f"TSLib public adapter does not support model_type={model_type!r}; "
            f"choose one of {sorted(PUBLIC_TSLIB_MODEL_TYPES)}."
        )
    return key


def _required_files(model_types: Iterable[str]) -> list[str]:
    model_types = {_normalise_model_type(item) for item in model_types}
    files = {TSLIB_MODEL_FILES[item] for item in model_types}
    if "dlinear" in model_types:
        files.add("layers/Autoformer_EncDec.py")
    if model_types - {"dlinear"}:
        files.update(
            {
                "layers/Transformer_EncDec.py",
                "layers/SelfAttention_Family.py",
                "layers/Embed.py",
                "utils/masking.py",
            }
        )
    return sorted(files)


def _resolve_root(root: str | Path | None = None) -> Path:
    raw_root = str(root or os.getenv("HGV_TSLIB_ROOT", "")).strip()
    if not raw_root:
        raise RuntimeError(
            "The paper-facing public baselines require an official TSLib checkout. "
            "Pass --tslib-root or set HGV_TSLIB_ROOT."
        )
    resolved = Path(raw_root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"TSLib checkout is missing: {resolved}")
    return resolved


def resolve_tslib_source(
    model_type: str | None = None,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Validate and describe the exact public TSLib source used by a model."""
    model_types = (
        [_normalise_model_type(model_type)]
        if model_type is not None
        else sorted(PUBLIC_TSLIB_MODEL_TYPES)
    )
    root_path = _resolve_root(root)
    # The paper-facing adapter is intentionally pinned in code.  A mutable
    # environment override would make two runs look comparable while loading
    # different upstream implementations.
    expected_commit = TSLIB_OFFICIAL_COMMIT
    actual_commit = _read_git_commit(root_path)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"TSLib commit mismatch: expected {expected_commit}, found {actual_commit}."
        )

    hashes: dict[str, str] = {}
    for relative in _required_files(model_types):
        path = root_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"TSLib source file is missing: {path}")
        expected_hash = TSLIB_FILE_SHA256.get(relative)
        actual_hash = _sha256(path).lower()
        if expected_hash is not None and actual_hash != expected_hash:
            raise RuntimeError(
                f"TSLib source hash mismatch for {relative}: "
                f"expected {expected_hash}, found {actual_hash}."
            )
        hashes[relative] = actual_hash

    license_path = root_path / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(f"TSLib MIT license file is missing: {license_path}")
    license_hash = _sha256(license_path).lower()
    if license_hash != TSLIB_LICENSE_SHA256:
        raise RuntimeError(
            f"TSLib license hash mismatch: expected {TSLIB_LICENSE_SHA256}, "
            f"found {license_hash}."
        )

    return {
        "kind": "TSLib",
        "repository": TSLIB_OFFICIAL_REPOSITORY,
        "root": str(root_path),
        "source_root": str(root_path),
        "commit": actual_commit,
        "commit_hash": actual_commit,
        "model_types": list(model_types),
        "source_files": hashes,
        "license": "MIT",
        "license_status": "MIT_license_present",
        "license_sha256": license_hash,
    }


def tslib_signature_files(
    model_types: Iterable[str], *, root: str | Path | None = None
) -> dict[str, Path]:
    """Return validated source files for inclusion in a run signature."""
    source = resolve_tslib_source(None, root=root)
    required = _required_files(model_types)
    root_path = Path(str(source["root"]))
    return {relative: root_path / relative for relative in required}


@contextmanager
def _tslib_import_context(root: Path):
    """Load TSLib modules without permanently replacing HGV packages."""
    package_names = {"layers", "utils"}
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "layers"
        or name == "utils"
        or name.startswith("layers.")
        or name.startswith("utils.")
    }
    previous_path = list(sys.path)
    for name in list(saved):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        for package_name in package_names:
            package_root = root / package_name
            init_path = package_root / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                package_name,
                init_path,
                submodule_search_locations=[str(package_root)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load TSLib package {package_name!r} from {package_root}")
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
        yield
    finally:
        for name in list(sys.modules):
            if (
                name == "layers"
                or name == "utils"
                or name.startswith("layers.")
                or name.startswith("utils.")
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = previous_path


def _load_tslib_model_class(source: dict[str, object], model_type: str):
    root = Path(str(source["root"]))
    model_path = root / TSLIB_MODEL_FILES[model_type]
    module_name = f"_hgv_tslib_{model_type}_{str(source['commit'])[:12]}"
    with _tslib_import_context(root):
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load TSLib model module from {model_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
        model_class = getattr(module, "Model", None)
        if model_class is None:
            raise AttributeError(f"TSLib module {model_path} does not define Model")
        return model_class


class TSLibForecastAdapter(nn.Module):
    """Expose one official TSLib model through the HGV forecast interface."""

    is_oneshot = True

    def __init__(
        self,
        *,
        model_type: str,
        input_dim: int = 6,
        output_dim: int = 3,
        seq_len: int = 256,
        pred_len: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 1024,
        factor: int = 1,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
        moving_avg: int = 25,
        source_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        model_type = _normalise_model_type(model_type)
        source = resolve_tslib_source(model_type, root=source_root)
        source["adapter_interface"] = "HGV [B, seq_len, 6] -> [B, pred_len, 3]"
        source["observation_protocol"] = "complete_observation_no_mask"
        config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(seq_len),
            label_len=max(1, min(int(seq_len), int(seq_len) // 2)),
            pred_len=int(pred_len),
            enc_in=int(input_dim),
            dec_in=int(output_dim),
            c_out=int(output_dim),
            d_model=int(d_model),
            n_heads=int(n_heads),
            e_layers=int(e_layers),
            d_layers=int(d_layers),
            d_ff=int(d_ff),
            factor=int(factor),
            embed="fixed",
            freq="h",
            dropout=float(dropout),
            activation="gelu",
            moving_avg=int(moving_avg),
        )
        source["adapter_config"] = {
            "d_model": int(d_model),
            "n_heads": int(n_heads),
            "e_layers": int(e_layers),
            "d_layers": int(d_layers),
            "d_ff": int(d_ff),
            "factor": int(factor),
            "dropout": float(dropout),
            "patch_len": int(patch_len),
            "stride": int(stride),
            "moving_avg": int(moving_avg),
        }
        model_class = _load_tslib_model_class(source, model_type)
        if model_type == "patchtst":
            self.external_model = model_class(
                config,
                patch_len=int(patch_len),
                stride=int(stride),
            )
        else:
            self.external_model = model_class(config)
        self.model_type = model_type
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.source_audit = source

    def forward(
        self,
        x: torch.Tensor,
        target_length: int | None = None,
        decoder_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.input_dim:
            raise ValueError(
                f"TSLib {self.model_type} expects [B, L, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"TSLib {self.model_type} expects seq_len={self.seq_len}, got {x.size(1)}"
            )
        target_length = self.pred_len if target_length is None else int(target_length)
        if not 1 <= target_length <= self.pred_len:
            raise ValueError(f"target_length must be in [1, {self.pred_len}]")
        if decoder_context is None:
            decoder_context = torch.zeros(
                x.size(0), self.pred_len, self.output_dim,
                device=x.device, dtype=x.dtype,
            )
        else:
            decoder_context = decoder_context[..., : self.output_dim]
        prediction = self.external_model(x, None, decoder_context, None)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
            raise RuntimeError(
                f"Unexpected TSLib {self.model_type} output: {type(prediction).__name__}"
            )
        if prediction.size(-1) < self.output_dim or prediction.size(1) < target_length:
            raise RuntimeError(
                f"TSLib {self.model_type} output has incompatible shape {tuple(prediction.shape)}"
            )
        return prediction[:, -target_length:, : self.output_dim]


__all__ = [
    "PUBLIC_TSLIB_MODEL_TYPES",
    "TSLIB_ADAPTER_DEFAULTS",
    "TSLIB_OFFICIAL_COMMIT",
    "TSLIB_OFFICIAL_REPOSITORY",
    "TSLibForecastAdapter",
    "resolve_tslib_source",
    "tslib_signature_files",
]
