from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "configs" / "formal_v3.json"
ACTIVE_DATASET_RELATIVE_PATH = Path(
    "data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz"
)
ACTIVE_MODEL_KEY = "full"
ACTIVE_PLGAFORMER_FLAGS = MappingProxyType(
    {
        "use_sparse_attention": False,
        "use_physics_corrector": False,
        "use_multi_head_output": True,
        "use_prior_fusion": True,
        "use_channel_residual": False,
        "prior_type": "rotating_3dof",
        "prior_blend_mode": "adaptive",
    }
)
PAPER_INFERENCE_POLICY = MappingProxyType(
    {
        "name": "short_horizon_physics_prior_lock_tau450_power2.5",
        "lock_steps": 64,
        "time_constant_s": 450.0,
        "decay_power": 2.5,
    }
)
ACTIVE_TRAINABLE_MODEL_KEYS = (
    "baseline",
    "full",
    "dlinear",
    "patchtst",
    "itransformer",
)


def load_mainline_config() -> dict[str, Any]:
    """Load the active config and fail closed on mainline contract drift."""
    payload = json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    if Path(payload["dataset"]["relative_path"]) != ACTIVE_DATASET_RELATIVE_PATH:
        raise ValueError("Active dataset relative path diverges from the mainline contract.")
    if tuple(payload["main_results"]["trainable_model_keys"]) != ACTIVE_TRAINABLE_MODEL_KEYS:
        raise ValueError("Active trainable model keys diverge from the mainline contract.")
    return payload
