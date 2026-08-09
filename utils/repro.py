#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducibility helpers shared by experiments."""

from __future__ import annotations

import os
import hashlib
import numpy as np
import torch


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    strict_deterministic: bool = False,
) -> None:
    """
    Set Python/NumPy/PyTorch seeds in one place.

    Args:
        seed: Random seed integer.
        deterministic: Whether to prefer deterministic backend behavior.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=not strict_deterministic)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Deterministically seed dataloader workers."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)


def build_torch_generator(seed: int) -> torch.Generator:
    """Build a torch Generator with a fixed seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def stable_seed_from_name(base_seed: int, name: str) -> int:
    """Derive a stable non-negative PyTorch seed from a base seed and name."""
    digest = hashlib.sha256(f"{int(base_seed)}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)
