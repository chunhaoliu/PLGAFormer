#!/usr/bin/env python3
"""In-memory cache helpers for deterministic PLGAFormer physics priors."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class _ProcessCacheEntry:
    base_dataset: Dataset
    physics_priors: torch.Tensor


_PROCESS_CACHE: dict[tuple[int, int, int, str], _ProcessCacheEntry] = {}


class PhysicsPriorDataset(Dataset):
    """Append a precomputed physics prior to every base-dataset sample."""

    def __init__(self, base_dataset: Dataset, physics_priors: torch.Tensor):
        if len(base_dataset) != int(physics_priors.size(0)):
            raise ValueError(
                "Physics-prior cache length does not match the source dataset: "
                f"{physics_priors.size(0)} != {len(base_dataset)}."
            )
        self.base_dataset = base_dataset
        self.physics_priors = physics_priors

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        sample = self.base_dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("PhysicsPriorDataset expects base samples containing source and target.")
        return (*sample, self.physics_priors[index])


@dataclass(frozen=True)
class PhysicsPriorCacheStats:
    enabled: bool
    samples: int = 0
    target_length: int = 0
    elapsed_sec: float = 0.0
    cache_bytes: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "samples": int(self.samples),
            "target_length": int(self.target_length),
            "elapsed_sec": float(self.elapsed_sec),
            "cache_bytes": int(self.cache_bytes),
            "reason": str(self.reason),
        }


def unpack_batch_with_optional_prior(batch):
    """Return ``(source, target, cached_prior_or_none)`` for 2/3-item batches."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Expected a batch containing at least source and target tensors.")
    physics_prior = batch[2] if len(batch) >= 3 else None
    return batch[0], batch[1], physics_prior


def _rebuild_loader(loader: DataLoader, dataset: Dataset) -> DataLoader:
    """Reuse the original batch sampler so ordering and seed semantics stay unchanged."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": loader.batch_sampler,
        "num_workers": int(loader.num_workers),
        "collate_fn": loader.collate_fn,
        "pin_memory": bool(loader.pin_memory),
        "timeout": float(loader.timeout),
        "worker_init_fn": loader.worker_init_fn,
        "generator": loader.generator,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = bool(loader.persistent_workers)
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(loader.prefetch_factor)
        if loader.multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = loader.multiprocessing_context
    return DataLoader(**kwargs)


def _physics_prior_signature(physics_prior_model) -> str:
    """Hash the deterministic prior configuration and scaler buffers."""
    digest = hashlib.sha256()
    digest.update(
        (
            f"{type(physics_prior_model).__module__}."
            f"{type(physics_prior_model).__qualname__}"
        ).encode("utf-8")
    )
    for name in ("sampling_interval_s", "identification_steps", "integration_stride"):
        digest.update(
            f"{name}={getattr(physics_prior_model, name, None)!r}".encode("utf-8")
        )
    for name, tensor in sorted(physics_prior_model.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def clear_process_physics_prior_cache() -> None:
    """Release process-local cache entries (primarily useful for tests)."""
    _PROCESS_CACHE.clear()


def build_cached_physics_prior_loader(
    loader: DataLoader,
    model,
    *,
    device,
    target_length: int,
    cache_batch_size: int | None = None,
) -> tuple[DataLoader, PhysicsPriorCacheStats]:
    """Precompute a deterministic PLGAFormer prior once for one dataset split.

    The returned loader preserves the original batch sampler. Models without an
    active frozen physics-prior module pass through unchanged.
    """
    physics_prior_model = getattr(model, "physics_prior", None)
    uses_physics_prior = bool(
        getattr(model, "use_prior_fusion", False)
        or getattr(model, "use_physics_corrector", False)
    )
    if not uses_physics_prior or physics_prior_model is None:
        return loader, PhysicsPriorCacheStats(
            enabled=False,
            reason="model_has_no_active_physics_prior",
        )
    if isinstance(loader.dataset, PhysicsPriorDataset):
        priors = loader.dataset.physics_priors
        return loader, PhysicsPriorCacheStats(
            enabled=True,
            samples=len(loader.dataset),
            target_length=int(priors.size(1)),
            cache_bytes=int(priors.numel() * priors.element_size()),
            reason="already_cached",
        )

    target_length = max(1, int(target_length))
    batch_size = max(1, int(cache_batch_size or loader.batch_size or 128))
    cache_key = (
        id(loader.dataset),
        len(loader.dataset),
        target_length,
        _physics_prior_signature(physics_prior_model),
    )
    process_entry = _PROCESS_CACHE.get(cache_key)
    if process_entry is not None and process_entry.base_dataset is loader.dataset:
        priors = process_entry.physics_priors
        cached_dataset = PhysicsPriorDataset(loader.dataset, priors)
        return _rebuild_loader(loader, cached_dataset), PhysicsPriorCacheStats(
            enabled=True,
            samples=len(cached_dataset),
            target_length=target_length,
            cache_bytes=int(priors.numel() * priors.element_size()),
            reason="reused_process_cache",
        )

    source_loader = DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(loader.pin_memory),
    )
    started = time.perf_counter()
    chunks: list[torch.Tensor] = []
    prior_training = bool(physics_prior_model.training)
    physics_prior_model.eval()
    try:
        with torch.inference_mode():
            for batch in source_loader:
                source, _, _ = unpack_batch_with_optional_prior(batch)
                source = source.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=torch.device(device).type == "cuda",
                )
                prior = physics_prior_model(source, target_length=target_length)
                chunks.append(prior.detach().to(device="cpu", dtype=torch.float32))
    finally:
        physics_prior_model.train(prior_training)

    if not chunks:
        raise RuntimeError("Cannot build a physics-prior cache from an empty loader.")
    priors = torch.cat(chunks, dim=0).contiguous()
    if len(loader.dataset) != int(priors.size(0)):
        raise RuntimeError(
            "Incomplete physics-prior cache: "
            f"{priors.size(0)}/{len(loader.dataset)} samples."
        )
    if not torch.isfinite(priors).all():
        raise FloatingPointError("Non-finite value in precomputed physics-prior cache.")

    cached_dataset = PhysicsPriorDataset(loader.dataset, priors)
    _PROCESS_CACHE[cache_key] = _ProcessCacheEntry(loader.dataset, priors)
    cached_loader = _rebuild_loader(loader, cached_dataset)
    elapsed = time.perf_counter() - started
    return cached_loader, PhysicsPriorCacheStats(
        enabled=True,
        samples=len(cached_dataset),
        target_length=target_length,
        elapsed_sec=elapsed,
        cache_bytes=int(priors.numel() * priors.element_size()),
        reason="precomputed_in_memory",
    )
