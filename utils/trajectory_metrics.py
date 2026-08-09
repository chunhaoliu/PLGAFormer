#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory-level metric aggregation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Return a deterministic non-parametric CI for an independent-unit mean."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("values must contain at least one finite observation.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")

    mean = float(samples.mean())
    if samples.size == 1:
        low = high = mean
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, samples.size, size=(n_resamples, samples.size))
        bootstrap_means = samples[indices].mean(axis=1)
        alpha = 1.0 - confidence
        low, high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "n": int(samples.size),
    }


def paired_permutation_test(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    n_resamples: int = 20_000,
    seed: int = 2026,
) -> dict[str, float | int | dict[str, float | int]]:
    """Compare paired trajectory errors using a two-sided sign-flip test.

    The reported difference is ``candidate - baseline``; therefore, a negative
    value favors the candidate for error metrics.
    """
    candidate_values = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline_values = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("candidate and baseline must contain the same number of paired values.")
    valid = np.isfinite(candidate_values) & np.isfinite(baseline_values)
    differences = candidate_values[valid] - baseline_values[valid]
    if differences.size < 2:
        raise ValueError("at least two finite paired observations are required.")

    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = min(2_000, n_resamples)
    while completed < n_resamples:
        current = min(chunk_size, n_resamples - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, differences.size))
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - np.finfo(float).eps))
        completed += current

    baseline_mean = float(baseline_values[valid].mean())
    mean_difference = float(differences.mean())
    relative_improvement = (
        float(-100.0 * mean_difference / baseline_mean)
        if abs(baseline_mean) > np.finfo(float).eps
        else float("nan")
    )
    return {
        "n_pairs": int(differences.size),
        "mean_difference": mean_difference,
        "relative_improvement_percent": relative_improvement,
        "p_value": float((extreme + 1) / (n_resamples + 1)),
        "n_permutations": int(n_resamples),
        "difference_ci": bootstrap_mean_ci(
            differences,
            n_resamples=n_resamples,
            seed=seed + 1,
        ),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in input order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be a finite one-dimensional sequence in [0, 1].")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def _as_position_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [windows, horizon, 3].")
    return arr


def aggregate_window_metrics_by_trajectory(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    trajectory_ids: np.ndarray,
) -> dict[str, Any]:
    """Aggregate ADE/FDE from window predictions to complete trajectory ids.

    This does not reconstruct a single stitched trajectory. It reports the mean
    window ADE/FDE for each source trajectory id, which is stable for overlapping
    windows and avoids pretending that duplicate horizons are independent full
    rollouts.
    """
    pred = _as_position_array("y_pred", y_pred)
    true = _as_position_array("y_true", y_true)
    if pred.shape != true.shape:
        raise ValueError(f"y_pred and y_true shapes must match; got {pred.shape} and {true.shape}.")

    ids = np.asarray(trajectory_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) != pred.shape[0]:
        raise ValueError("trajectory_ids must be a 1D array with one id per prediction window.")

    point_errors = np.linalg.norm(pred - true, axis=-1)
    ade_per_window = point_errors.mean(axis=1)
    fde_per_window = point_errors[:, -1]

    trajectories: dict[int, dict[str, float | int]] = {}
    for trajectory_id in np.unique(ids):
        mask = ids == trajectory_id
        trajectories[int(trajectory_id)] = {
            "window_count": int(mask.sum()),
            "ade": float(ade_per_window[mask].mean()),
            "fde": float(fde_per_window[mask].mean()),
        }

    mean_ade = float(np.mean([item["ade"] for item in trajectories.values()])) if trajectories else float("nan")
    mean_fde = float(np.mean([item["fde"] for item in trajectories.values()])) if trajectories else float("nan")
    trajectory_ids_sorted = sorted(trajectories)
    ade_values = np.asarray([trajectories[key]["ade"] for key in trajectory_ids_sorted], dtype=np.float64)
    fde_values = np.asarray([trajectories[key]["fde"] for key in trajectory_ids_sorted], dtype=np.float64)
    return {
        "trajectory_count": int(len(trajectories)),
        "mean_ade": mean_ade,
        "mean_fde": mean_fde,
        "trajectories": trajectories,
        "ade_ci": bootstrap_mean_ci(ade_values, seed=2026) if trajectories else None,
        "fde_ci": bootstrap_mean_ci(fde_values, seed=2027) if trajectories else None,
    }
