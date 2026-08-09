#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared experiment IO helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.formal_evidence import load_evidence_bundle, resolve_bundle_checkpoint


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)


def get_experiment_dirs(project_root: str | Path, exp_name: str) -> tuple[Path, Path]:
    """
    Return `(results_dir, models_dir)` for an experiment.

    Example exp_name: `exp1_sota`, `exp2_ablation`.
    """
    root = Path(project_root)
    exp_root = root / "experiments" / exp_name
    results_dir = ensure_dir(exp_root / "results")
    models_dir = ensure_dir(exp_root / "trained_models")
    return results_dir, models_dir


def _safe_path_component(value: object) -> str:
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return safe or "unknown_protocol"


def get_formal_protocol_dirs(
    project_root: str | Path,
    exp_name: str,
    dataset_protocol: str,
    evidence_tier: str,
    *,
    generation: str = "formal_v3",
) -> tuple[Path, Path]:
    """Return isolated result/model directories for one formal data protocol."""
    results_dir, models_dir = get_experiment_dirs(project_root, exp_name)
    protocol = _safe_path_component(dataset_protocol)
    tier = _safe_path_component(evidence_tier)
    generation_name = _safe_path_component(generation)
    return (
        ensure_dir(results_dir / generation_name / protocol / tier),
        ensure_dir(models_dir / generation_name / protocol / tier),
    )


def resolve_exp1_checkpoint(
    project_root: str | Path,
    model_type: str,
    *,
    seed: int | None = None,
    run_signature: str | None = None,
    bundle_path: str | Path | None = None,
    allow_legacy: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a checkpoint from the active formal-v3 bundle.

    The legacy ``partial_runs.json`` registry remains available only for an
    explicit diagnostic request.  A temporary project root used by compatibility
    tests is also treated as a legacy fixture; the canonical HGV project root is
    fail-closed when its formal bundle is absent.
    """
    normalized_model = str(model_type).strip().lower()
    requested_seed = int(seed if seed is not None else os.getenv("HGV_EXP1_CHECKPOINT_SEED", "42"))
    key_aliases = {
        "transformer": "baseline",
        "baseline": "baseline",
        "plgaformer": "full",
        "proposed": "full",
    }
    model_key = key_aliases.get(normalized_model, normalized_model)
    override_name = f"HGV_EXP1_{normalized_model.upper()}_CHECKPOINT"
    override = os.getenv(override_name, "").strip()
    root = Path(project_root).expanduser().resolve()
    formal_bundle = (
        Path(bundle_path).expanduser().resolve()
        if bundle_path
        else root
        / "experiments"
        / "exp1_sota"
        / "results"
        / "formal_v3"
        / "hgv_multiregime_state_v2_1"
        / "final"
        / "main_run_set_manifest.json"
    )
    if formal_bundle.is_file():
        bundle = load_evidence_bundle(formal_bundle)
        if bundle.get("paper_eligible") is not True:
            codes = sorted({str(item.get("code")) for item in bundle.get("blockers", [])})
            raise RuntimeError(
                "Formal-v3 Main Results bundle is not paper-eligible; blockers="
                + ",".join(codes)
            )
        checkpoint = resolve_bundle_checkpoint(bundle, model_key, requested_seed)
        candidates = [
            item
            for item in bundle.get("source_records", [])
            if item.get("model_key") == model_key and int(item.get("seed", -1)) == requested_seed
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Formal bundle has {len(candidates)} records for {model_key}:{requested_seed}.")
        return checkpoint, {
            "source": "formal_v3_main_results_bundle",
            "bundle": str(formal_bundle),
            "bundle_id": bundle.get("bundle_id"),
            "record": candidates[0].get("source_path"),
            "record_sha256": candidates[0].get("source_sha256"),
            "model_type": normalized_model,
            "model_key": model_key,
            "seed": requested_seed,
            "run_signature": candidates[0].get("run_signature"),
            "checkpoint_sha256": candidates[0].get("checkpoint_sha256"),
        }

    # Environment overrides are retained only for explicit diagnostics or
    # temporary compatibility fixtures; they cannot bypass the canonical gate.
    if override and (allow_legacy or root != Path(__file__).resolve().parents[1]):
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{override_name} does not point to a file: {path}")
        return path, {
            "source": "environment_override",
            "source_class": "explicit_diagnostic_override",
            "environment_variable": override_name,
            "model_type": normalized_model,
            "model_key": model_key,
            "seed": requested_seed,
            "run_signature": run_signature,
        }

    # Preserve the old fixture contract and require an explicit diagnostic flag
    # for the real project root, so active paper analyses cannot silently fall
    # through to partial_runs.json.
    if not allow_legacy and root == Path(__file__).resolve().parents[1]:
        raise FileNotFoundError(
            "Formal-v3 Main Results bundle is missing; legacy partial_runs.json "
            "requires allow_legacy=True."
        )

    registry_path = root / "experiments" / "exp1_sota" / "results" / "partial_runs.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"exp1 partial-run registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    active = registry.get("active_config", {})
    requested_signature = run_signature or os.getenv("HGV_EXP1_RUN_SIGNATURE", "").strip() or active.get("run_signature")
    if not requested_signature:
        raise RuntimeError("No exp1 run signature is available for the legacy diagnostic resolver.")
    default_seeds = active.get("random_seeds") or [42]
    if seed is None:
        requested_seed = int(os.getenv("HGV_EXP1_CHECKPOINT_SEED", default_seeds[0]))
    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, record in registry.get("runs", {}).items():
        if f"|model={normalized_model}|" not in key:
            continue
        if f"|signature={requested_signature}" not in key:
            continue
        if int(record.get("seed", -1)) != requested_seed:
            continue
        if record.get("checkpoint_path"):
            candidates.append((key, record))
    if not candidates:
        raise FileNotFoundError(
            "No completed exp1 checkpoint matches "
            f"model={normalized_model}, seed={requested_seed}, signature={requested_signature}."
        )
    key, record = max(candidates, key=lambda item: item[1].get("completed_at", ""))
    checkpoint = Path(record["checkpoint_path"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Registered exp1 checkpoint is missing: {checkpoint}")
    return checkpoint, {
        "source": "exp1_partial_runs",
        "source_class": "legacy_diagnostic",
        "registry": str(registry_path.resolve()),
        "registry_key": key,
        "model_type": normalized_model,
        "model_key": model_key,
        "seed": requested_seed,
        "run_signature": requested_signature,
        "completed_at": record.get("completed_at"),
    }

def _to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np  # local import to keep helper lightweight

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(path: str | Path, data: Any, ensure_ascii: bool = False) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=ensure_ascii)
    return p


def save_run_metadata(
    results_dir: str | Path,
    metadata: dict[str, Any],
    filename: str = "run_metadata.json",
    with_timestamp_copy: bool = True,
) -> dict[str, Path]:
    """
    Save run metadata into results directory.

    Returns saved paths mapping.
    """
    rd = ensure_dir(results_dir)
    latest = save_json(rd / filename, metadata, ensure_ascii=False)
    saved = {"latest": latest}
    if with_timestamp_copy:
        ts_name = f"run_metadata_{timestamp_str()}.json"
        ts_path = save_json(rd / ts_name, metadata, ensure_ascii=False)
        saved["timestamped"] = ts_path
    return saved


def save_experiment_results(
    results_dir: str | Path,
    filename: str,
    experiment: str,
    config: dict[str, Any],
    results: dict[str, Any],
    extra: dict[str, Any] | None = None,
    with_timestamp_copy: bool = True,
) -> dict[str, Path]:
    """
    Save unified experiment results payload.

    Payload schema:
    {
      "experiment": "...",
      "config": {...},
      "results": {...},
      ...extra
    }
    """
    payload: dict[str, Any] = {
        "experiment": experiment,
        "config": config,
        "results": results,
    }
    if extra:
        payload.update(extra)

    rd = ensure_dir(results_dir)
    latest = save_json(rd / filename, payload, ensure_ascii=False)
    saved = {"latest": latest}
    if with_timestamp_copy:
        stem = Path(filename).stem
        suffix = Path(filename).suffix or ".json"
        ts_name = f"{stem}_{timestamp_str()}{suffix}"
        ts_path = save_json(rd / ts_name, payload, ensure_ascii=False)
        saved["timestamped"] = ts_path
    return saved
