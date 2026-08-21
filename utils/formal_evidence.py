#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only normalization and evidence-gate helpers for formal-v3 records.

The module deliberately depends only on the Python standard library.  It is the
boundary used by ``formal status``, ``formal aggregate``, and paper generators;
training code, PyTorch, and dataset loaders must not be imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from utils.mainline_contract import ACTIVE_PLGAFORMER_FLAGS


CONFIG_REQUIRED_FIELDS = {
    "schema_version",
    "generation",
    "dataset",
    "task",
    "split",
    "training",
    "main_results",
    "ablation",
    "robustness",
    "efficiency",
    "studies",
    "external_sources",
    "artifact_layout",
}
FORMAL_STUDY_KEYS = ("main", "mechanism", "robustness", "efficiency")
REQUIRED_HORIZONS = (32, 64, 128, 256)
REQUIRED_METRICS = ("ade", "fde", "rmse_cart_m")
ANALYTICAL_MODEL_KEYS = {"kinematic", "rotating_3dof"}
FINAL_MODEL_FLAGS = ACTIVE_PLGAFORMER_FLAGS
DISPLAY_NAMES = {
    "baseline": "Transformer (baseline)",
    "full": "PLGAFormer (proposed)",
    "pit": "PIT",
    "kinematic": "Spherical kinematics",
    "rotating_3dof": "Rotating-Earth 3-DOF",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "af_ciln": "AF-CILN",
    "spherical_prior": "PLGAFormer spherical-prior fusion",
    "schedule_only": "PLGAFormer rotating-prior schedule only",
}


class FormalConfig:
    """Validated configuration plus its canonical content hash."""

    def __init__(self, path: Path, data: dict[str, Any], config_sha256: str) -> None:
        self.path = path
        self.data = data
        self.config_sha256 = config_sha256
        self.project_root = path.parent.parent

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def resolve(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def as_dict(self) -> dict[str, Any]:
        return dict(self.data)


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for all evidence hashes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(data, ensure_ascii=False))
    payload.pop("config_sha256", None)
    return payload


def load_formal_config(path: str | Path) -> FormalConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Formal configuration is missing: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Formal configuration is not valid JSON: {config_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Formal configuration root must be an object.")
    missing = sorted(CONFIG_REQUIRED_FIELDS.difference(data))
    if missing:
        raise ValueError(f"Formal configuration lacks fields: {', '.join(missing)}")
    if data.get("dataset", {}).get("protocol") != "hgv_multiregime_state_v2_1":
        raise ValueError("formal_v3 configuration must target hgv_multiregime_state_v2_1.")
    horizons = tuple(int(item) for item in data.get("task", {}).get("reporting_horizons", []))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(f"Formal horizons must be {list(REQUIRED_HORIZONS)}, got {list(horizons)}")
    seeds = tuple(int(item) for item in data.get("training", {}).get("seeds", []))
    if seeds != (42, 123, 456):
        raise ValueError(f"Formal seeds must be [42, 123, 456], got {list(seeds)}")
    test_id_hash = str(data.get("split", {}).get("test_trajectory_ids_sha256", ""))
    try:
        valid_test_id_hash = len(test_id_hash) == 64 and int(test_id_hash, 16) >= 0
    except ValueError:
        valid_test_id_hash = False
    if not valid_test_id_hash:
        raise ValueError("Formal configuration must sign the frozen test trajectory-ID set.")
    studies = data.get("studies")
    if not isinstance(studies, dict):
        raise ValueError("Formal configuration studies must be an object.")
    missing_studies = [name for name in FORMAL_STUDY_KEYS if name not in studies]
    if missing_studies:
        raise ValueError(
            "Formal configuration lacks study registrations: "
            + ", ".join(missing_studies)
        )
    for study_name in FORMAL_STUDY_KEYS:
        study = studies[study_name]
        if not isinstance(study, dict):
            raise ValueError(f"Formal study {study_name!r} must be an object.")
        required_study_fields = {"study_id", "runner", "input_bundle", "artifact_root", "paper_facing"}
        missing_fields = sorted(required_study_fields.difference(study))
        if missing_fields:
            raise ValueError(
                f"Formal study {study_name!r} lacks fields: {', '.join(missing_fields)}"
            )
        if study_name in {"main", "mechanism"} and not study.get("bundle_kind"):
            raise ValueError(f"Formal study {study_name!r} must declare bundle_kind.")
        if study_name == "main" and study.get("input_bundle") is not None:
            raise ValueError("The formal Main study cannot depend on another bundle.")
        if study_name != "main" and study.get("input_bundle") != "main":
            raise ValueError(
                f"Formal study {study_name!r} must declare input_bundle='main'."
            )
    attestations = data.get("generation", {}).get("legacy_record_attestations", [])
    if not isinstance(attestations, list):
        raise ValueError("legacy_record_attestations must be a list.")
    seen_attestations: set[str] = set()
    for item in attestations:
        if not isinstance(item, dict):
            raise ValueError("Each legacy record attestation must be an object.")
        source_hash = str(item.get("source_sha256", "")).lower()
        try:
            valid_source_hash = len(source_hash) == 64 and int(source_hash, 16) >= 0
        except ValueError:
            valid_source_hash = False
        if not valid_source_hash or not all(
            key in item for key in ("run_id", "model_key", "seed")
        ):
            raise ValueError("Legacy record attestation lacks a signed identity.")
        if source_hash in seen_attestations:
            raise ValueError(f"Duplicate legacy record attestation: {source_hash}")
        seen_attestations.add(source_hash)
    digest = sha256_bytes(canonical_json(_config_hash_payload(data)).encode("utf-8"))
    return FormalConfig(config_path, data, digest)


def _as_dict(config: FormalConfig | dict[str, Any]) -> dict[str, Any]:
    return config.data if isinstance(config, FormalConfig) else config


def resolve_config_path(config: FormalConfig | dict[str, Any], key: str) -> Path:
    if isinstance(config, FormalConfig):
        return config.resolve(_as_dict(config)["artifact_layout"][key])
    root = Path.cwd()
    return (root / _as_dict(config)["artifact_layout"][key]).resolve()


def get_formal_study(
    config: FormalConfig | dict[str, Any], study_name: str
) -> dict[str, Any]:
    """Return one registered paper-level study without introducing a second config."""
    if study_name not in FORMAL_STUDY_KEYS:
        raise KeyError(f"Unknown formal study: {study_name}")
    studies = _as_dict(config).get("studies", {})
    study = studies.get(study_name)
    if not isinstance(study, dict):
        raise KeyError(f"Formal study is not registered: {study_name}")
    return study


def resolve_study_artifact_root(
    config: FormalConfig | dict[str, Any], study_name: str
) -> Path:
    """Resolve a registered study root relative to the formal config project."""
    study = get_formal_study(config, study_name)
    if isinstance(config, FormalConfig):
        return config.resolve(study["artifact_root"])
    return (Path.cwd() / study["artifact_root"]).resolve()


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "on"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _as_horizons(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, dict):
        value = list(value)
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _resolve_reference(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _relocate_project_reference(path: Path) -> tuple[Path, Path | None]:
    """Relocate a missing absolute path after the repository root moved.

    Formal-v3 records created in the OneDrive working copy contain absolute
    paths. The code repository now lives under ``D:\\Research``. Relocation is
    limited to the suffix below the same repository directory name; callers
    must still verify the recorded content hash.
    """
    original = path.expanduser().resolve()
    if original.exists():
        return original, None

    project_root = Path(__file__).resolve().parents[1]
    marker = project_root.name.casefold()
    matching_indices = [
        index
        for index, part in enumerate(original.parts)
        if part.casefold() == marker
    ]
    for index in reversed(matching_indices):
        suffix = original.parts[index + 1 :]
        if not suffix:
            continue
        candidate = (project_root / Path(*suffix)).resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate, original
    return original, None


def _protocol_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("protocol_identity")
    return value if isinstance(value, dict) else {}


def _dataset_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("dataset_identity")
    return value if isinstance(value, dict) else {}


def _run_config(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("config")
    return value if isinstance(value, dict) else {}


def _canonical_model_key(payload: dict[str, Any]) -> str:
    model_key = str(payload.get("model_key", "")).strip().lower()
    aliases = {
        "transformer": "baseline",
        "proposed": "full",
        "plgaformer": "full",
        "prior_only": "full",
        "spherical-prior": "spherical_prior",
        "schedule-only": "schedule_only",
    }
    return aliases.get(model_key, model_key)


def _model_config_hash(model_config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(model_config).encode("utf-8"))


def _scientific_protocol_core(payload: dict[str, Any], horizons: list[int]) -> dict[str, Any]:
    identity = _protocol_identity(payload)
    dataset = _dataset_identity(payload)
    run_config = _run_config(payload)
    return {
        "dataset_protocol": str(_first(identity.get("dataset_protocol"), dataset.get("dataset_protocol"), payload.get("dataset_protocol"), default="")),
        "dataset_sha256": str(_first(identity.get("dataset_sha256"), dataset.get("dataset_sha256"), payload.get("dataset_sha256"), default="")),
        "sampling_interval_s": float(_first(identity.get("sampling_interval_s"), dataset.get("sampling_interval_s"), default=1.0)),
        "subset_ratio": float(_first(identity.get("subset_ratio"), run_config.get("subset_ratio"), default=1.0)),
        "epochs": int(_first(identity.get("epochs"), run_config.get("epochs"), default=-1)),
        "batch_size": int(_first(identity.get("batch_size"), run_config.get("batch_size"), default=-1)),
        "prediction_length": int(_first(identity.get("prediction_length"), run_config.get("prediction_length"), default=-1)),
        "prediction_horizons": list(horizons),
        "train_supervision_protocol": str(_first(identity.get("train_supervision_protocol"), run_config.get("train_supervision_protocol"), default="")),
        "eval_protocol": str(_first(identity.get("eval_protocol"), run_config.get("eval_protocol"), default="")),
        "eval_ar_seed_mode": str(_first(identity.get("eval_ar_seed_mode"), run_config.get("eval_ar_seed_mode"), default="")),
        "label_len": int(_first(identity.get("label_len"), run_config.get("label_len"), default=-1)),
        "learning_rate": float(_first(identity.get("learning_rate"), run_config.get("learning_rate"), default=-1.0)),
        "weight_decay": float(_first(identity.get("weight_decay"), run_config.get("weight_decay"), default=-1.0)),
        "warmup_epochs": int(_first(identity.get("warmup_epochs"), identity.get("warmup_epochs"), run_config.get("warmup_epochs"), default=-1)),
        "patience": int(_first(identity.get("patience"), run_config.get("patience"), default=-1)),
        "gradient_clip_norm": float(_first(identity.get("gradient_clip_norm"), run_config.get("gradient_clip_norm"), default=-1.0)),
        "mixed_precision": _as_bool(_first(identity.get("mixed_precision"), run_config.get("amp"), default=False)) is True,
        "mixed_precision_dtype": str(_first(identity.get("mixed_precision_dtype"), run_config.get("amp_dtype"), default="float32")),
        "cache_physics_prior": _as_bool(_first(identity.get("cache_physics_prior"), run_config.get("cache_physics_prior"), default=False)) is True,
        "physics_prior_cache_batch_size": int(_first(identity.get("physics_prior_cache_batch_size"), run_config.get("physics_prior_cache_batch_size"), default=0)),
    }


def _external_provenance(payload: dict[str, Any], model_key: str) -> dict[str, Any] | None:
    raw = payload.get("external_source_provenance")
    if isinstance(raw, dict):
        return dict(raw)
    if model_key != "af_ciln":
        return None
    run_config = _run_config(payload)
    model_config = payload.get("model_config")
    model_config = model_config if isinstance(model_config, dict) else {}
    root = _first(
        run_config.get("af_ciln_root"),
        model_config.get("af_ciln_root"),
        default=None,
    )
    provenance = {"kind": "AF-CILN", "root": str(root) if root else ""}
    for key in ("source_hash", "commit_hash", "source_root"):
        if key in model_config:
            provenance[key] = model_config[key]
    return provenance


def normalize_run_record(path: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    if payload is None:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Formal run record must be an object: {source_path}")
    model_key = _canonical_model_key(payload)
    model_config = payload.get("model_config")
    model_config = dict(model_config) if isinstance(model_config, dict) else {}
    identity = _protocol_identity(payload)
    dataset = _dataset_identity(payload)
    horizons = _as_horizons(
        _first(payload.get("horizons"), identity.get("prediction_horizons"), default=[])
    )
    checkpoint_value = payload.get("checkpoint")
    recorded_checkpoint = (
        _resolve_reference(checkpoint_value, source_path.parent)
        if checkpoint_value
        else None
    )
    checkpoint = recorded_checkpoint
    relocated_checkpoint = None
    if recorded_checkpoint is not None:
        checkpoint, relocated_checkpoint = _relocate_project_reference(recorded_checkpoint)
    external_provenance = _external_provenance(payload, model_key)
    if isinstance(external_provenance, dict):
        for key in ("root", "source_root"):
            value = external_provenance.get(key)
            if not value:
                continue
            recorded_root = _resolve_reference(value, source_path.parent)
            resolved_root, relocated_root = _relocate_project_reference(recorded_root)
            if relocated_root is not None:
                external_provenance[f"recorded_{key}"] = str(relocated_root)
                external_provenance[key] = str(resolved_root)
    record = {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "schema_version": payload.get("schema_version"),
        "run_id": str(payload.get("run_id", source_path.stem)),
        "experiment": str(payload.get("experiment", "exp1_sota" if "formal_sota" in source_path.name else "exp2_ablation")),
        "phase": payload.get("phase"),
        "timestamp": payload.get("timestamp"),
        "seed": int(payload.get("seed", -1)),
        "model": str(payload.get("model", DISPLAY_NAMES.get(model_key, model_key))),
        "model_key": model_key,
        "model_config": model_config,
        "model_config_sha256": _model_config_hash(model_config),
        "formal_config_sha256": str(payload.get("formal_config_sha256", "")),
        "dataset_protocol": str(_first(identity.get("dataset_protocol"), dataset.get("dataset_protocol"), payload.get("dataset_protocol"), default="")),
        "dataset_sha256": str(_first(identity.get("dataset_sha256"), dataset.get("dataset_sha256"), payload.get("dataset_sha256"), default="")),
        "dataset_identity": dataset,
        "scientific_protocol_core": _scientific_protocol_core(payload, horizons),
        "run_signature": str(_first(identity.get("run_signature"), payload.get("run_signature"), default="")),
        "horizons": horizons,
        "eval_results": payload.get("eval_results") if isinstance(payload.get("eval_results"), dict) else {},
        "evidence_tier": payload.get("evidence_tier"),
        "test_evaluation_performed": _as_bool(payload.get("test_evaluation_performed")),
        "checkpoint": str(checkpoint) if checkpoint else "",
        "checkpoint_recorded": str(relocated_checkpoint) if relocated_checkpoint else "",
        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
        "source_selection_record": payload.get("source_selection_record"),
        "external_source_provenance": external_provenance,
        "raw_payload": payload,
    }
    return record


def discover_run_records(root: str | Path, experiment: str = "main") -> list[dict[str, Any]]:
    """Discover only per-run JSON files; manifests and summaries are excluded."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        return []
    if str(experiment).lower() in {"main", "sota", "exp1_sota"}:
        patterns = ("formal_sota_*.json",)
    elif str(experiment).lower() in {"ablation", "exp2_ablation"}:
        patterns = ("formal_*.json",)
    else:
        patterns = ("*.json",)
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root_path.rglob(pattern)
            if path.name not in {"main_run_set_manifest.json", "ablation_run_set_manifest.json"}
            and not path.name.endswith("_summary.json")
            and not path.name.endswith("_stats.json")
        },
        key=lambda path: str(path).lower(),
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            records.append(normalize_run_record(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Keep malformed files visible to the caller without making status
            # crash before it can report the source path.
            records.append(
                {
                    "source_path": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                    "model_key": "",
                    "seed": -1,
                    "malformed": True,
                    "raw_payload": None,
                    "normalization_error": "invalid_json_or_record",
                }
            )
    return records


def _blocker(code: str, message: str, **context: Any) -> dict[str, Any]:
    item = {"code": code, "message": message}
    if context:
        item["context"] = context
    return item


def _expected_core(config: FormalConfig | dict[str, Any]) -> dict[str, Any]:
    data = _as_dict(config)
    task = data["task"]
    train = data["training"]
    return {
        "dataset_protocol": data["dataset"]["protocol"],
        "dataset_sha256": data["dataset"]["sha256"],
        "sampling_interval_s": float(data["dataset"]["sampling_interval_s"]),
        "subset_ratio": 1.0,
        "epochs": int(train["epochs_max"]),
        "batch_size": int(train["batch_size"]),
        "prediction_length": int(task["pred_len"]),
        "prediction_horizons": [int(item) for item in task["reporting_horizons"]],
        "train_supervision_protocol": task["train_supervision_protocol"],
        "eval_protocol": task["eval_protocol"],
        "eval_ar_seed_mode": task["eval_ar_seed_mode"],
        "label_len": int(task["label_len"]),
        "learning_rate": float(train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]),
        "warmup_epochs": int(train["warmup_epochs"]),
        "patience": int(train["patience"]),
        "gradient_clip_norm": float(train["gradient_clip_norm"]),
        "mixed_precision": bool(train["amp"]),
        "mixed_precision_dtype": str(train["precision"]),
        "cache_physics_prior": bool(train["cache_physics_prior"]),
        "physics_prior_cache_batch_size": int(train["physics_prior_cache_batch_size"]),
    }


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= max(1e-12, abs(expected) * 1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _same_flag(actual: Any, expected: Any) -> bool:
    """Compare resolved model flags without accepting truthy numeric aliases."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    return isinstance(actual, str) and actual == expected


def _record_metric_item(record: dict[str, Any], horizon: int) -> dict[str, Any]:
    results = record.get("eval_results", {})
    if not isinstance(results, dict):
        return {}
    return results.get(str(horizon), results.get(horizon, {})) or {}


def _metric_value(item: dict[str, Any], metric: str) -> Any:
    aliases = {
        "ade": ("ade", "trajectory_window_ade"),
        "fde": ("fde", "trajectory_window_fde"),
        "rmse_cart_m": ("rmse_cart_m",),
    }
    for key in aliases.get(metric, (metric,)):
        if key in item and item[key] is not None:
            return item[key]
    return None
def _metric_is_finite(item: dict[str, Any], metric: str) -> bool:
    value = _metric_value(item, metric)
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False




def final_model_configuration_blockers(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Require explicit, machine-readable provenance for the final PLGAFormer."""
    blockers: list[dict[str, Any]] = []
    model_config = record.get("model_config")
    source_path = record.get("source_path")
    if not isinstance(model_config, dict):
        return [
            _blocker(
                "WRONG_FINAL_MODEL_CONFIGURATION",
                "Final PLGAFormer resolved model configuration is missing.",
                source_path=source_path,
                field="model_config",
            )
        ]

    innovations = model_config.get("innovations")
    if innovations in (None, "", [], {}):
        blockers.append(
            _blocker(
                "WRONG_FINAL_MODEL_CONFIGURATION",
                "Final PLGAFormer innovations provenance must be non-empty.",
                source_path=source_path,
                field="innovations",
                observed=innovations,
            )
        )

    nested_flags = model_config.get("resolved_model_flags")
    if not isinstance(nested_flags, dict):
        nested_flags = {}
    if isinstance(innovations, dict):
        nested_flags = {**innovations, **nested_flags}

    # Pre-consolidation formal-v3 records retained the actual best-checkpoint
    # gate state in immutable training history rather than model_config.
    raw_payload = record.get("raw_payload")
    raw_payload = raw_payload if isinstance(raw_payload, dict) else {}
    training_history = raw_payload.get("training_history")
    training_history = training_history if isinstance(training_history, dict) else {}
    gate_snapshot = training_history.get("best_model_gate_snapshot")
    gate_snapshot = gate_snapshot if isinstance(gate_snapshot, dict) else {}
    if "use_physics_attention" in gate_snapshot and "use_sparse_attention" not in gate_snapshot:
        gate_snapshot = {
            **gate_snapshot,
            # PLGAFormer.physics_gate_snapshot() historically exposed
            # use_sparse_attention under this diagnostic name.
            "use_sparse_attention": gate_snapshot["use_physics_attention"],
        }
    nested_flags = {**gate_snapshot, **nested_flags}

    for key, expected in FINAL_MODEL_FLAGS.items():
        if key in model_config:
            observed = model_config[key]
        elif key in nested_flags:
            observed = nested_flags[key]
        else:
            blockers.append(
                _blocker(
                    "WRONG_FINAL_MODEL_CONFIGURATION",
                    f"Final PLGAFormer field {key} is not explicit.",
                    source_path=source_path,
                    field=key,
                    expected=expected,
                )
            )
            continue
        if not _same_flag(observed, expected):
            blockers.append(
                _blocker(
                    "WRONG_FINAL_MODEL_CONFIGURATION",
                    f"Final PLGAFormer field {key} has the wrong value.",
                    source_path=source_path,
                    field=key,
                    expected=expected,
                    observed=observed,
                )
            )
    return blockers


def _legacy_config_attestation(
    record: dict[str, Any], config: FormalConfig | dict[str, Any]
) -> dict[str, Any] | None:
    """Return an exact source-hash attestation for a pre-config-hash record."""
    generation = _as_dict(config).get("generation", {})
    attestations = generation.get("legacy_record_attestations", [])
    if not isinstance(attestations, list):
        return None
    source_hash = str(record.get("source_sha256", "")).lower()
    for item in attestations:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_sha256", "")).lower() != source_hash:
            continue
        if str(item.get("run_id", "")) != str(record.get("run_id", "")):
            continue
        if str(item.get("model_key", "")) != str(record.get("model_key", "")):
            continue
        try:
            seed_matches = int(item.get("seed", -1)) == int(record.get("seed", -1))
        except (TypeError, ValueError):
            seed_matches = False
        if seed_matches:
            return item
    return None


def _trajectory_metric_blockers(
    record: dict[str, Any], config: FormalConfig | dict[str, Any]
) -> list[dict[str, Any]]:
    """Require the frozen test-trajectory population at every horizon."""
    blockers: list[dict[str, Any]] = []
    split = _as_dict(config).get("split", {})
    expected_count = int(split.get("complete_trajectory_counts", {}).get("test", 0))
    expected_id_hash = str(split.get("test_trajectory_ids_sha256", ""))
    observed_sets: list[list[str]] = []
    for horizon in REQUIRED_HORIZONS:
        item = _record_metric_item(record, horizon)
        metrics = item.get("trajectory_metrics") if isinstance(item, dict) else None
        if not isinstance(metrics, dict):
            blockers.append(_blocker(
                "MISSING_TRAJECTORY_METRICS",
                "Per-trajectory metrics are required for paper-facing paired inference.",
                source_path=record.get("source_path"),
                horizon=horizon,
            ))
            continue
        try:
            ids = sorted((str(int(value)) for value in metrics), key=int)
        except (TypeError, ValueError):
            ids = []
        observed_sets.append(ids)
        if len(ids) != expected_count:
            blockers.append(_blocker(
                "INCOMPLETE_TEST_TRAJECTORY_COVERAGE",
                "Per-trajectory metrics do not cover the frozen test population.",
                source_path=record.get("source_path"), horizon=horizon,
                expected_count=expected_count, observed_count=len(ids),
            ))
        observed_hash = sha256_bytes(",".join(ids).encode("utf-8")) if ids else ""
        if expected_id_hash and observed_hash != expected_id_hash:
            blockers.append(_blocker(
                "TEST_TRAJECTORY_ID_SET_MISMATCH",
                "Per-trajectory metrics use a different trajectory-ID set from the frozen test split.",
                source_path=record.get("source_path"), horizon=horizon,
                expected=expected_id_hash, observed=observed_hash,
            ))
        declared_count = item.get("trajectory_count")
        try:
            declared_count_matches = (
                declared_count is None or int(declared_count) == len(ids)
            )
        except (TypeError, ValueError):
            declared_count_matches = False
        if not declared_count_matches:
            blockers.append(_blocker(
                "TRAJECTORY_COUNT_METADATA_MISMATCH",
                "Declared trajectory_count disagrees with trajectory_metrics.",
                source_path=record.get("source_path"), horizon=horizon,
                declared=declared_count, observed=len(ids),
            ))
        invalid = next((
            trajectory_id for trajectory_id, values in metrics.items()
            if not isinstance(values, dict)
            or any(not _metric_is_finite(values, metric) for metric in ("ade", "fde"))
        ), None)
        if invalid is not None:
            blockers.append(_blocker(
                "INCOMPLETE_TRAJECTORY_METRIC",
                "Each test trajectory must contain ADE and FDE.",
                source_path=record.get("source_path"), horizon=horizon, trajectory_id=invalid,
            ))
    if observed_sets and any(ids != observed_sets[0] for ids in observed_sets[1:]):
        blockers.append(_blocker(
            "INCONSISTENT_TRAJECTORY_IDS_ACROSS_HORIZONS",
            "All reporting horizons must use the same frozen test trajectories.",
            source_path=record.get("source_path"),
        ))
    return blockers


def _record_blockers(
    record: dict[str, Any],
    config: FormalConfig | dict[str, Any],
    *,
    required_metrics: Iterable[str] = REQUIRED_METRICS,
) -> list[dict[str, Any]]:
    data = _as_dict(config)
    blockers: list[dict[str, Any]] = []
    if record.get("malformed"):
        return [_blocker("MALFORMED_RUN_RECORD", "Run record could not be normalized.", source_path=record.get("source_path"))]
    expected_core = _expected_core(config)
    core = record.get("scientific_protocol_core", {})
    recorded_config_hash = str(record.get("formal_config_sha256", ""))
    expected_config_hash = config.config_sha256 if isinstance(config, FormalConfig) else ""
    legacy_attestation = _legacy_config_attestation(record, config)
    if expected_config_hash and not recorded_config_hash and legacy_attestation is None:
        blockers.append(
            _blocker(
                "MISSING_FORMAL_CONFIG_HASH",
                "Run record does not identify the frozen formal-v3 configuration.",
                source_path=record.get("source_path"),
                expected=expected_config_hash,
            )
        )
    elif expected_config_hash and not recorded_config_hash:
        record["formal_config_binding"] = "legacy_source_sha256_attestation"
        record["formal_config_attestation"] = dict(legacy_attestation)
    elif recorded_config_hash and expected_config_hash and recorded_config_hash != expected_config_hash:
        blockers.append(
            _blocker(
                "FORMAL_CONFIG_HASH_MISMATCH",
                "Run record was produced under another frozen formal configuration.",
                source_path=record.get("source_path"),
                expected=expected_config_hash,
                observed=recorded_config_hash,
            )
        )
    for key, expected in expected_core.items():
        if not _same_value(core.get(key), expected):
            blockers.append(
                _blocker(
                    "PROTOCOL_CORE_MISMATCH",
                    f"{key} does not match the frozen formal-v3 protocol.",
                    source_path=record.get("source_path"),
                    field=key,
                    expected=expected,
                    observed=core.get(key),
                )
            )
    if record.get("evidence_tier") != data["main_results"]["required_evidence_tier"]:
        blockers.append(_blocker("WRONG_EVIDENCE_TIER", "Record is not in the required final evidence tier.", source_path=record.get("source_path"), observed=record.get("evidence_tier")))
    if record.get("test_evaluation_performed") is not True:
        blockers.append(_blocker("TEST_EVALUATION_NOT_FROZEN", "Record does not document one frozen test evaluation.", source_path=record.get("source_path")))
    if record.get("horizons") != list(REQUIRED_HORIZONS):
        blockers.append(_blocker("WRONG_HORIZONS", "Record does not contain the four reporting horizons.", source_path=record.get("source_path"), observed=record.get("horizons"), expected=list(REQUIRED_HORIZONS)))
    checkpoint = Path(str(record.get("checkpoint", "")))
    if not checkpoint.is_file():
        blockers.append(_blocker("MISSING_CHECKPOINT", "Signed checkpoint is missing.", source_path=record.get("source_path"), checkpoint=str(checkpoint)))
    else:
        recorded_hash = str(record.get("checkpoint_sha256", ""))
        if not recorded_hash:
            blockers.append(_blocker("MISSING_CHECKPOINT_HASH", "Checkpoint hash is not recorded.", source_path=record.get("source_path"), checkpoint=str(checkpoint)))
        else:
            actual_hash = sha256_file(checkpoint)
            if recorded_hash.lower() != actual_hash.lower():
                blockers.append(_blocker("CHECKPOINT_HASH_MISMATCH", "Checkpoint bytes do not match the recorded hash.", source_path=record.get("source_path"), checkpoint=str(checkpoint), expected=recorded_hash, observed=actual_hash))
    for horizon in REQUIRED_HORIZONS:
        item = _record_metric_item(record, horizon)
        for metric in required_metrics:
            if not _metric_is_finite(item, metric):
                blockers.append(_blocker("MISSING_REQUIRED_METRIC", f"Missing {metric} at horizon {horizon}.", source_path=record.get("source_path"), horizon=horizon, metric=metric))
    blockers.extend(_trajectory_metric_blockers(record, config))
    if not record.get("model_config"):
        blockers.append(_blocker("MISSING_MODEL_CONFIGURATION", "Resolved model configuration is absent.", source_path=record.get("source_path")))
    if record.get("model_key") == "full":
        blockers.extend(final_model_configuration_blockers(record))
    if record.get("model_key") == "af_ciln":
        provenance = record.get("external_source_provenance") or {}
        root_value = provenance.get("root") or provenance.get("source_root")
        if not root_value or not Path(str(root_value)).is_dir():
            blockers.append(_blocker("INVALID_EXTERNAL_SOURCE_PROVENANCE", "AF-CILN source checkout is missing or not recorded.", source_path=record.get("source_path"), root=root_value))
    public_source_keys = {
        str(item) for item in data["main_results"].get("public_source_model_keys", [])
    }
    if record.get("model_key") in public_source_keys:
        provenance = record.get("external_source_provenance") or {}
        tslib_config = data.get("external_sources", {}).get("tslib", {})
        required_fields = {
            "kind": "TSLib",
            "repository": "https://github.com/thuml/Time-Series-Library",
            "commit": str(tslib_config.get("commit", "")),
            "license": "MIT",
        }
        missing_or_mismatched = [
            key for key, expected in required_fields.items()
            if not provenance.get(key) or (expected and str(provenance.get(key)) != expected)
        ]
        expected_adapter_config = tslib_config.get("adapter_config")
        observed_adapter_config = provenance.get("adapter_config")
        if expected_adapter_config and observed_adapter_config != expected_adapter_config:
            missing_or_mismatched.append("adapter_config")
        source_files = provenance.get("source_files")
        if not isinstance(source_files, dict) or not source_files:
            missing_or_mismatched.append("source_files")
        else:
            try:
                from models.public_baselines import (
                    TSLIB_FILE_SHA256,
                    _required_files,
                )

                source_model_type = (
                    "transformer"
                    if record.get("model_key") == "baseline"
                    else str(record.get("model_config", {}).get("model_type", ""))
                )
                expected_files = _required_files([source_model_type])
                for relative in expected_files:
                    expected_hash = TSLIB_FILE_SHA256.get(relative)
                    if str(source_files.get(relative, "")).lower() != str(expected_hash or "").lower():
                        missing_or_mismatched.append(f"source_files:{relative}")
            except (ImportError, KeyError, ValueError):
                missing_or_mismatched.append("source_files:expected_hash_registry")
        if missing_or_mismatched:
            blockers.append(
                _blocker(
                    "INVALID_PUBLIC_SOURCE_PROVENANCE",
                    "Paper-facing public baseline lacks the pinned TSLib source identity.",
                    source_path=record.get("source_path"),
                    model_key=record.get("model_key"),
                    fields=missing_or_mismatched,
                )
            )
    return blockers


def validate_protocol_core(
    records: list[dict[str, Any]], config: FormalConfig | dict[str, Any]
) -> dict[str, Any]:
    expected = _expected_core(config)
    blockers: list[dict[str, Any]] = []
    observed: dict[str, int] = {}
    per_record: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        path = str(record.get("source_path", ""))
        core = record.get("scientific_protocol_core", {})
        key = canonical_json(core)
        observed[key] = observed.get(key, 0) + 1
        issues = []
        for field, value in expected.items():
            if not _same_value(core.get(field), value):
                issues.append(_blocker("PROTOCOL_CORE_MISMATCH", f"{field} does not match the frozen formal-v3 protocol.", field=field, expected=value, observed=core.get(field)))
        if issues:
            per_record[path] = issues
            blockers.extend(issues)
    return {
        "kind": "protocol",
        "passed": not blockers,
        "paper_eligible": not blockers,
        "config_sha256": config.config_sha256 if isinstance(config, FormalConfig) else None,
        "expected_core": expected,
        "observed_core_count": len(observed),
        "record_count": len(records),
        "record_issues": per_record,
        "blockers": blockers,
    }


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        try:
            key = (str(record.get("model_key", "")), int(record.get("seed", -1)))
        except (TypeError, ValueError):
            key = (str(record.get("model_key", "")), -1)
        grouped.setdefault(key, []).append(record)
    return grouped


def _metric_fingerprint(record: dict[str, Any]) -> str:
    payload = {}
    for horizon in REQUIRED_HORIZONS:
        item = _record_metric_item(record, horizon)
        payload[str(horizon)] = {
            metric: _metric_value(item, metric) for metric in REQUIRED_METRICS
        }
    return canonical_json(payload)


def _selected_record_metadata(record: dict[str, Any], *, source: str = "formal_v3") -> dict[str, Any]:
    return {
        "source": source,
        "source_path": record.get("source_path"),
        "source_sha256": record.get("source_sha256"),
        "run_id": record.get("run_id"),
        "model": record.get("model"),
        "model_key": record.get("model_key"),
        "seed": int(record.get("seed", -1)),
        "model_config_sha256": record.get("model_config_sha256"),
        "formal_config_sha256": record.get("formal_config_sha256"),
        "formal_config_binding": record.get("formal_config_binding"),
        "formal_config_attestation": record.get("formal_config_attestation"),
        "dataset_protocol": record.get("dataset_protocol"),
        "dataset_sha256": record.get("dataset_sha256"),
        "checkpoint": record.get("checkpoint"),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "run_signature": record.get("run_signature"),
        "phase": record.get("phase"),
        "timestamp": record.get("timestamp"),
        "external_source_provenance": record.get("external_source_provenance"),
    }


def validate_main_completeness(
    records: list[dict[str, Any]], config: FormalConfig | dict[str, Any]
) -> dict[str, Any]:
    data = _as_dict(config)
    trainable = [str(item) for item in data["main_results"]["trainable_model_keys"]]
    analytical = data["main_results"].get("analytical_models", {})
    required_keys = trainable + [str(item) for item in analytical]
    selected: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    protocol_audit = validate_protocol_core(records, config)
    blockers.extend(protocol_audit["blockers"])
    grouped = _group_records(records)
    matrix: dict[str, dict[str, Any]] = {}
    record_issues: dict[str, list[dict[str, Any]]] = {}
    for model_key in required_keys:
        model_rows = {seed: rows for (key, seed), rows in grouped.items() if key == model_key}
        expected_seeds = [42, 123, 456] if model_key in trainable else [int(analytical[model_key].get("anchor_seed", 42))]
        matrix[model_key] = {
            "required_seeds": expected_seeds,
            "present_seeds": sorted(model_rows),
            "missing_seeds": [seed for seed in expected_seeds if seed not in model_rows],
            "duplicates": {},
        }
        for seed in expected_seeds:
            rows = model_rows.get(seed, [])
            if not rows:
                blockers.append(_blocker("MISSING_REQUIRED_RUN", f"Missing formal-v3 run for {model_key}:{seed}.", model_key=model_key, seed=seed))
                continue
            valid_rows = []
            for row in rows:
                issues = _record_blockers(row, config, required_metrics=data["main_results"].get("required_metrics", REQUIRED_METRICS))
                if issues:
                    record_issues[str(row.get("source_path"))] = issues
                    blockers.extend(issues)
                else:
                    valid_rows.append(row)
            if len(rows) > 1:
                fingerprints = {_metric_fingerprint(row) for row in rows if not row.get("malformed")}
                matrix[model_key]["duplicates"][str(seed)] = {
                    "count": len(rows),
                    "identical_metrics": len(fingerprints) == 1,
                    "source_paths": [row.get("source_path") for row in rows],
                }
                if model_key in ANALYTICAL_MODEL_KEYS and len(fingerprints) == 1:
                    chosen = valid_rows[0] if valid_rows else rows[0]
                else:
                    blockers.append(_blocker("AMBIGUOUS_DUPLICATE_RECORDS", f"Multiple non-identical or non-deterministic records exist for {model_key}:{seed}.", model_key=model_key, seed=seed, source_paths=[row.get("source_path") for row in rows]))
                    chosen = valid_rows[0] if len(valid_rows) == 1 else None
            else:
                chosen = valid_rows[0] if valid_rows else None
            if chosen is not None and not any(item.get("source_path") == chosen.get("source_path") for item in selected):
                selected.append(_selected_record_metadata(chosen))
        if model_key in analytical:
            extra = []
            for (key, seed), rows in grouped.items():
                if key == model_key and seed not in expected_seeds:
                    extra.extend(rows)
            if extra:
                fingerprints = {_metric_fingerprint(row) for row in extra}
                anchor_rows = model_rows.get(expected_seeds[0], [])
                anchor_fingerprints = {_metric_fingerprint(row) for row in anchor_rows}
                if fingerprints and anchor_fingerprints and fingerprints != anchor_fingerprints:
                    blockers.append(_blocker("DETERMINISTIC_DUPLICATE_MISMATCH", f"Extra deterministic records for {model_key} are not numerically identical to the anchor.", model_key=model_key, source_paths=[row.get("source_path") for row in extra]))
    expected_set = set(required_keys)
    unexpected = [row for row in records if row.get("model_key") not in expected_set]
    for row in unexpected:
        blockers.append(
            _blocker(
                "UNEXPECTED_MODEL_RECORD",
                "Main Results contains a model key outside the frozen paper matrix.",
                source_path=row.get("source_path"),
                model_key=row.get("model_key"),
            )
        )
    return {
        "kind": "main",
        "passed": not blockers,
        "paper_eligible": not blockers,
        "config_sha256": config.config_sha256 if isinstance(config, FormalConfig) else None,
        "record_count": len(records),
        "required_model_keys": required_keys,
        "matrix": matrix,
        "selected_records": selected,
        "record_issues": record_issues,
        "unexpected_records": [_selected_record_metadata(row, source="unexpected") for row in unexpected],
        "blockers": blockers,
        "claim_boundary": "complete simulated HGV trajectories; simulation-only evidence",
    }


def validate_ablation_completeness(
    records: list[dict[str, Any]],
    main_bundle: dict[str, Any],
    config: FormalConfig | dict[str, Any],
) -> dict[str, Any]:
    data = _as_dict(config)
    ablation_cfg = data["ablation"]
    required_keys = [str(item) for item in ablation_cfg["required_trainable_model_keys"]]
    seeds = [int(item) for item in ablation_cfg["required_seeds"]]
    blockers: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    main_records = []
    if isinstance(main_bundle, dict):
        main_records = main_bundle.get("selected_records") or main_bundle.get("source_records") or []
    main_lookup = {(str(row.get("model_key")), int(row.get("seed", -1))): row for row in main_records}
    if isinstance(main_bundle, dict):
        main_kind = main_bundle.get("bundle_kind")
        if main_kind is not None and main_kind != "main":
            blockers.append(
                _blocker(
                    "MAIN_BUNDLE_KIND_MISMATCH",
                    "Ablation controls must reference a Main Results bundle.",
                    observed=main_kind,
                )
            )
        expected_config_hash = str(_as_dict(config).get("config_sha256", ""))
        if isinstance(config, FormalConfig):
            expected_config_hash = config.config_sha256
        observed_config_hash = str(main_bundle.get("config_sha256", ""))
        if observed_config_hash and expected_config_hash and observed_config_hash != expected_config_hash:
            blockers.append(
                _blocker(
                    "MAIN_BUNDLE_CONFIG_HASH_MISMATCH",
                    "Ablation controls reference a Main Results bundle from another configuration.",
                    expected=expected_config_hash,
                    observed=observed_config_hash,
                )
            )
        expected_dataset_hash = str(_as_dict(config)["dataset"]["sha256"])
        observed_dataset_hash = str(main_bundle.get("dataset_sha256", ""))
        if observed_dataset_hash and observed_dataset_hash != expected_dataset_hash:
            blockers.append(
                _blocker(
                    "MAIN_BUNDLE_DATASET_HASH_MISMATCH",
                    "Ablation controls reference a Main Results bundle from another dataset.",
                    expected=expected_dataset_hash,
                    observed=observed_dataset_hash,
                )
            )
    for control_key in ("baseline", "full"):
        for seed in seeds:
            source = main_lookup.get((control_key, seed))
            if source is None:
                blockers.append(_blocker("MISSING_REUSED_MAIN_CONTROL", f"Ablation control {control_key}:{seed} is unavailable from the Main Results bundle.", model_key=control_key, seed=seed))
            else:
                selected.append(dict(source, source="main_results_bundle", reused_for_ablation=True))
    protocol_audit = validate_protocol_core(records, config)
    blockers.extend(protocol_audit["blockers"])
    grouped = _group_records(records)
    matrix: dict[str, Any] = {}
    record_issues: dict[str, list[dict[str, Any]]] = {}
    for model_key in required_keys:
        matrix[model_key] = {"required_seeds": seeds, "present_seeds": [], "missing_seeds": []}
        for seed in seeds:
            rows = grouped.get((model_key, seed), [])
            if not rows:
                matrix[model_key]["missing_seeds"].append(seed)
                blockers.append(_blocker("MISSING_REQUIRED_ABLATION_RUN", f"Missing formal-v3 ablation run for {model_key}:{seed}.", model_key=model_key, seed=seed))
                continue
            matrix[model_key]["present_seeds"].append(seed)
            if len(rows) != 1:
                blockers.append(_blocker("AMBIGUOUS_DUPLICATE_ABLATION_RECORDS", f"Multiple ablation records exist for {model_key}:{seed}.", model_key=model_key, seed=seed, source_paths=[row.get("source_path") for row in rows]))
                continue
            row = rows[0]
            issues = _record_blockers(row, config, required_metrics=ablation_cfg.get("required_metrics", REQUIRED_METRICS))
            if str(row.get("phase", "")) != str(ablation_cfg["phase"]):
                issues.append(
                    _blocker(
                        "WRONG_ABLATION_PHASE",
                        "Ablation record is not from the paper-required phase4 mechanism-control run.",
                        source_path=row.get("source_path"),
                        expected=ablation_cfg["phase"],
                        observed=row.get("phase"),
                    )
                )
            if issues:
                record_issues[str(row.get("source_path"))] = issues
                blockers.extend(issues)
            else:
                selected.append(_selected_record_metadata(row))
    expected_ablation_keys = set(required_keys)
    for row in records:
        if row.get("model_key") not in expected_ablation_keys:
            blockers.append(
                _blocker(
                    "UNEXPECTED_ABLATION_RECORD",
                    "Ablation directory contains a model outside the phase4 mechanism-control matrix.",
                    source_path=row.get("source_path"),
                    model_key=row.get("model_key"),
                )
            )
    if main_bundle.get("paper_eligible") is not True:
        blockers.append(_blocker("MAIN_BUNDLE_NOT_ELIGIBLE", "Paper-facing ablation controls require a paper-eligible Main Results bundle."))
    return {
        "kind": "ablation",
        "passed": not blockers,
        "paper_eligible": not blockers,
        "config_sha256": config.config_sha256 if isinstance(config, FormalConfig) else None,
        "dataset_sha256": _as_dict(config)["dataset"]["sha256"],
        "main_bundle_id": main_bundle.get("bundle_id") if isinstance(main_bundle, dict) else None,
        "main_bundle_path": main_bundle.get("manifest_path") if isinstance(main_bundle, dict) else None,
        "record_count": len(records),
        "required_model_keys": ["baseline", *required_keys, "full"],
        "matrix": matrix,
        "selected_records": selected,
        "record_issues": record_issues,
        "blockers": blockers,
        "claim_boundary": "matched mechanism-control evidence on complete simulated HGV trajectories",
    }


def _manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source",
        "source_path",
        "source_sha256",
        "run_id",
        "model",
        "model_key",
        "seed",
        "model_config_sha256",
        "formal_config_sha256",
        "formal_config_binding",
        "formal_config_attestation",
        "checkpoint",
        "checkpoint_sha256",
        "run_signature",
        "phase",
        "timestamp",
        "dataset_protocol",
        "dataset_sha256",
        "reused_for_ablation",
    )
    return {key: record[key] for key in keys if key in record}


def build_evidence_bundle(
    records: list[dict[str, Any]],
    audit: dict[str, Any],
    output_path: str | Path,
    *,
    config: FormalConfig | dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write one manifest atomically; incomplete bundles remain paper-ineligible."""
    target = Path(output_path).expanduser()
    if target.suffix.lower() != ".json":
        filename = "ablation_run_set_manifest.json" if audit.get("kind") == "ablation" else "main_run_set_manifest.json"
        target = target / filename
    target = target.resolve()
    if target.exists() and not force:
        raise FileExistsError(f"Evidence manifest already exists: {target}")
    config_hash = audit.get("config_sha256")
    if not config_hash and isinstance(config, FormalConfig):
        config_hash = config.config_sha256
    config_data = _as_dict(config) if config is not None else {}
    dataset_hash = str(config_data.get("dataset", {}).get("sha256", ""))
    bundle_kind = str(audit.get("kind", "unknown"))
    if bundle_kind not in {"main", "ablation"}:
        raise ValueError(f"Unsupported evidence bundle kind: {bundle_kind!r}")
    source_records = [_manifest_record(item) for item in audit.get("selected_records", [])]
    if not source_records:
        raise ValueError("Evidence bundle cannot be created with empty source_records.")
    if not config_hash or not dataset_hash:
        raise ValueError("Evidence bundle requires signed config_sha256 and dataset_sha256.")
    if bundle_kind == "ablation" and audit.get("paper_eligible") is True and not audit.get("main_bundle_id"):
        raise ValueError("Eligible ablation bundle must identify its Main Results bundle.")
    source_hashes = sorted(str(item.get("source_sha256", "")) for item in source_records)
    if any(not item for item in source_hashes):
        raise ValueError("Evidence bundle source_records must contain source hashes.")
    bundle_id = sha256_bytes(canonical_json({"config_sha256": config_hash, "source_record_sha256": source_hashes, "kind": bundle_kind}).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "bundle_kind": bundle_kind,
        "paper_eligible": bool(audit.get("paper_eligible")),
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
        "dataset_protocol": str(config_data.get("dataset", {}).get("protocol", "")),
        "test_trajectory_count": int(config_data.get("split", {}).get("complete_trajectory_counts", {}).get("test", 0)),
        "config_path": str(config.path) if isinstance(config, FormalConfig) else None,
        "main_bundle_id": audit.get("main_bundle_id"),
        "main_bundle_path": audit.get("main_bundle_path"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count_discovered": int(audit.get("record_count", len(records))),
        "source_records": source_records,
        "source_record_sha256": source_hashes,
        "completeness": {
            "required_model_keys": audit.get("required_model_keys", []),
            "matrix": audit.get("matrix", {}),
            "record_issues": audit.get("record_issues", {}),
        },
        "blockers": audit.get("blockers", []),
        "claim_boundary": audit.get("claim_boundary"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, target)
    return manifest | {"manifest_path": str(target)}


def load_evidence_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Evidence bundle is missing: {bundle_path}")
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evidence bundle is not valid JSON: {bundle_path}") from exc
    if not isinstance(bundle, dict) or not bundle.get("bundle_id"):
        raise ValueError(f"Invalid evidence bundle: {bundle_path}")
    source_records = bundle.get("source_records", [])
    if not isinstance(source_records, list):
        raise ValueError(f"Evidence bundle source_records must be a list: {bundle_path}")
    if not source_records:
        raise ValueError(f"Evidence bundle source_records must be non-empty: {bundle_path}")
    if bundle.get("bundle_kind") not in {"main", "ablation"}:
        raise ValueError(f"Evidence bundle has an invalid bundle_kind: {bundle_path}")
    if not str(bundle.get("config_sha256", "")) or not str(bundle.get("dataset_sha256", "")):
        raise ValueError(f"Evidence bundle lacks signed config/dataset hashes: {bundle_path}")
    if any(not isinstance(item, dict) for item in source_records):
        raise ValueError(f"Evidence bundle source_records must contain objects: {bundle_path}")
    recorded_hashes = sorted(str(item.get("source_sha256", "")) for item in source_records)
    if any(not item for item in recorded_hashes):
        raise ValueError(f"Evidence bundle source_records contain an empty source hash: {bundle_path}")
    manifest_hashes = sorted(str(item) for item in bundle.get("source_record_sha256", []))
    if recorded_hashes != manifest_hashes:
        raise ValueError(f"Evidence bundle source hash index is inconsistent: {bundle_path}")
    expected_bundle_id = sha256_bytes(
        canonical_json(
            {
                "config_sha256": bundle.get("config_sha256"),
                "source_record_sha256": manifest_hashes,
                "kind": bundle.get("bundle_kind"),
            }
        ).encode("utf-8")
    )
    if str(bundle.get("bundle_id")) != expected_bundle_id:
        raise ValueError(f"Evidence bundle ID does not match its signed contents: {bundle_path}")
    for entry in source_records:
        source_path = Path(str(entry.get("source_path", ""))).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Bundle source record is missing: {source_path}")
        observed = sha256_file(source_path)
        expected = str(entry.get("source_sha256", ""))
        if not expected or observed.lower() != expected.lower():
            raise ValueError(f"Bundle source record hash mismatch: {source_path}")
    bundle["manifest_path"] = str(bundle_path)
    return bundle


def _bundle_record_key(record: dict[str, Any]) -> tuple[str, str, str, int, str, str]:
    return (
        str(record.get("source_path", "")),
        str(record.get("source_sha256", "")),
        str(record.get("model_key", "")),
        int(record.get("seed", -1)),
        str(record.get("checkpoint", "")),
        str(record.get("checkpoint_sha256", "")),
    )


def validate_evidence_bundle(
    bundle_path: str | Path,
    config: FormalConfig | dict[str, Any],
    *,
    expected_kind: str,
    main_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate a signed bundle against the current config and source records."""
    bundle = load_evidence_bundle(bundle_path)
    blockers: list[dict[str, Any]] = []
    expected_config_hash = config.config_sha256 if isinstance(config, FormalConfig) else ""
    expected_dataset_hash = str(_as_dict(config)["dataset"]["sha256"])
    if bundle.get("bundle_kind") != expected_kind:
        blockers.append(
            _blocker(
                "BUNDLE_KIND_MISMATCH",
                f"Expected a {expected_kind} evidence bundle.",
                expected=expected_kind,
                observed=bundle.get("bundle_kind"),
            )
        )
    if str(bundle.get("config_sha256", "")) != expected_config_hash:
        blockers.append(
            _blocker(
                "BUNDLE_CONFIG_HASH_MISMATCH",
                "Evidence bundle does not match the current formal-v3 configuration.",
                expected=expected_config_hash,
                observed=bundle.get("config_sha256"),
            )
        )
    if str(bundle.get("dataset_sha256", "")) != expected_dataset_hash:
        blockers.append(
            _blocker(
                "BUNDLE_DATASET_HASH_MISMATCH",
                "Evidence bundle does not match the frozen formal-v3 dataset.",
                expected=expected_dataset_hash,
                observed=bundle.get("dataset_sha256"),
            )
        )

    entries = list(bundle.get("source_records", []))
    normalized: list[dict[str, Any]] = []
    entry_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in entries:
        try:
            record = normalize_run_record(entry["source_path"])
            normalized.append(record)
            entry_pairs.append((entry, record))
            if str(entry.get("source_sha256", "")).lower() != str(record.get("source_sha256", "")).lower():
                blockers.append(
                    _blocker(
                        "BUNDLE_SOURCE_METADATA_MISMATCH",
                        "Bundle source metadata does not match the immutable source record.",
                        source_path=record.get("source_path"),
                    )
                )
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(
                _blocker(
                    "INVALID_BUNDLE_SOURCE_RECORD",
                    "Bundle source record cannot be normalized.",
                    error=str(exc),
                )
            )

    if expected_kind == "main":
        audit = validate_main_completeness(normalized, config)
    else:
        if main_bundle is None:
            blockers.append(
                _blocker(
                    "MISSING_MAIN_BUNDLE_REFERENCE",
                    "Ablation paper evidence must be checked against a Main Results bundle.",
                )
            )
            audit = validate_ablation_completeness([], {}, config)
        else:
            if main_bundle.get("bundle_kind") != "main":
                blockers.append(
                    _blocker(
                        "MAIN_BUNDLE_KIND_MISMATCH",
                        "Ablation bundle does not reference a Main Results bundle.",
                        observed=main_bundle.get("bundle_kind"),
                    )
                )
            if str(bundle.get("main_bundle_id", "")) != str(main_bundle.get("bundle_id", "")):
                blockers.append(
                    _blocker(
                        "MAIN_ABLATION_BUNDLE_MISMATCH",
                        "Ablation bundle does not reference the selected Main Results bundle.",
                        expected=main_bundle.get("bundle_id"),
                        observed=bundle.get("main_bundle_id"),
                    )
                )
            main_entries = list(main_bundle.get("source_records", []))
            expected_controls = {
                (str(entry.get("model_key")), int(entry.get("seed", -1))): entry
                for entry in main_entries
                if str(entry.get("model_key")) in {"baseline", "full"}
            }
            reused_pairs = [pair for pair in entry_pairs if pair[0].get("reused_for_ablation") is True]
            observed_controls = {
                (str(entry.get("model_key")), int(entry.get("seed", -1))): entry
                for entry, _ in reused_pairs
            }
            required_controls = {(key, seed) for key in ("baseline", "full") for seed in (42, 123, 456)}
            if set(observed_controls) != required_controls:
                blockers.append(
                    _blocker(
                        "MAIN_ABLATION_CONTROL_MISMATCH",
                        "Ablation bundle must reuse exactly the baseline/full controls from Main Results.",
                        expected=sorted(required_controls),
                        observed=sorted(observed_controls),
                    )
                )
            for key in sorted(required_controls):
                expected = expected_controls.get(key)
                observed = observed_controls.get(key)
                if expected is None or observed is None:
                    continue
                if _bundle_record_key(expected) != _bundle_record_key(observed):
                    blockers.append(
                        _blocker(
                            "MAIN_ABLATION_CONTROL_MISMATCH",
                            "Ablation reused control does not match the selected Main source record/checkpoint.",
                            model_key=key[0],
                            seed=key[1],
                        )
                    )
            independent = [record for entry, record in entry_pairs if entry.get("reused_for_ablation") is not True]
            main_basis = dict(main_bundle)
            main_basis["selected_records"] = main_entries
            audit = validate_ablation_completeness(independent, main_basis, config)

    expected_selected = {
        _bundle_record_key(item)
        for item in audit.get("selected_records", [])
        if item.get("source_path")
    }
    observed_selected = {_bundle_record_key(item) for item in entries}
    if expected_selected != observed_selected:
        blockers.append(
            _blocker(
                "BUNDLE_SOURCE_SELECTION_MISMATCH",
                "Bundle source_records do not equal the records selected by the evidence audit.",
                expected_count=len(expected_selected),
                observed_count=len(observed_selected),
            )
        )
    if bundle.get("paper_eligible") is True and (blockers or not audit.get("paper_eligible")):
        blockers.append(
            _blocker(
                "FORGED_PAPER_ELIGIBILITY",
                "paper_eligible=true is not supported by the current bundle and source-record audit.",
            )
        )
    if bundle.get("paper_eligible") is not True:
        blockers.append(
            _blocker(
                "BUNDLE_NOT_PAPER_ELIGIBLE",
                "Evidence bundle is not marked paper-eligible.",
            )
        )
    return {
        "bundle": bundle,
        "audit": audit,
        "passed": not blockers,
        "paper_eligible": not blockers and bundle.get("paper_eligible") is True,
        "blockers": blockers,
    }


def resolve_bundle_checkpoint(bundle: dict[str, Any], model_key: str, seed: int) -> Path:
    candidates = [
        item
        for item in bundle.get("source_records", [])
        if str(item.get("model_key", "")) == str(model_key) and int(item.get("seed", -1)) == int(seed)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Evidence bundle has {len(candidates)} checkpoint candidates for {model_key}:{seed}.")
    candidate = candidates[0]
    recorded_checkpoint = Path(str(candidate.get("checkpoint", ""))).expanduser().resolve()
    checkpoint, _ = _relocate_project_reference(recorded_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Bundle checkpoint is missing: {checkpoint}")
    expected = str(candidate.get("checkpoint_sha256", ""))
    if not expected:
        raise RuntimeError(f"Bundle checkpoint hash is missing: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"Bundle checkpoint hash mismatch: {checkpoint}")
    return checkpoint


def audit_dataset(config: FormalConfig | dict[str, Any]) -> dict[str, Any]:
    data = _as_dict(config)
    if isinstance(config, FormalConfig):
        path = config.resolve(data["dataset"]["relative_path"])
        config_hash = config.config_sha256
    else:
        path = (Path.cwd() / data["dataset"]["relative_path"]).resolve()
        config_hash = None
    report: dict[str, Any] = {
        "path": str(path),
        "protocol": data["dataset"]["protocol"],
        "expected_sha256": data["dataset"]["sha256"],
        "config_sha256": config_hash,
        "exists": path.is_file(),
        "passed": False,
    }
    if path.is_file():
        report["observed_sha256"] = sha256_file(path)
        report["passed"] = report["observed_sha256"].lower() == report["expected_sha256"].lower()
    return report


def bundle_as_registry(bundle_path: str | Path) -> tuple[dict[str, Any], str]:
    """Expose eligible Main bundle records through legacy analysis adapters."""
    bundle = load_evidence_bundle(bundle_path)
    if bundle.get("bundle_kind") != "main" or bundle.get("paper_eligible") is not True:
        raise RuntimeError("An eligible formal-v3 Main Results bundle is required.")
    signature = str(bundle["bundle_id"])
    model_types = {
        "baseline": "transformer",
        "full": "plgaformer",
        "pit": "pit",
        "dlinear": "dlinear",
        "patchtst": "patchtst",
        "itransformer": "itransformer",
        "af_ciln": "af_ciln",
        "kinematic": "kinematic",
        "rotating_3dof": "rotating_3dof",
    }
    runs: dict[str, dict[str, Any]] = {}
    for entry in bundle.get("source_records", []):
        model_key = str(entry.get("model_key", ""))
        model_type = model_types.get(model_key)
        if model_type is None:
            continue
        run = {
            "run_signature": signature,
            "seed": int(entry.get("seed", -1)),
            "model_type": model_type,
            "model_key": model_key,
            "model_name": DISPLAY_NAMES.get(model_key, model_key),
            "checkpoint_path": entry.get("checkpoint"),
            "checkpoint_sha256": entry.get("checkpoint_sha256"),
            "source_json": entry.get("source_path"),
            "source_json_sha256": entry.get("source_sha256"),
            "completed_at": entry.get("timestamp"),
        }
        runs[f"{model_type}|seed={run['seed']}|source={entry.get('source_path')}"] = run
    return {"active_config": {"run_signature": signature}, "runs": runs, "bundle_id": signature}, signature


__all__ = [
    "ANALYTICAL_MODEL_KEYS",
    "DISPLAY_NAMES",
    "FINAL_MODEL_FLAGS",
    "FormalConfig",
    "audit_dataset",
    "build_evidence_bundle",
    "bundle_as_registry",
    "canonical_json",
    "discover_run_records",
    "load_evidence_bundle",
    "load_formal_config",
    "normalize_run_record",
    "resolve_bundle_checkpoint",
    "resolve_config_path",
    "final_model_configuration_blockers",
    "sha256_file",
    "validate_ablation_completeness",
    "validate_evidence_bundle",
    "validate_main_completeness",
    "validate_protocol_core",
]
