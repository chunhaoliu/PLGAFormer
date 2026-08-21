"""Resolve the formal-v3 Main Results PLGAFormer checkpoint authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from utils.formal_evidence import (
    final_model_configuration_blockers,
    load_formal_config,
    resolve_bundle_checkpoint,
    sha256_file,
    validate_evidence_bundle,
)
from utils.mainline_contract import ACTIVE_MODEL_KEY, ACTIVE_PLGAFORMER_FLAGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_MAIN_RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
)
MAIN_BUNDLE_PATH = FORMAL_MAIN_RESULTS_DIR / "main_run_set_manifest.json"
LEGACY_FORMAL_ABLATION_DIR = (
    PROJECT_ROOT / "experiments" / "exp2_ablation" / "results" / "formal_v2"
)
FINAL_MODEL_KEY = ACTIVE_MODEL_KEY
FINAL_ARCHITECTURE = (
    "rotating-Earth 3-DOF prior with adaptive bounded fusion and no channel residual"
)
FINAL_SEEDS = (42, 123, 456)


def final_plgaformer_kwargs() -> dict[str, Any]:
    """Return the resolved constructor flags for the paper-facing architecture."""
    return dict(ACTIVE_PLGAFORMER_FLAGS)


def _load_source_payload(record: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(record.get("source_path", ""))).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Bundle source record is missing: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bundle source record is not an object: {source_path}")
    return payload


def _assert_final_configuration(payload: dict[str, Any]) -> None:
    if str(payload.get("model_key", "")).lower() != FINAL_MODEL_KEY:
        raise RuntimeError(f"Unexpected final-model key: {payload.get('model_key')}")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("Formal-v3 full record lacks a resolved model configuration.")
    issues = final_model_configuration_blockers(
        {"source_path": payload.get("run_id", "<formal-v3-full-record>"), "model_config": config}
    )
    if issues:
        raise RuntimeError(
            "Formal-v3 full record has invalid final-model configuration: "
            + "; ".join(str(item.get("message")) for item in issues)
        )


def _resolve_legacy_final_plgaformer(seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve the historical formal-v2 candidate for diagnostics only."""
    candidates = sorted(
        LEGACY_FORMAL_ABLATION_DIR.glob(
            f"formal_phase1_structural_seed{int(seed)}_prior_only_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No legacy PLGAFormer ablation record for seed={seed}.")
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    expected = {
        "subset_ratio": 1.0,
        "epochs": 10,
        "batch_size": 128,
        "prediction_length": 256,
        "prediction_horizons": "32,64,128,256",
        "train_supervision_protocol": "source_context_pred_window",
        "eval_protocol": "source_context_decoder",
        "label_len": 128,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Legacy final PLGAFormer protocol mismatch: {mismatches}.")
    checkpoint = Path(payload.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Legacy PLGAFormer checkpoint is missing: {checkpoint}")
    audit = {
        "source": "legacy_formal_v2_diagnostic",
        "seed": int(seed),
        "architecture": "historical formal-v2 prior_only candidate",
        "record": str(path.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return checkpoint, payload, audit


def resolve_final_plgaformer(
    seed: int,
    *,
    bundle_path: str | Path | None = None,
    allow_legacy: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Resolve an exact formal-v3 ``model_key=full`` checkpoint.

    Legacy formal-v2 resolution is opt-in and marked diagnostic in its audit.
    Paper-facing callers must use the default formal-v3 route.
    """
    selected_bundle_path = Path(bundle_path).expanduser().resolve() if bundle_path else MAIN_BUNDLE_PATH
    if not selected_bundle_path.is_file():
        if allow_legacy:
            return _resolve_legacy_final_plgaformer(seed)
        raise FileNotFoundError(
            "Formal-v3 Main Results bundle is missing. Run `python run.py formal aggregate` "
            f"after the complete matrix is available: {selected_bundle_path}"
        )
    config = load_formal_config(PROJECT_ROOT / "configs" / "formal_v3.json")
    verification = validate_evidence_bundle(
        selected_bundle_path,
        config,
        expected_kind="main",
    )
    if not verification["passed"]:
        codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
        raise RuntimeError(
            "Formal-v3 Main Results bundle failed verification; blockers="
            + ",".join(codes)
        )
    bundle = verification["bundle"]
    checkpoint = resolve_bundle_checkpoint(bundle, FINAL_MODEL_KEY, int(seed))
    candidates = [
        item
        for item in bundle.get("source_records", [])
        if item.get("model_key") == FINAL_MODEL_KEY and int(item.get("seed", -1)) == int(seed)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Bundle does not contain exactly one full record for seed={seed}.")
    payload = _load_source_payload(candidates[0])
    _assert_final_configuration(payload)
    audit = {
        "source": "formal_v3_main_results_bundle",
        "bundle_id": bundle["bundle_id"],
        "bundle": str(selected_bundle_path),
        "seed": int(seed),
        "model_key": FINAL_MODEL_KEY,
        "architecture": FINAL_ARCHITECTURE,
        "record": str(Path(candidates[0]["source_path"]).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": candidates[0].get("model_config_sha256"),
        "constructor_flags": final_plgaformer_kwargs(),
    }
    return checkpoint, payload, audit


def final_evidence_signature(
    base_signature: str,
    *,
    bundle_path: str | Path | None = None,
    allow_legacy: bool = False,
) -> tuple[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for seed in FINAL_SEEDS:
        _, _, audit = resolve_final_plgaformer(
            seed, bundle_path=bundle_path, allow_legacy=allow_legacy
        )
        audits[str(seed)] = audit
    signature_payload = {
        "base_exp1_signature": str(base_signature),
        "source": "formal_v3_main_results_bundle" if not allow_legacy else "explicit_legacy_diagnostic",
        "architecture": FINAL_ARCHITECTURE,
        "bundle_id": next(iter(audits.values())).get("bundle_id"),
        "checkpoints": {
            seed: item["checkpoint_sha256"] for seed, item in audits.items()
        },
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return signature, {"signature_payload": signature_payload, "audits": audits}


__all__ = [
    "FINAL_ARCHITECTURE",
    "FINAL_MODEL_KEY",
    "FINAL_SEEDS",
    "FORMAL_MAIN_RESULTS_DIR",
    "MAIN_BUNDLE_PATH",
    "final_evidence_signature",
    "final_plgaformer_kwargs",
    "resolve_final_plgaformer",
]
