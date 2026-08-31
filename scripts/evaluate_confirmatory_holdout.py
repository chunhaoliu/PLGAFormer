#!/usr/bin/env python3
"""Evaluate frozen Main Results checkpoints on the locked confirmatory holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.overall_prediction import main_results as exp1
from utils.final_plgaformer import apply_paper_inference_policy, final_plgaformer_kwargs
from utils.formal_evidence import load_formal_config, validate_evidence_bundle
from utils.seq2seq_protocol import align_source_position_scale


DATASET = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_confirmatory_holdout.npz"
)
DATASET_MANIFEST = DATASET.with_suffix(".manifest.json")
PROTOCOL_DOCUMENT = PROJECT_ROOT / "docs" / "CONFIRMATORY_HOLDOUT_PROTOCOL.md"
FORMAL_CONFIG = PROJECT_ROOT / "configs" / "formal_v3.json"
MAIN_BUNDLE = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "main_run_set_manifest.json"
)
RESULT_DIR = PROJECT_ROOT / "experiments" / "confirmatory_holdout" / "results"
PUBLIC_EVIDENCE = PROJECT_ROOT / "PublicRelease" / "evidence" / "confirmatory_holdout.json"
MODEL_KEYS = ("full", "baseline", "dlinear", "patchtst", "itransformer")
SEEDS = (42, 123, 456)
HORIZONS = (32, 64, 128, 256)
EXPECTED_DATASET_SHA256 = "691371f35940b252d72261071db907b4e81a3ad9050a305cb7e7b2ff7278dff3"
EXPECTED_PROTOCOL_SHA256 = "b950b81756be3f10340c9a1b182117b29712ff30dabe7fad99ccd8dda03e1af8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def validate_locked_dataset() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file():
        raise FileNotFoundError("Locked confirmatory dataset or manifest is missing.")
    manifest = _load_json(DATASET_MANIFEST)
    dataset_hash = sha256_file(DATASET)
    protocol_hash = sha256_file(PROTOCOL_DOCUMENT)
    if dataset_hash != EXPECTED_DATASET_SHA256 or manifest.get("sha256") != dataset_hash:
        raise RuntimeError("Confirmatory dataset SHA-256 does not match the frozen contract.")
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Confirmatory protocol document changed after data generation.")
    if manifest.get("protocol_document_sha256") != protocol_hash:
        raise RuntimeError("Dataset manifest is not bound to the frozen protocol document.")

    with np.load(DATASET, allow_pickle=False) as archive:
        required = (
            "X_confirmatory",
            "y_confirmatory",
            "trajectory_ids_confirmatory",
            "maneuver_labels_confirmatory",
            "trajectory_ids",
            "joint_strata",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise RuntimeError(f"Confirmatory dataset keys are missing: {missing}")
        data = {key: np.asarray(archive[key]) for key in required}
    if data["X_confirmatory"].shape != (35280, 256, 6):
        raise RuntimeError(f"Unexpected confirmatory X shape: {data['X_confirmatory'].shape}")
    if data["y_confirmatory"].shape != (35280, 256, 3):
        raise RuntimeError(f"Unexpected confirmatory y shape: {data['y_confirmatory'].shape}")
    unique_ids = np.unique(data["trajectory_ids"])
    if not np.array_equal(unique_ids, np.arange(1800, 2160, dtype=np.int64)):
        raise RuntimeError("Confirmatory trajectory IDs do not match 1800--2159.")
    counts = {
        str(label): int(np.count_nonzero(data["joint_strata"] == label))
        for label in np.unique(data["joint_strata"])
    }
    if len(counts) != 6 or set(counts.values()) != {60}:
        raise RuntimeError(f"Confirmatory joint-stratum balance is invalid: {counts}")
    return manifest, data


def validate_main_bundle() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    config = load_formal_config(FORMAL_CONFIG)
    verification = validate_evidence_bundle(MAIN_BUNDLE, config, expected_kind="main")
    if not verification["passed"]:
        raise RuntimeError(f"Main Results bundle failed verification: {verification['blockers']}")
    bundle = verification["bundle"]
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for record in bundle.get("source_records", []):
        key = (str(record.get("model_key")), int(record.get("seed", -1)))
        if key[0] not in MODEL_KEYS or key[1] not in SEEDS:
            continue
        if key in records:
            raise RuntimeError(f"Duplicate Main Results record: {key}")
        records[key] = record
    expected = {(model_key, seed) for model_key in MODEL_KEYS for seed in SEEDS}
    if set(records) != expected:
        raise RuntimeError(f"Incomplete learning-checkpoint matrix: missing={sorted(expected-set(records))}")
    return bundle, records


def _source_payload(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(record["source_path"])).resolve()
    if sha256_file(path) != str(record["source_sha256"]):
        raise RuntimeError(f"Source-record hash mismatch: {path}")
    return _load_json(path)


def _load_scalers(payload: dict[str, Any]) -> tuple[Any, Any, dict[str, str]]:
    provenance = payload.get("scaler_provenance", {})
    paths = {
        "input": Path(str(provenance.get("input_scaler_path", ""))).resolve(),
        "output": Path(str(provenance.get("output_scaler_path", ""))).resolve(),
        "signature": Path(str(provenance.get("scaler_signature_path", ""))).resolve(),
    }
    expected = {
        "input": str(provenance.get("input_scaler_sha256", "")),
        "output": str(provenance.get("output_scaler_sha256", "")),
        "signature": str(provenance.get("scaler_signature_sha256", "")),
    }
    actual = {key: sha256_file(path) for key, path in paths.items()}
    if actual != expected:
        raise RuntimeError(f"Scaler provenance mismatch: expected={expected}, actual={actual}")
    return joblib.load(paths["input"]), joblib.load(paths["output"]), actual


def prepare_loader(
    data: dict[str, np.ndarray], input_scaler: Any, output_scaler: Any, batch_size: int
) -> DataLoader:
    x_raw = data["X_confirmatory"]
    y_raw = data["y_confirmatory"]
    x_scaled = input_scaler.transform(x_raw.reshape(-1, 6)).reshape(x_raw.shape)
    x_scaled = align_source_position_scale(
        x_scaled,
        input_mean=input_scaler.mean_,
        input_scale=input_scaler.scale_,
        output_mean=output_scaler.mean_,
        output_scale=output_scaler.scale_,
        output_dim=3,
    ).astype(np.float32, copy=False)
    y_scaled = output_scaler.transform(y_raw.reshape(-1, 3)).reshape(y_raw.shape)
    dataset = TensorDataset(
        torch.from_numpy(x_scaled),
        torch.from_numpy(y_scaled.astype(np.float32, copy=False)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def reconstruct_model(
    payload: dict[str, Any], input_scaler: Any, output_scaler: Any, device: torch.device
) -> torch.nn.Module:
    model_type = str(payload["model_config"]["model_type"]).lower()
    kwargs = exp1._model_reconstruction_kwargs(model_type, input_scaler, output_scaler) or {}
    if model_type == "plgaformer":
        kwargs.update(final_plgaformer_kwargs())
    external = payload.get("external_source_provenance")
    if model_type in {"transformer", "dlinear", "patchtst", "itransformer"}:
        if not isinstance(external, dict) or not external.get("root"):
            raise RuntimeError(f"Missing public-source provenance for {model_type}.")
        os.environ["HGV_TSLIB_ROOT"] = str(external["root"])
        kwargs.update({"source": "tslib", "tslib_root": str(external["root"])})
    return exp1.create_model(model_type, input_dim=6, device=device, plgaformer_kwargs=kwargs)


def compact_metrics(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for horizon, values in results.items():
        compact[str(horizon)] = {
            key: values[key]
            for key in (
                "ade",
                "fde",
                "rmse_cart_m",
                "trajectory_window_ade",
                "trajectory_window_fde",
                "trajectory_count",
                "trajectory_ade_ci",
                "trajectory_fde_ci",
                "trajectory_metrics",
            )
            if key in values
        }
    return compact


def maneuver_summary(
    compact: dict[str, Any], window_trajectory_ids: np.ndarray, window_labels: np.ndarray
) -> dict[str, Any]:
    label_sets: dict[int, set[str]] = {}
    for trajectory_id, label in zip(window_trajectory_ids, window_labels):
        label_sets.setdefault(int(trajectory_id), set()).add(str(label))
    inconsistent = {key: values for key, values in label_sets.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"Inconsistent maneuver labels by trajectory: {inconsistent}")
    label_by_id = {key: next(iter(values)) for key, values in label_sets.items()}
    output: dict[str, Any] = {}
    for horizon, values in compact.items():
        buckets: dict[str, dict[str, list[float]]] = {}
        for trajectory_id, metrics in values["trajectory_metrics"].items():
            label = label_by_id[int(trajectory_id)]
            bucket = buckets.setdefault(label, {"ade": [], "fde": []})
            bucket["ade"].append(float(metrics["ade"]))
            bucket["fde"].append(float(metrics["fde"]))
        output[horizon] = {
            label: {
                "trajectory_count": len(metrics["ade"]),
                "ade": float(np.mean(metrics["ade"])),
                "fde": float(np.mean(metrics["fde"])),
            }
            for label, metrics in buckets.items()
        }
    return output


def aggregate_records(records: list[dict[str, Any]], bundle_id: str) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        model_records = [record for record in records if record["model_key"] == model_key]
        aggregate[model_key] = {}
        for horizon in HORIZONS:
            aggregate[model_key][str(horizon)] = {}
            for metric in ("ade", "fde", "rmse_cart_m"):
                values = [float(record["metrics"][str(horizon)][metric]) for record in model_records]
                aggregate[model_key][str(horizon)][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    identity = {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "protocol_document_sha256": EXPECTED_PROTOCOL_SHA256,
        "main_bundle_id": bundle_id,
        "model_keys": list(MODEL_KEYS),
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "aggregate": aggregate,
    }
    identity["result_bundle_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return identity


def audit_existing_results(
    main_bundle: dict[str, Any], source_records: dict[tuple[str, int], dict[str, Any]]
) -> dict[str, Any]:
    bundle_path = RESULT_DIR / "confirmatory_results.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Confirmatory result bundle is missing: {bundle_path}")
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for model_key in MODEL_KEYS:
        for seed in SEEDS:
            path = RESULT_DIR / f"{model_key}_seed{seed}.json"
            record = _load_json(path)
            identity = (str(record.get("model_key")), int(record.get("seed", -1)))
            if identity != (model_key, seed) or identity in seen:
                raise RuntimeError(f"Invalid or duplicate confirmatory record identity: {path}")
            seen.add(identity)
            source = source_records[identity]
            expected_fields = {
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "protocol_document_sha256": EXPECTED_PROTOCOL_SHA256,
                "main_bundle_id": main_bundle["bundle_id"],
                "formal_config_sha256": main_bundle["config_sha256"],
                "source_record_sha256": source["source_sha256"],
                "checkpoint_sha256": source["checkpoint_sha256"],
            }
            mismatches = {
                key: (record.get(key), value)
                for key, value in expected_fields.items()
                if record.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Confirmatory provenance mismatch in {path}: {mismatches}")
            if record.get("training_performed") is not False or record.get(
                "test_evaluation_performed"
            ) is not True:
                raise RuntimeError(f"Invalid evaluation/training flags in {path}")
            for horizon in HORIZONS:
                metrics = record.get("metrics", {}).get(str(horizon), {})
                for metric in ("ade", "fde", "rmse_cart_m"):
                    value = metrics.get(metric)
                    if not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise RuntimeError(
                            f"Missing or non-finite {metric}@{horizon} in {path}"
                        )
                if int(metrics.get("trajectory_count", -1)) != 360:
                    raise RuntimeError(f"Trajectory count mismatch at {horizon} in {path}")
            records.append(record)

    expected_aggregate = aggregate_records(records, str(main_bundle["bundle_id"]))
    saved_bundle = _load_json(bundle_path)
    for key in (
        "dataset_sha256",
        "protocol_document_sha256",
        "main_bundle_id",
        "result_bundle_id",
    ):
        if saved_bundle.get(key) != expected_aggregate.get(key):
            raise RuntimeError(f"Aggregate identity mismatch for {key}")
    if saved_bundle.get("aggregate") != expected_aggregate.get("aggregate"):
        raise RuntimeError("Saved confirmatory aggregate does not reproduce from records.")
    if int(saved_bundle.get("record_count", -1)) != 15 or saved_bundle.get(
        "paper_eligible"
    ) is not True:
        raise RuntimeError("Confirmatory result bundle is incomplete or ineligible.")

    ranking: dict[str, Any] = {}
    for horizon in HORIZONS:
        ranking[str(horizon)] = {}
        for metric in ("ade", "fde", "rmse_cart_m"):
            ordered = sorted(
                (
                    (
                        model_key,
                        expected_aggregate["aggregate"][model_key][str(horizon)][metric]["mean"],
                    )
                    for model_key in MODEL_KEYS
                ),
                key=lambda item: item[1],
            )
            best_key, best_value = ordered[0]
            second_key, second_value = ordered[1]
            ranking[str(horizon)][metric] = {
                "order": [key for key, _ in ordered],
                "best": best_key,
                "best_value_m": best_value,
                "second": second_key,
                "second_value_m": second_value,
                "reduction_vs_second_percent": 100.0
                * (second_value - best_value)
                / second_value,
            }
    return {
        "passed": True,
        "record_count": len(records),
        "result_bundle_id": expected_aggregate["result_bundle_id"],
        "ranking": ranking,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    manifest, data = validate_locked_dataset()
    bundle, source_records = validate_main_bundle()
    if args.audit_only:
        print(json.dumps(audit_existing_results(bundle, source_records), indent=2))
        return 0
    representative_payload = _source_payload(source_records[("full", 42)])
    input_scaler, output_scaler, scaler_hashes = _load_scalers(representative_payload)
    if args.preflight_only:
        preflight = []
        for model_key in MODEL_KEYS:
            for seed in SEEDS:
                record = source_records[(model_key, seed)]
                payload = _source_payload(record)
                checkpoint = Path(str(record["checkpoint"])).resolve()
                checkpoint_hash = sha256_file(checkpoint)
                if checkpoint_hash != str(record["checkpoint_sha256"]):
                    raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
                model = reconstruct_model(payload, input_scaler, output_scaler, device)
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(state, strict=True)
                if model_key == "full":
                    apply_paper_inference_policy(model)
                preflight.append(
                    {"model_key": model_key, "seed": seed, "checkpoint_sha256": checkpoint_hash}
                )
                del model, state
        print(
            json.dumps(
                {
                    "passed": True,
                    "dataset_sha256": manifest["sha256"],
                    "main_bundle_id": bundle["bundle_id"],
                    "scaler_sha256": scaler_hashes,
                    "checkpoint_count": len(preflight),
                    "checkpoints": preflight,
                },
                indent=2,
            )
        )
        return 0

    loader = prepare_loader(data, input_scaler, output_scaler, args.batch_size)
    output_mean = torch.from_numpy(output_scaler.mean_.astype(np.float32)).to(device)
    output_scale = torch.from_numpy(output_scaler.scale_.astype(np.float32)).to(device)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for seed in SEEDS:
            output_path = RESULT_DIR / f"{model_key}_seed{seed}.json"
            if output_path.exists():
                if not args.skip_existing:
                    raise FileExistsError(f"Refusing to replace result: {output_path}")
                existing = _load_json(output_path)
                if (
                    existing.get("dataset_sha256") != EXPECTED_DATASET_SHA256
                    or existing.get("main_bundle_id") != bundle["bundle_id"]
                    or existing.get("model_key") != model_key
                    or int(existing.get("seed", -1)) != seed
                ):
                    raise RuntimeError(f"Existing result failed identity checks: {output_path}")
                completed.append(existing)
                continue

            record = source_records[(model_key, seed)]
            payload = _source_payload(record)
            checkpoint = Path(str(record["checkpoint"])).resolve()
            checkpoint_hash = sha256_file(checkpoint)
            if checkpoint_hash != str(record["checkpoint_sha256"]):
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
            model = reconstruct_model(payload, input_scaler, output_scaler, device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)
            if model_key == "full":
                inference_policy = apply_paper_inference_policy(model)
            else:
                inference_policy = None
            model.to(device)
            results = exp1.run_evaluation(
                model,
                loader,
                output_mean,
                output_scale,
                device,
                list(HORIZONS),
                trajectory_ids=data["trajectory_ids_confirmatory"],
            )
            compact = compact_metrics(results)
            run_record = {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "evidence_tier": "confirmatory_final",
                "test_evaluation_performed": True,
                "training_performed": False,
                "model_key": model_key,
                "model": payload.get("model"),
                "seed": seed,
                "dataset_protocol": manifest["dataset_protocol"],
                "dataset_sha256": EXPECTED_DATASET_SHA256,
                "protocol_document_sha256": EXPECTED_PROTOCOL_SHA256,
                "main_bundle_id": bundle["bundle_id"],
                "formal_config_sha256": bundle["config_sha256"],
                "source_record": record["source_path"],
                "source_record_sha256": record["source_sha256"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "scaler_sha256": scaler_hashes,
                "inference_policy": inference_policy,
                "device": str(device),
                "metrics": compact,
                "maneuver_metrics": maneuver_summary(
                    compact,
                    data["trajectory_ids_confirmatory"],
                    data["maneuver_labels_confirmatory"],
                ),
            }
            output_path.write_text(
                json.dumps(run_record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            completed.append(run_record)
            print(f"[confirmatory] completed {model_key} seed={seed}")
            del model, state
            if device.type == "cuda":
                torch.cuda.empty_cache()

    bundle_payload = aggregate_records(completed, str(bundle["bundle_id"]))
    bundle_payload.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(completed),
            "paper_eligible": len(completed) == len(MODEL_KEYS) * len(SEEDS),
            "claim_boundary": "same-simulator-family simulation-only confirmation",
        }
    )
    bundle_path = RESULT_DIR / "confirmatory_results.json"
    if bundle_path.exists() and not args.skip_existing:
        raise FileExistsError(f"Refusing to replace aggregate bundle: {bundle_path}")
    bundle_path.write_text(
        json.dumps(bundle_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    PUBLIC_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_EVIDENCE.write_text(
        json.dumps(bundle_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"bundle": str(bundle_path), **bundle_payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
