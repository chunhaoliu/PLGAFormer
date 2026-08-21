#!/usr/bin/env python3
"""Export compact, path-free evidence for the public repository snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "PublicRelease" / "evidence"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty evidence table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _discover_complete_bundle() -> Path:
    base = (
        ROOT
        / "experiments"
        / "taes_submission_artifacts"
        / "generated"
        / "formal_v3"
        / "hgv_multiregime_state_v2_1"
    )
    candidates = []
    for manifest_path in base.glob("*/artifact_manifest.json"):
        manifest = _read_json(manifest_path)
        if manifest.get("optional_evidence", {}).keys() >= {"robustness", "efficiency"}:
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete paper bundle, found {len(candidates)}")
    return candidates[0]


def _main_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, horizons in summary["seed_statistics"].items():
        for horizon, metrics in horizons.items():
            for metric, values in metrics.items():
                samples = values["values_m"]
                rows.append(
                    {
                        "model": model,
                        "horizon_s": int(horizon),
                        "metric": metric,
                        "mean_m": values["mean_m"],
                        "sample_std_m": values["std_m"],
                        "seed_count": len(samples),
                    }
                )
    return rows


def _maneuver_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model, maneuvers in summary["maneuver_statistics"].items():
        for maneuver, metrics in maneuvers.items():
            for metric, values in metrics.items():
                samples = values["values_m"]
                rows.append(
                    {
                        "model": model,
                        "maneuver": maneuver,
                        "horizon_s": 256,
                        "metric": metric,
                        "mean_m": values["mean_m"],
                        "sample_std_m": values["std_m"],
                        "seed_count": len(samples),
                    }
                )
    return rows


def _ablation_rows(main_summary: dict[str, Any], config_sha: str, dataset_sha: str) -> list[dict[str, Any]]:
    rows = []
    labels = {
        "spherical_prior": "Spherical prior + adaptive fusion",
        "schedule_only": "Rotating-Earth prior + fixed schedule",
    }
    final_dir = (
        ROOT
        / "experiments"
        / "exp2_ablation"
        / "results"
        / "formal_v3"
        / "hgv_multiregime_state_v2_1"
        / "final"
    )
    for key, label in labels.items():
        records = [_read_json(path) for path in sorted(final_dir.glob(f"*_{key}_*.json"))]
        if len(records) != 3:
            raise RuntimeError(f"Expected three {key} records, found {len(records)}")
        if any(
            record.get("evidence_tier") != "final"
            or record.get("test_evaluation_performed") is not True
            or record.get("formal_config_sha256") != config_sha
            or record.get("dataset_identity", {}).get("dataset_sha256") != dataset_sha
            for record in records
        ):
            raise RuntimeError(f"Invalid formal provenance for {key}")
        for metric in ("ade", "fde"):
            values = [float(record["eval_results"]["256"][metric]) for record in records]
            rows.append(
                {
                    "configuration": label,
                    "horizon_s": 256,
                    "metric": metric,
                    "mean_m": statistics.mean(values),
                    "sample_std_m": statistics.stdev(values),
                    "seed_count": len(values),
                }
            )
    for model, label in (
        ("Transformer (baseline)", "Transformer"),
        ("PLGAFormer (proposed)", "PLGAFormer"),
    ):
        for metric in ("ade", "fde"):
            values = main_summary["seed_statistics"][model]["256"][metric]
            rows.append(
                {
                    "configuration": label,
                    "horizon_s": 256,
                    "metric": metric,
                    "mean_m": values["mean_m"],
                    "sample_std_m": values["std_m"],
                    "seed_count": len(values["values_m"]),
                }
            )
    return rows


def _robustness_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition, models in payload["results"].items():
        for model, values in models.items():
            rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "ade_mean_m": values["ade_m"],
                    "ade_sample_std_m": values["ade_std_m"],
                    "fde_mean_m": values["fde_m"],
                    "fde_sample_std_m": values["fde_std_m"],
                    "trajectory_count": values["trajectory_count"],
                    "seed_count": len(values["model_seeds"]),
                }
            )
    return rows


def _efficiency_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    keep = (
        "model", "params", "flops_mflops", "peak_inference_memory_mb",
        "latency_batch1_ms", "throughput_batch_size", "throughput_trajectories_s",
        "ADE_256_m", "FDE_256_m", "RMSE_256_m",
    )
    return [{key: row.get(key) for key in keep} for row in payload["results"]]


def generate(output: Path) -> dict[str, Any]:
    bundle_dir = _discover_complete_bundle()
    artifact = _read_json(bundle_dir / "artifact_manifest.json")
    main = _read_json(bundle_dir / "main_results_summary.json")
    config_sha = artifact["config_sha256"]
    dataset_sha = artifact["dataset_sha256"]
    if main["source_audit"]["config_sha256"] != config_sha or main["source_audit"]["dataset_sha256"] != dataset_sha:
        raise RuntimeError("Main summary provenance does not match the paper bundle")

    robustness_path = Path(artifact["optional_evidence"]["robustness"]["path"])
    efficiency_path = Path(artifact["optional_evidence"]["efficiency"]["path"])
    for label, path in (("robustness", robustness_path), ("efficiency", efficiency_path)):
        expected = artifact["optional_evidence"][label]["sha256"]
        if _sha256(path) != expected:
            raise RuntimeError(f"{label} evidence hash mismatch")
    robustness = _read_json(robustness_path)
    efficiency = _read_json(efficiency_path)
    paired_path = ROOT / "experiments" / "mechanism_analysis" / "results" / "paired_gate_evidence" / "full_vs_schedule_only_paired_evidence.json"
    paired = _read_json(paired_path)
    if paired["formal_config"]["canonical_sha256"] != config_sha or paired["dataset"]["sha256"] != dataset_sha:
        raise RuntimeError("Paired-gate evidence provenance mismatch")

    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "main_results.csv": _main_rows(main),
        "maneuver_results_256s.csv": _maneuver_rows(main),
        "strongest_comparator.csv": main["strongest_comparator_rows"],
        "ablation_256s.csv": _ablation_rows(main, config_sha, dataset_sha),
        "robustness.csv": _robustness_rows(robustness),
        "efficiency.csv": _efficiency_rows(efficiency),
        "adaptive_gate_paired.csv": paired["comparisons"],
    }
    for name, rows in tables.items():
        if name == "adaptive_gate_paired.csv":
            flat = []
            for row in rows:
                flat.append({
                    "horizon_s": row["horizon_s"],
                    "metric": row["metric"],
                    "mean_difference_m": row["conditional_trajectory_inference"]["mean_difference"],
                    "relative_improvement_percent": row["conditional_trajectory_inference"]["relative_improvement_percent"],
                    "holm_p": row["conditional_trajectory_inference"]["p_value_holm"],
                    "crossed_ci_low_m": row["crossed_seed_trajectory_bootstrap"]["ci_low_m"],
                    "crossed_ci_high_m": row["crossed_seed_trajectory_bootstrap"]["ci_high_m"],
                    "seeds_favoring_full": row["seeds_favoring_full"],
                    "seed_count": row["seed_count"],
                    "evidence_class": row["evidence_class"],
                })
            rows = flat
        _write_csv(output / name, rows)

    manifest = {
        "schema_version": 1,
        "artifact_kind": "public_release_evidence",
        "claim_boundary": artifact["claim_boundary"],
        "dataset_protocol": artifact["dataset_protocol"],
        "dataset_sha256": dataset_sha,
        "config_sha256": config_sha,
        "paper_bundle_id": artifact["bundle_id"],
        "main_bundle_id": artifact["main_bundle_id"],
        "ablation_bundle_id": artifact["ablation_bundle_id"],
        "paired_analysis_id": paired["analysis_id"],
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "source_artifact_sha256": {
            "paper_artifact_manifest": _sha256(bundle_dir / "artifact_manifest.json"),
            "robustness": _sha256(robustness_path),
            "efficiency": _sha256(efficiency_path),
            "paired_gate": _sha256(paired_path),
        },
        "files": {name: _sha256(output / name) for name in tables},
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = generate(args.output.resolve())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
