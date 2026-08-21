#!/usr/bin/env python3
"""Create post-hoc paired evidence for full versus schedule-only fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.formal_evidence import load_formal_config
from utils.trajectory_metrics import holm_adjust, paired_permutation_test


SEEDS = (42, 123, 456)
HORIZONS = (32, 64, 128, 256)
METRICS = ("ade", "fde")
PROTOCOL_FIELDS = (
    "dataset_protocol", "dataset_sha256", "sampling_interval_s", "subset_ratio",
    "epochs", "batch_size", "prediction_length", "prediction_horizons",
    "prediction_horizons_s", "train_supervision_protocol", "eval_protocol",
    "eval_ar_seed_mode", "label_len", "learning_rate", "weight_decay",
    "warmup_epochs", "patience", "gradient_clip_norm", "mixed_precision",
    "mixed_precision_dtype",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def crossed_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_resamples: int = 20_000,
    seed: int = 20260820,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Resample both trained seeds (rows) and held-out trajectories (columns)."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or np.any(~np.isfinite(values)):
        raise ValueError("differences must be a finite 2D array with both dimensions >= 2.")
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings.")
    n_seed, n_trajectory = values.shape
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        count = min(1_000, n_resamples - done)
        seed_idx = rng.integers(0, n_seed, size=(count, n_seed))
        trajectory_idx = rng.integers(
            0, n_trajectory, size=(count, n_trajectory)
        )
        sampled = values[seed_idx[:, :, None], trajectory_idx[:, None, :]]
        means[done : done + count] = sampled.mean(axis=(1, 2))
        done += count
    alpha = 1.0 - confidence
    low, high = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0))
    return {
        "mean_difference_m": float(values.mean()),
        "ci_low_m": float(low),
        "ci_high_m": float(high),
        "confidence": confidence,
        "n_seeds": n_seed,
        "n_trajectories": n_trajectory,
        "n_resamples": n_resamples,
        "random_seed": seed,
    }


def evidence_class(
    mean_difference: float,
    holm_p: float,
    crossed_ci_high: float,
    seed_differences: list[float],
) -> str:
    conditional = mean_difference < 0.0 and holm_p < 0.05
    seed_consistent = all(value < 0.0 for value in seed_differences)
    if conditional and crossed_ci_high < 0.0 and seed_consistent:
        return "seed_and_trajectory_robust_improvement"
    if conditional:
        return "trajectory_conditional_average_improvement_only"
    return "no_supported_average_improvement"


def load_records(
    directory: Path,
    pattern: str,
    model_key: str,
    config_sha256: str,
) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        record = read_json(path)
        if record.get("model_key") != model_key:
            continue
        seed = int(record.get("seed", -1))
        if seed in records:
            raise ValueError(f"Duplicate {model_key} seed {seed}.")
        if seed not in SEEDS:
            raise ValueError(f"Unexpected {model_key} seed {seed}.")
        if record.get("evidence_tier") != "final":
            raise ValueError(f"Non-final record: {path}")
        if record.get("test_evaluation_performed") is not True:
            raise ValueError(f"Missing test-evaluation attestation: {path}")
        if record.get("formal_config_sha256") != config_sha256:
            raise ValueError(f"Formal-config hash mismatch: {path}")
        checkpoint = Path(str(record.get("checkpoint", "")))
        if not checkpoint.is_file():
            raise ValueError(f"Missing checkpoint: {checkpoint}")
        if file_sha256(checkpoint) != record.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        record["_path"] = path.resolve()
        record["_sha256"] = file_sha256(path)
        records[seed] = record
    if tuple(sorted(records)) != SEEDS:
        raise ValueError(f"{model_key} requires seeds {SEEDS}; found {tuple(sorted(records))}.")
    return records


def protocol_projection(record: dict[str, Any]) -> dict[str, Any]:
    protocol = record.get("protocol_identity", {})
    missing = [field for field in PROTOCOL_FIELDS if field not in protocol]
    if missing:
        raise ValueError(f"Missing protocol fields in {record.get('run_id')}: {missing}")
    return {field: protocol[field] for field in PROTOCOL_FIELDS}


def trajectory_values(
    record: dict[str, Any], horizon: int, metric: str
) -> tuple[dict[int, float], dict[int, int]]:
    result = record.get("eval_results", {}).get(str(horizon), {})
    raw = result.get("trajectory_metrics")
    if not isinstance(raw, dict) or len(raw) < 2:
        raise ValueError(f"Missing trajectory metrics: {record.get('run_id')}, {horizon}")
    values: dict[int, float] = {}
    counts: dict[int, int] = {}
    for raw_id, item in raw.items():
        trajectory_id = int(raw_id)
        values[trajectory_id] = float(item[metric])
        counts[trajectory_id] = int(item["window_count"])
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite trajectory metric.")
    if int(result.get("trajectory_count", -1)) != len(values):
        raise ValueError("Declared trajectory count does not match trajectory metrics.")
    return values, counts


def source_manifest(role: str, records: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "seed": seed,
            "run_id": records[seed]["run_id"],
            "record_path": str(records[seed]["_path"].relative_to(ROOT)),
            "record_sha256": records[seed]["_sha256"],
            "checkpoint_path": records[seed]["checkpoint"],
            "checkpoint_sha256": records[seed]["checkpoint_sha256"],
        }
        for seed in SEEDS
    ]


def generate(
    main_dir: Path,
    ablation_dir: Path,
    config_path: Path,
    n_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    formal_config = load_formal_config(config_path)
    config_sha = formal_config.config_sha256
    full = load_records(
        main_dir, "formal_sota_seed*_full_*.json", "full", config_sha
    )
    schedule = load_records(
        ablation_dir,
        "formal_phase4_final_mechanism_controls_seed*_schedule_only_*.json",
        "schedule_only",
        config_sha,
    )
    all_records = [*(full[seed] for seed in SEEDS), *(schedule[seed] for seed in SEEDS)]
    protocol = protocol_projection(all_records[0])
    if any(protocol_projection(record) != protocol for record in all_records[1:]):
        raise ValueError("Full and schedule-only records do not share matched protocol fields.")

    dataset_paths = {
        str(record.get("dataset_identity", {}).get("dataset_path", ""))
        for record in all_records
    }
    dataset_hashes = {
        str(record.get("dataset_identity", {}).get("dataset_sha256", ""))
        for record in all_records
    }
    if len(dataset_paths) != 1 or "" in dataset_paths or len(dataset_hashes) != 1 or "" in dataset_hashes:
        raise ValueError("Dataset provenance is not identical and complete.")
    dataset_path = Path(next(iter(dataset_paths)))
    dataset_sha = next(iter(dataset_hashes))
    if not dataset_path.is_file() or file_sha256(dataset_path) != dataset_sha:
        raise ValueError("Frozen dataset is missing or its hash differs.")

    comparisons: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    reference_ids: list[int] | None = None
    for horizon in HORIZONS:
        for metric_index, metric in enumerate(METRICS):
            full_rows: list[np.ndarray] = []
            schedule_rows: list[np.ndarray] = []
            ids_for_test: list[int] | None = None
            for seed in SEEDS:
                full_values, full_counts = trajectory_values(full[seed], horizon, metric)
                schedule_values, schedule_counts = trajectory_values(
                    schedule[seed], horizon, metric
                )
                ids = sorted(full_values)
                if ids != sorted(schedule_values) or full_counts != schedule_counts:
                    raise ValueError(f"Unmatched test units for seed {seed}, {horizon}, {metric}.")
                if ids_for_test is not None and ids_for_test != ids:
                    raise ValueError("Trajectory IDs differ across seeds.")
                ids_for_test = ids
                full_rows.append(np.asarray([full_values[key] for key in ids]))
                schedule_rows.append(np.asarray([schedule_values[key] for key in ids]))
            assert ids_for_test is not None
            if reference_ids is not None and reference_ids != ids_for_test:
                raise ValueError("Trajectory IDs differ across horizons.")
            reference_ids = ids_for_test
            full_matrix = np.vstack(full_rows)
            schedule_matrix = np.vstack(schedule_rows)
            differences = full_matrix - schedule_matrix
            seed_differences = differences.mean(axis=1).tolist()
            for index, seed in enumerate(SEEDS):
                rows.append({
                    "horizon_s": horizon,
                    "metric": metric,
                    "seed": seed,
                    "trajectory_count": len(ids_for_test),
                    "full_mean_m": float(full_matrix[index].mean()),
                    "schedule_only_mean_m": float(schedule_matrix[index].mean()),
                    "difference_full_minus_schedule_m": float(seed_differences[index]),
                    "relative_improvement_percent": float(
                        -100.0 * seed_differences[index] / schedule_matrix[index].mean()
                    ),
                    "fraction_trajectories_full_better": float(
                        np.mean(differences[index] < 0.0)
                    ),
                })
            random_seed = 20260820 + horizon + 1000 * metric_index
            conditional = paired_permutation_test(
                full_matrix.mean(axis=0),
                schedule_matrix.mean(axis=0),
                n_resamples=n_resamples,
                seed=random_seed,
            )
            conditional.update({
                "independent_unit": "held-out source trajectory",
                "seed_handling": "mean within trajectory before inference",
                "units": "m",
                "random_seed": random_seed,
            })
            crossed = crossed_bootstrap_ci(
                differences,
                n_resamples=n_resamples,
                seed=random_seed + 100_000,
            )
            comparisons.append({
                "horizon_s": horizon,
                "metric": metric,
                "units": "m",
                "difference_direction": "full - schedule_only; negative favors full",
                "trajectory_count": len(ids_for_test),
                "seed_count": len(SEEDS),
                "seed_mean_differences_m": {
                    str(seed): float(value) for seed, value in zip(SEEDS, seed_differences)
                },
                "seeds_favoring_full": int(np.count_nonzero(np.asarray(seed_differences) < 0.0)),
                "conditional_trajectory_inference": conditional,
                "crossed_seed_trajectory_bootstrap": crossed,
            })
            p_values.append(float(conditional["p_value"]))

    for comparison, adjusted_p in zip(comparisons, holm_adjust(p_values)):
        conditional = comparison["conditional_trajectory_inference"]
        conditional["p_value_holm"] = float(adjusted_p)
        conditional["significant_fwer_0_05"] = bool(adjusted_p < 0.05)
        crossed = comparison["crossed_seed_trajectory_bootstrap"]
        comparison["evidence_class"] = evidence_class(
            float(conditional["mean_difference"]),
            float(adjusted_p),
            float(crossed["ci_high_m"]),
            list(comparison["seed_mean_differences_m"].values()),
        )

    assert reference_ids is not None
    script_path = Path(__file__).resolve()
    inputs = {
        "full": source_manifest("full", full),
        "schedule_only": source_manifest("schedule_only", schedule),
    }
    settings = {
        "seeds": list(SEEDS),
        "horizons_s": list(HORIZONS),
        "metrics": list(METRICS),
        "n_resamples": n_resamples,
        "independent_test_unit": "held-out source trajectory",
        "conditional_test": "two-sided paired sign-flip permutation",
        "multiplicity_control": "Holm correction across eight tests",
        "crossed_uncertainty": "bootstrap resampling seed and trajectory axes",
    }
    identity = {
        "inputs": inputs,
        "settings": settings,
        "generator_sha256": file_sha256(script_path),
    }
    payload = {
        "schema_version": 1,
        "analysis": "full_vs_schedule_only_paired_gate_evidence",
        "analysis_id": json_sha256(identity),
        "claim_boundary": (
            "Marginal adaptive-fusion evidence on complete simulated HGV trajectories; "
            "no flight, radar, or deployment validation."
        ),
        "formal_config": {
            "path": str(config_path.relative_to(ROOT)),
            "canonical_sha256": config_sha,
            "file_sha256": file_sha256(config_path),
        },
        "dataset": {
            "path": str(dataset_path),
            "protocol": protocol["dataset_protocol"],
            "sha256": dataset_sha,
        },
        "test_trajectory_count": len(reference_ids),
        "test_trajectory_ids_sha256": json_sha256(reference_ids),
        "protocol_projection": protocol,
        "generator": {
            "path": str(script_path.relative_to(ROOT)),
            "sha256": file_sha256(script_path),
        },
        "settings": settings,
        "inputs": inputs,
        "comparisons": comparisons,
        "interpretation_rule": {
            "seed_and_trajectory_robust_improvement": (
                "Holm-adjusted trajectory evidence, crossed-bootstrap interval, and all "
                "three seed effects support lower error for full."
            ),
            "trajectory_conditional_average_improvement_only": (
                "The three-seed average improves conditionally on these runs, but "
                "optimization-seed robustness is not established."
            ),
            "no_supported_average_improvement": "The prespecified average-improvement criterion is not met.",
        },
    }
    return payload, rows


def write_outputs(
    payload: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "full_vs_schedule_only_paired_evidence.json"
    csv_path = output_dir / "full_vs_schedule_only_seed_effects.csv"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-dir", type=Path,
        default=ROOT / "experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final",
    )
    parser.add_argument(
        "--ablation-dir", type=Path,
        default=ROOT / "experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/formal_v3.json")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "experiments/mechanism_analysis/results/paired_gate_evidence",
    )
    parser.add_argument("--n-resamples", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, rows = generate(
        args.main_dir.resolve(),
        args.ablation_dir.resolve(),
        args.config.resolve(),
        int(args.n_resamples),
    )
    json_path, csv_path = write_outputs(payload, rows, args.output_dir.resolve())
    print(json.dumps({
        "analysis_id": payload["analysis_id"],
        "json": str(json_path),
        "csv": str(csv_path),
        "evidence_classes": {
            f"{item['horizon_s']}_{item['metric']}": item["evidence_class"]
            for item in payload["comparisons"]
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
