#!/usr/bin/env python3
"""Generate signed TAES main-result artifacts from frozen-checkpoint tests.

The script intentionally refuses incomplete experiments. It uses the active
formal protocol, equal trajectory weighting for ADE/FDE, seed variability for
table uncertainty, and held-out source trajectories for paired inference.
The main table additionally reports physical ECEF RMSE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.trajectory_metrics import bootstrap_mean_ci, holm_adjust, paired_permutation_test
from data_generation.data_paths import get_dataset_npz_path
from scripts.run_formal_sota_unit import sha256_file
from utils.formal_evidence import (
    DISPLAY_NAMES,
    load_evidence_bundle,
    load_formal_config,
    normalize_run_record,
    validate_evidence_bundle,
)


EXPECTED_MODELS = [
    "Transformer (baseline)",
    "PLGAFormer (proposed)",
    "PIT",
    "Spherical kinematics",
    "Rotating-Earth 3-DOF",
    "DLinear",
    "AF-CILN",
    "PatchTST",
    "iTransformer",
]
EXPECTED_SEEDS = [42, 123, 456]
HORIZONS = [32, 64, 128, 256]
METRICS = ["ade", "fde"]
TABLE_METRICS = ["ade", "fde", "rmse_cart_m"]
TABLE_MODELS = [
    "Spherical kinematics",
    "Rotating-Earth 3-DOF",
    "DLinear",
    "Transformer (baseline)",
    "PIT",
    "PatchTST",
    "iTransformer",
    "AF-CILN",
    "PLGAFormer (proposed)",
]
DISPLAY_NAMES = {
    "Transformer (baseline)": "Transformer",
    "PLGAFormer (proposed)": "PLGAFormer",
}
ANALYTICAL_MODELS = {"Spherical kinematics", "Rotating-Earth 3-DOF"}
MANEUVERS = ["longitudinal", "turning", "weaving"]
FINAL_ARCHITECTURE = (
    "PLGAFormer with an identified rotating-Earth 3-DOF prior, adaptive bounded "
    "fusion, and no channel-residual head"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "formal_v3.json",
        help="Current frozen formal-v3 configuration.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "exp1_sota"
            / "results"
            / "formal_v3"
            / "hgv_multiregime_state_v2_1"
            / "final"
            / "main_run_set_manifest.json"
        ),
        help="Paper-eligible formal-v3 Main Results bundle manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Bundle-specific staging directory; defaults below generated/formal_v3.",
    )
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Leave the final artifact_manifest.json to the formal paper coordinator.",
    )
    parser.add_argument(
        "--allow-legacy-diagnostic",
        action="store_true",
        help="Explicitly allow historical formal-v2 directory input; never paper-facing.",
    )
    return parser.parse_args(argv)


def _load_legacy_frozen_records(
    input_dir: Path,
) -> tuple[str, dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Read the historical frozen directory only under an explicit diagnostic flag."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Frozen-result directory is missing: {input_dir}")
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    source_audit: dict[str, Any] = {"source_class": "legacy_formal_v2_diagnostic"}
    signature_items: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("frozen_sota_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("evidence_tier") != "final_frozen"
            or payload.get("test_evaluation_performed") is not True
        ):
            continue
        model = str(payload.get("model", ""))
        if model not in EXPECTED_MODELS:
            continue
        seed = int(payload.get("seed", -1))
        if model in ANALYTICAL_MODELS:
            seed = EXPECTED_SEEDS[0]
        elif seed not in EXPECTED_SEEDS:
            continue
        key = (seed, model)
        if key in selected:
            raise RuntimeError(f"Duplicate frozen record: seed={key[0]}, model={key[1]}.")
        results = payload.get("eval_results", {})
        record = {
            "seed": seed,
            "model_name": model,
            "model_type": payload.get("model_config", {}).get("model_type"),
            "results": results,
            "checkpoint_path": payload.get("checkpoint"),
            "protocol_identity": payload.get("protocol_identity"),
            "dataset_sha256": payload.get("dataset_sha256"),
            "source_json": str(path.resolve()),
        }
        selected[key] = record
        record_hash = sha256_file(path)
        source_audit[f"{seed}:{model}"] = {
            "record": str(path.resolve()),
            "record_sha256": record_hash,
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
        }
        signature_items.append({"path": str(path.resolve()), "sha256": record_hash})
    protocol_cores = set()
    dataset_hashes = set()
    for record in selected.values():
        identity = dict(record.get("protocol_identity") or {})
        identity.pop("run_signature", None)
        protocol_cores.add(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        dataset_hashes.add(str(record.get("dataset_sha256", "")))
    if len(protocol_cores) > 1:
        raise RuntimeError("Frozen records do not share one scientific protocol.")
    if len(dataset_hashes) > 1 or "" in dataset_hashes:
        raise RuntimeError("Frozen records do not share one signed dataset.")
    signature = hashlib.sha256(
        json.dumps(signature_items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return signature, selected, source_audit


def load_frozen_records(
    input_path: Path,
    *,
    allow_legacy: bool = False,
) -> tuple[str, dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Load selected records from an eligible formal-v3 bundle."""
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_dir():
        if not allow_legacy:
            raise RuntimeError(
                "Paper generation requires a formal-v3 Main Results manifest; "
                "pass --allow-legacy-diagnostic only for historical diagnostics."
            )
        return _load_legacy_frozen_records(input_path)
    bundle = load_evidence_bundle(input_path)
    if bundle.get("bundle_kind") != "main":
        raise RuntimeError(f"Expected a Main Results bundle, got {bundle.get('bundle_kind')!r}.")
    if bundle.get("paper_eligible") is not True:
        codes = sorted({str(item.get("code")) for item in bundle.get("blockers", [])})
        raise RuntimeError("Formal-v3 Main Results is incomplete: " + ", ".join(codes))
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    source_audit: dict[str, Any] = {
        "source_class": "formal_v3_main_results_bundle",
        "bundle_id": bundle.get("bundle_id"),
        "bundle": str(input_path),
        "config_sha256": bundle.get("config_sha256"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "dataset_protocol": bundle.get("dataset_protocol"),
        "test_trajectory_count": bundle.get("test_trajectory_count"),
    }
    signature_items: list[dict[str, str]] = []
    for entry in bundle.get("source_records", []):
        normalized = normalize_run_record(entry["source_path"])
        model_key = str(normalized.get("model_key", ""))
        model = DISPLAY_NAMES.get(model_key, str(normalized.get("model", model_key)))
        seed = EXPECTED_SEEDS[0] if model in ANALYTICAL_MODELS else int(normalized["seed"])
        key = (seed, model)
        if key in selected:
            raise RuntimeError(f"Duplicate bundle record: seed={seed}, model={model}.")
        selected[key] = {
            "seed": seed,
            "model_name": model,
            "model_type": normalized.get("model_config", {}).get("model_type"),
            "results": normalized.get("eval_results", {}),
            "checkpoint_path": normalized.get("checkpoint"),
            "protocol_identity": normalized.get("scientific_protocol_core"),
            "dataset_sha256": normalized.get("dataset_sha256"),
            "source_json": normalized.get("source_path"),
        }
        source_audit[f"{seed}:{model}"] = {
            "record": normalized.get("source_path"),
            "record_sha256": normalized.get("source_sha256"),
            "checkpoint": normalized.get("checkpoint"),
            "checkpoint_sha256": normalized.get("checkpoint_sha256"),
            "run_signature": normalized.get("run_signature"),
        }
        signature_items.append({"path": normalized["source_path"], "sha256": normalized["source_sha256"]})
    signature = hashlib.sha256(
        json.dumps(signature_items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_audit["record_count"] = len(selected)
    return signature, selected, source_audit


def validate_complete(
    records: dict[tuple[int, str], dict[str, Any]],
    *,
    models: list[str] = EXPECTED_MODELS,
    seeds: list[int] = EXPECTED_SEEDS,
) -> None:
    missing = []
    for model in models:
        expected_model_seeds = [seeds[0]] if model in ANALYTICAL_MODELS else seeds
        for seed in expected_model_seeds:
            if (seed, model) not in records:
                missing.append((seed, model))
    if missing:
        formatted = ", ".join(f"{seed}:{model}" for seed, model in missing)
        raise RuntimeError(f"Formal Exp1 is incomplete ({len(missing)} missing): {formatted}")

    for model in models:
        expected_model_seeds = [seeds[0]] if model in ANALYTICAL_MODELS else seeds
        for seed in expected_model_seeds:
            results = records[(seed, model)].get("results", {})
            missing_horizons = [h for h in HORIZONS if str(h) not in results and h not in results]
            if missing_horizons:
                raise RuntimeError(
                    f"Run seed={seed}, model={model} lacks horizons {missing_horizons}."
                )


def _horizon_result(record: dict[str, Any], horizon: int) -> dict[str, Any]:
    results = record["results"]
    return results.get(str(horizon), results.get(horizon, {}))


def _run_metric(record: dict[str, Any], horizon: int, metric: str) -> float:
    item = _horizon_result(record, horizon)
    preferred = f"trajectory_window_{metric}"
    if preferred in item:
        return float(item[preferred])
    if metric in item:
        return float(item[metric])
    raise RuntimeError(
        f"Run seed={record.get('seed')}, model={record.get('model_name')} "
        f"lacks {preferred}/{metric} at horizon {horizon}."
    )


def aggregate_seed_statistics(
    records: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    output: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for model in EXPECTED_MODELS:
        output[model] = {}
        for horizon in HORIZONS:
            output[model][horizon] = {}
            for metric in TABLE_METRICS:
                model_seeds = (
                    [EXPECTED_SEEDS[0]]
                    if model in ANALYTICAL_MODELS
                    else EXPECTED_SEEDS
                )
                values = np.asarray(
                    [
                        _run_metric(records[(seed, model)], horizon, metric)
                        for seed in model_seeds
                    ],
                    dtype=np.float64,
                )
                output[model][horizon][metric] = {
                    "values_m": values.tolist(),
                    "mean_m": float(values.mean()),
                    "std_m": (
                        0.0 if len(values) == 1 else float(values.std(ddof=1))
                    ),
                }
    return output


def _trajectory_values(
    records: dict[tuple[int, str], dict[str, Any]],
    model: str,
    horizon: int,
    metric: str,
) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    model_seeds = (
        [EXPECTED_SEEDS[0]] if model in ANALYTICAL_MODELS else EXPECTED_SEEDS
    )
    for seed in model_seeds:
        item = _horizon_result(records[(seed, model)], horizon)
        trajectory_metrics = item.get("trajectory_metrics", {})
        for trajectory_id, trajectory_item in trajectory_metrics.items():
            values[int(trajectory_id)].append(float(trajectory_item[metric]))
    if not values:
        raise RuntimeError(
            f"No trajectory-level {metric} values for model={model}, horizon={horizon}."
        )
    incomplete = [
        trajectory_id
        for trajectory_id, row in values.items()
        if len(row) != len(model_seeds)
    ]
    if incomplete:
        raise RuntimeError(
            f"Trajectory metrics are not available for every seed: model={model}, "
            f"horizon={horizon}, ids={incomplete[:5]}."
        )
    return {trajectory_id: float(np.mean(row)) for trajectory_id, row in values.items()}


def trajectory_inference(
    records: dict[tuple[int, str], dict[str, Any]],
    *,
    permutations: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    proposed = "PLGAFormer (proposed)"
    output: dict[str, Any] = {"comparisons": [], "plgaformer_ci": {}}
    pending: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        output["plgaformer_ci"][str(horizon)] = {}
        for metric in METRICS:
            proposed_values = _trajectory_values(records, proposed, horizon, metric)
            ordered_ids = sorted(proposed_values)
            proposed_array = np.asarray([proposed_values[key] for key in ordered_ids])
            output["plgaformer_ci"][str(horizon)][metric] = bootstrap_mean_ci(
                proposed_array,
                n_resamples=bootstrap_resamples,
                seed=4200 + horizon + METRICS.index(metric),
            )

            for comparator in EXPECTED_MODELS:
                if comparator == proposed:
                    continue
                baseline_values = _trajectory_values(records, comparator, horizon, metric)
                common_ids = sorted(set(proposed_values) & set(baseline_values))
                comparison = paired_permutation_test(
                    np.asarray([proposed_values[key] for key in common_ids]),
                    np.asarray([baseline_values[key] for key in common_ids]),
                    n_resamples=permutations,
                    seed=5200 + horizon + 100 * METRICS.index(metric),
                )
                comparison.update(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "candidate": proposed,
                        "comparator": comparator,
                        "trajectory_ids": common_ids,
                        "independent_unit": "held-out source trajectory",
                        "seed_aggregation": "mean within trajectory before inference",
                    }
                )
                pending.append(comparison)

    adjusted = holm_adjust([float(item["p_value"]) for item in pending])
    for item, p_adjusted in zip(pending, adjusted):
        item["p_value_holm"] = float(p_adjusted)
        item["significant_fwer_0_05"] = bool(p_adjusted < 0.05)
        output["comparisons"].append(item)
    return output


def _rank_map(
    seed_stats: dict[str, dict[int, dict[str, dict[str, Any]]]],
    horizon: int,
    metric: str,
) -> dict[str, int]:
    ordered = sorted(
        EXPECTED_MODELS,
        key=lambda model: seed_stats[model][horizon][metric]["mean_m"],
    )
    return {model: rank for rank, model in enumerate(ordered)}


def _format_result(
    mean_m: float,
    std_m: float,
    rank: int,
    *,
    deterministic: bool = False,
) -> str:
    raw = f"{mean_m / 1000.0:.3f}"
    if not deterministic:
        raw += f"$\\pm${std_m / 1000.0:.3f}"
    if rank == 0:
        return f"\\textbf{{{raw}}}"
    if rank == 1:
        return f"\\underline{{{raw}}}"
    return raw


def render_main_table(
    seed_stats: dict[str, dict[int, dict[str, dict[str, Any]]]],
    test_trajectory_count: int | None = None,
) -> str:
    ranks = {
        (horizon, metric): _rank_map(seed_stats, horizon, metric)
        for horizon in HORIZONS
        for metric in TABLE_METRICS
    }
    count_label = "the configured number of"
    if test_trajectory_count is not None:
        count_label = str(int(test_trajectory_count))
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{Multi-horizon ECEF position errors on {count_label} held-out source "
        r"trajectories. Learned methods report mean$\pm$sample standard deviation "
        r"across three seeds in kilometers; deterministic analytical methods are "
        r"reported once. Best results are bold; second-best results are underlined.}",
        r"\label{tab:main_results}",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{clccccccccc}",
        r"\toprule",
        r"Horizon & Metric & Spherical & Rot.\ 3-DOF & DLinear & Transformer & "
        r"PIT & PatchTST & iTransformer & AF-CILN & \textbf{PLGAFormer} \\",
        r"\midrule",
    ]
    for horizon_index, horizon in enumerate(HORIZONS):
        for metric_index, metric in enumerate(TABLE_METRICS):
            cells = []
            for model in TABLE_MODELS:
                stats = seed_stats[model][horizon][metric]
                cells.append(
                    _format_result(
                        stats["mean_m"],
                        stats["std_m"],
                        ranks[(horizon, metric)][model],
                        deterministic=model in ANALYTICAL_MODELS,
                    )
                )
            horizon_label = (
                rf"\multirow{{3}}{{*}}{{{horizon} s}}" if metric_index == 0 else ""
            )
            metric_label = {
                "ade": "ADE",
                "fde": "FDE",
                "rmse_cart_m": "RMSE",
            }[metric]
            lines.append(
                horizon_label + " & " + metric_label + " & " + " & ".join(cells) + r" \\"
            )
        if horizon_index != len(HORIZONS) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def strongest_competitor_rows(
    seed_stats: dict[str, dict[int, dict[str, dict[str, Any]]]],
    inference: dict[str, Any],
) -> list[dict[str, Any]]:
    comparison_lookup = {
        (int(row["horizon"]), str(row["metric"]), str(row["comparator"])): row
        for row in inference["comparisons"]
    }
    rows = []
    proposed = "PLGAFormer (proposed)"
    for horizon in HORIZONS:
        for metric in METRICS:
            comparator = min(
                (model for model in EXPECTED_MODELS if model != proposed),
                key=lambda model: seed_stats[model][horizon][metric]["mean_m"],
            )
            proposed_mean = seed_stats[proposed][horizon][metric]["mean_m"]
            comparator_mean = seed_stats[comparator][horizon][metric]["mean_m"]
            improvement = 100.0 * (comparator_mean - proposed_mean) / comparator_mean
            test = comparison_lookup[(horizon, metric, comparator)]
            rows.append(
                {
                    "horizon": horizon,
                    "metric": metric,
                    "comparator": comparator,
                    "proposed_mean_m": proposed_mean,
                    "comparator_mean_m": comparator_mean,
                    "relative_improvement_percent": improvement,
                    "p_value_holm": float(test["p_value_holm"]),
                    "significant_fwer_0_05": bool(test["significant_fwer_0_05"]),
                }
            )
    return rows


def render_significance_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{PLGAFormer versus the strongest non-PLGAFormer comparator for each "
        r"horizon and metric. Negative improvement indicates that the comparator is better.}",
        r"\label{tab:strongest_comparator}",
        r"\scriptsize",
        r"\begin{tabular}{cclrr}",
        r"\toprule",
        r"Horizon & Metric & Strongest comparator & Change (\%) & Holm $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        p_value = float(row["p_value_holm"])
        p_text = "$<0.001$" if p_value < 0.001 else f"{p_value:.3f}"
        comparator = DISPLAY_NAMES.get(row["comparator"], row["comparator"])
        lines.append(
            f"{row['horizon']} s & {str(row['metric']).upper()} & {comparator} & "
            f"{row['relative_improvement_percent']:.1f} & {p_text} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def trajectory_maneuver_map() -> dict[int, str]:
    with np.load(get_dataset_npz_path(PROJECT_ROOT), allow_pickle=False) as dataset:
        trajectory_ids = np.asarray(dataset["trajectory_ids_test"], dtype=np.int64)
        labels = np.asarray(dataset["maneuver_labels_test"]).astype(str)
    mapping: dict[int, str] = {}
    for trajectory_id, label in zip(trajectory_ids, labels):
        trajectory_id = int(trajectory_id)
        if trajectory_id in mapping and mapping[trajectory_id] != label:
            raise RuntimeError(f"Trajectory {trajectory_id} has inconsistent maneuver labels.")
        mapping[trajectory_id] = label
    if set(mapping.values()) != set(MANEUVERS):
        raise RuntimeError(f"Unexpected test-set maneuver taxonomy: {sorted(set(mapping.values()))}.")
    return mapping


def aggregate_maneuver_statistics(
    records: dict[tuple[int, str], dict[str, Any]],
    mapping: dict[int, str],
) -> dict[str, dict[str, dict[str, dict[str, float | list[float]]]]]:
    output = {}
    for model in EXPECTED_MODELS:
        output[model] = {}
        for maneuver in MANEUVERS:
            output[model][maneuver] = {}
            maneuver_ids = {trajectory_id for trajectory_id, label in mapping.items() if label == maneuver}
            for metric in METRICS:
                seed_values = []
                model_seeds = (
                    [EXPECTED_SEEDS[0]]
                    if model in ANALYTICAL_MODELS
                    else EXPECTED_SEEDS
                )
                for seed in model_seeds:
                    item = _horizon_result(records[(seed, model)], 256)
                    trajectory_metrics = item.get("trajectory_metrics", {})
                    values = [
                        float(trajectory_metrics[str(trajectory_id)][metric])
                        for trajectory_id in sorted(maneuver_ids)
                        if str(trajectory_id) in trajectory_metrics
                    ]
                    if len(values) != len(maneuver_ids):
                        raise RuntimeError(
                            f"Incomplete maneuver metrics: model={model}, seed={seed}, "
                            f"maneuver={maneuver}, metric={metric}."
                        )
                    seed_values.append(float(np.mean(values)))
                values_array = np.asarray(seed_values, dtype=float)
                output[model][maneuver][metric] = {
                    "values_m": values_array.tolist(),
                    "mean_m": float(values_array.mean()),
                    "std_m": 0.0 if len(values_array) == 1 else float(values_array.std(ddof=1)),
                    "trajectory_count": len(maneuver_ids),
                }
    return output


def render_maneuver_table(maneuver_stats: dict) -> str:
    ranks = {}
    for maneuver in MANEUVERS:
        for metric in METRICS:
            ordered = sorted(
                EXPECTED_MODELS,
                key=lambda model: maneuver_stats[model][maneuver][metric]["mean_m"],
            )
            ranks[(maneuver, metric)] = {
                model: rank for rank, model in enumerate(ordered)
            }
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Maneuver-resolved trajectory-level errors at 256 s. Learned "
        r"methods report mean$\pm$sample standard deviation across three seeds "
        r"in kilometers; deterministic analytical methods are reported once.}",
        r"\label{tab:maneuver_results}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Longitudinal} & \multicolumn{2}{c}{Turning} & "
        r"\multicolumn{2}{c}{Weaving} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & ADE & FDE & ADE & FDE & ADE & FDE \\",
        r"\midrule",
    ]
    for model in EXPECTED_MODELS:
        cells = []
        for maneuver in MANEUVERS:
            for metric in METRICS:
                row = maneuver_stats[model][maneuver][metric]
                cells.append(
                    _format_result(
                        float(row["mean_m"]),
                        float(row["std_m"]),
                        ranks[(maneuver, metric)][model],
                        deterministic=model in ANALYTICAL_MODELS,
                    )
                )
        label = DISPLAY_NAMES.get(model, model)
        if model == "PLGAFormer (proposed)":
            label = rf"\textbf{{{label}}}"
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    counts = [
        next(iter(maneuver_stats.values()))[maneuver]["ade"]["trajectory_count"]
        for maneuver in MANEUVERS
    ]
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{Held-out trajectory counts: "
            + f"longitudinal {counts[0]}, turning {counts[1]}, weaving {counts[2]}."
            + r"} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_legacy_diagnostic:
        config = load_formal_config(args.config)
        verification = validate_evidence_bundle(args.input, config, expected_kind="main")
        if not verification["passed"]:
            codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
            raise RuntimeError("Formal-v3 Main Results bundle failed verification: " + ", ".join(codes))
    signature, records, source_audit = load_frozen_records(
        args.input, allow_legacy=bool(args.allow_legacy_diagnostic)
    )
    validate_complete(records)
    seed_stats = aggregate_seed_statistics(records)
    inference = trajectory_inference(
        records,
        permutations=args.permutations,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    strongest_rows = strongest_competitor_rows(seed_stats, inference)
    maneuver_stats = aggregate_maneuver_statistics(records, trajectory_maneuver_map())

    if args.output_dir is None:
        bundle_id = source_audit.get("bundle_id", signature)
        args.output_dir = (
            PROJECT_ROOT
            / "experiments"
            / "taes_submission_artifacts"
            / "generated"
            / "formal_v3"
            / "hgv_multiregime_state_v2_1"
            / str(bundle_id)
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Paper staging directory is nonempty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_table = args.output_dir / "table_main_results.tex"
    significance_table = args.output_dir / "table_significance.tex"
    maneuver_table = args.output_dir / "table_maneuver_results.tex"
    summary_json = args.output_dir / "main_results_summary.json"
    main_table.write_text(
        render_main_table(seed_stats, source_audit.get("test_trajectory_count")),
        encoding="utf-8",
    )
    significance_table.write_text(render_significance_table(strongest_rows), encoding="utf-8")
    maneuver_table.write_text(render_maneuver_table(maneuver_stats), encoding="utf-8")
    summary = {
        "experiment": "exp1_sota_with_validation_selected_final_architecture",
        "run_signature": signature,
        "final_architecture": FINAL_ARCHITECTURE,
        "source_audit": source_audit,
        "expected_models": EXPECTED_MODELS,
        "expected_seeds": EXPECTED_SEEDS,
        "horizons": HORIZONS,
        "reporting_unit": "kilometers in LaTeX; meters in JSON",
        "seed_statistics": seed_stats,
        "trajectory_inference": inference,
        "strongest_comparator_rows": strongest_rows,
        "maneuver_statistics": maneuver_stats,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    artifact_files = [main_table, significance_table, maneuver_table, summary_json]
    artifact_manifest = args.output_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "formal_v3_taes_main_results",
        "bundle_id": source_audit.get("bundle_id"),
        "config_sha256": source_audit.get("config_sha256"),
        "dataset_sha256": source_audit.get("dataset_sha256") or next(iter({str(item.get("dataset_sha256", "")) for item in records.values()}), ""),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_audit": source_audit,
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in artifact_files
        },
        "claim_boundary": "complete simulated HGV trajectories; simulation-only evidence",
    }
    if not args.skip_manifest:
        temp_manifest = artifact_manifest.with_name(artifact_manifest.name + ".tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_manifest.replace(artifact_manifest)
    return {
        "run_signature": signature,
        "bundle_id": source_audit.get("bundle_id"),
        "records": len(records),
        "main_table": str(main_table),
        "significance_table": str(significance_table),
        "maneuver_table": str(maneuver_table),
        "summary_json": str(summary_json),
        "artifact_files": [str(path) for path in artifact_files],
        "artifact_manifest": None if args.skip_manifest else str(artifact_manifest),
        "test_trajectory_count": source_audit.get("test_trajectory_count"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_artifacts(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
