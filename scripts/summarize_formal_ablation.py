#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize resumable formal ablation CSV files into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PAPER_PHASE = "phase4_final_mechanism_controls"
PAPER_TRAINED_MODEL_KEYS = {"spherical_prior", "schedule_only"}
PAPER_SEEDS = {"42", "123", "456"}


def select_paper_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select only formal-v3 phase4 mechanism controls for paper staging."""
    selected = [
        row
        for row in rows
        if str(row.get("phase", "")) == PAPER_PHASE
        and str(row.get("model_key", "")) in PAPER_TRAINED_MODEL_KEYS
    ]
    missing = []
    for model_key in sorted(PAPER_TRAINED_MODEL_KEYS):
        for seed in sorted(PAPER_SEEDS, key=int):
            if not any(str(row.get("model_key")) == model_key and str(row.get("seed")) == seed for row in selected):
                missing.append(f"{model_key}:{seed}")
    if missing:
        raise RuntimeError(
            "Formal-v3 paper ablation is incomplete; missing phase4 units: "
            + ", ".join(missing)
        )
    return selected


SUMMARY_FIELDNAMES = [
    "phase",
    "model_key",
    "horizon",
    "metric",
    "n",
    "mean",
    "std",
    "min",
    "max",
    "baseline_improvement_pct",
    "full_delta_pct",
    "seeds",
]


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("timestamp", "")), str(row.get("run_id", "")))


def select_latest_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the latest row for each seed/model/horizon/metric tuple."""
    latest: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("phase", "")),
            str(row.get("seed", "")),
            str(row.get("model_key", "")),
            str(row.get("horizon", "")),
            str(row.get("metric", "")),
        )
        if key not in latest or _row_sort_key(row) >= _row_sort_key(latest[key]):
            latest[key] = row
    return list(latest.values())


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate latest metric rows by model/horizon/metric."""
    latest = select_latest_metric_rows(rows)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in latest:
        grouped[(str(row.get("phase", "")), str(row["model_key"]), str(row["horizon"]), str(row["metric"]))].append(row)

    means: dict[tuple[str, str, str, str], float] = {}
    for key, group in grouped.items():
        values = [float(row["value"]) for row in group]
        means[key] = statistics.fmean(values)

    out_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], int(item[2]), item[3], item[1])):
        phase, model_key, horizon, metric = key
        group = grouped[key]
        values = [float(row["value"]) for row in group]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        baseline_mean = means.get((phase, "baseline", horizon, metric), math.nan)
        reference_key = {
            "phase1_structural": "proposed",
            "phase2_objective_isolation": "ecef",
            "phase3_attention_prior_isolation": "attention_all",
        }.get(phase, "full")
        full_mean = means.get((phase, reference_key, horizon, metric), math.nan)
        baseline_improvement = math.nan
        if math.isfinite(baseline_mean) and baseline_mean != 0.0:
            baseline_improvement = 100.0 * (baseline_mean - mean) / baseline_mean
        full_delta = math.nan
        if math.isfinite(full_mean) and mean != 0.0:
            full_delta = 100.0 * (mean - full_mean) / mean
        out_rows.append(
            {
                "phase": phase,
                "model_key": model_key,
                "horizon": int(horizon),
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
                "baseline_improvement_pct": baseline_improvement,
                "full_delta_pct": full_delta,
                "seeds": ",".join(sorted({str(row["seed"]) for row in group}, key=int)),
            }
        )
    return out_rows


def summarize_validation_selection(
    rows: list[dict[str, Any]], records_dir: Path
) -> dict[str, Any]:
    """Summarize the validation objective used for structural model selection."""
    latest_runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("phase", "")) != "phase1_structural":
            continue
        key = (
            str(row.get("phase", "")),
            str(row.get("seed", "")),
            str(row.get("model_key", "")),
        )
        if key not in latest_runs or _row_sort_key(row) >= _row_sort_key(latest_runs[key]):
            latest_runs[key] = row

    grouped: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for (_, seed, model_key), row in latest_runs.items():
        run_id = str(row.get("run_id", "")).strip()
        record_path = records_dir / f"{run_id}.json"
        if not record_path.is_file():
            raise FileNotFoundError(
                f"Missing run record needed for validation-based selection: {record_path}"
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        best_val_loss = record.get("history", {}).get("best_val_loss")
        if best_val_loss is None:
            raise RuntimeError(f"Run record lacks best validation objective: {record_path}")
        grouped[model_key].append((int(seed), float(best_val_loss), run_id))

    candidates = []
    for model_key, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        losses = [value for _, value, _ in values]
        candidates.append(
            {
                "model_key": model_key,
                "selection_metric": "mean minimum validation objective across seeds",
                "n": len(losses),
                "mean_best_val_loss": statistics.fmean(losses),
                "std_best_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "seeds": ",".join(str(seed) for seed, _, _ in values),
                "run_ids": [run_id for _, _, run_id in values],
            }
        )
    if not candidates:
        raise RuntimeError("No phase1 structural records are available for validation selection.")
    selected = min(candidates, key=lambda item: item["mean_best_val_loss"])["model_key"]
    return {
        "rule": "minimum mean best-validation objective across seeds",
        "selected_model_key": selected,
        "candidates": candidates,
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(
    input_csv: Path,
    output_dir: Path,
    *,
    paper_facing: bool = False,
) -> dict[str, str]:
    rows = read_csv_rows(input_csv)
    if paper_facing:
        rows = select_paper_ablation_rows(rows)
    summary_rows = summarize_metric_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv = output_dir / "formal_ablation_stats.csv"
    summary_json = output_dir / "formal_ablation_summary.json"
    with stats_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "input_csv": str(input_csv),
        "stats_csv": str(stats_csv),
        "rows": summary_rows,
        "paper_facing": bool(paper_facing),
        "paper_phase": PAPER_PHASE if paper_facing else None,
        "paper_rows": ["baseline", "spherical_prior", "schedule_only", "full"] if paper_facing else None,
        "validation_selection": (
            None
            if paper_facing
            else summarize_validation_selection(rows, output_dir)
        ),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"stats_csv": str(stats_csv), "summary_json": str(summary_json)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=str,
        default="experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final/formal_ablation_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final",
    )
    parser.add_argument(
        "--paper-facing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the formal-v3 phase4 mechanism-control matrix.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    result = write_summary(input_csv, output_dir, paper_facing=bool(args.paper_facing))
    print("[formal-summary] wrote:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
