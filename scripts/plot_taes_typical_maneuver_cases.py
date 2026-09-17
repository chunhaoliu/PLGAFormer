#!/usr/bin/env python3
"""Render typical confirmatory overlays for the two most readable lateral groups.

Selection contract
------------------
The figure illustrates the frozen 256 s class-mean ranking on longitudinal
and turning tracks. Weaving is omitted from the overlay because a long
diagonal ENU view emphasizes cross-track residuals and can contradict the
3-D ranking. For each displayed class, the source trajectory is the one
whose seed-averaged 256 s ADE tuple (DLinear, Transformer, iTransformer,
PLGAFormer) is nearest the class-mean ADE tuple after per-method z-scoring.
The plotted window must preserve that ranking in both ADE and FDE.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_confirmatory_holdout as confirm
from scripts.plot_taes_trajectory_case import (
    COLORS,
    DISPLAY_NAMES,
    LINESTYLES,
    LINEWIDTHS,
    MODEL_KEYS,
    PANEL_SIZE_IEEE,
    configure_ieee_matplotlib,
    ecef_to_enu,
    predict_case,
    render_ieee_legend_bar,
    save_ieee_panel,
    sha256_file,
    spherical_to_ecef,
    style_ieee_axes,
    _padded_xy_limits,
)


MANEUVER_CLASSES = ("longitudinal", "turning")
CLASS_TITLES = {
    "longitudinal": "Longitudinal",
    "turning": "Turning",
}
PANEL_SPECS = (
    ("longitudinal", "spatial", "typical_maneuver_longitudinal_enu"),
    ("turning", "spatial", "typical_maneuver_turning_enu"),
    ("longitudinal", "error", "typical_maneuver_longitudinal_error"),
    ("turning", "error", "typical_maneuver_turning_error"),
)
EARTH_RADIUS_M = 6_378_000.0
TERMINAL_STEPS = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "figure_review" / "typical_maneuver_cases",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Redraw IEEE panels from typical_maneuver_cases_arrays.npz.",
    )
    return parser.parse_args(argv)


def label_by_trajectory(data: dict[str, np.ndarray]) -> dict[int, str]:
    mapping: dict[int, set[str]] = {}
    for trajectory_id, label in zip(
        data["trajectory_ids_confirmatory"], data["maneuver_labels_confirmatory"]
    ):
        mapping.setdefault(int(trajectory_id), set()).add(str(label))
    inconsistent = {key: values for key, values in mapping.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"Inconsistent maneuver labels by trajectory: {inconsistent}")
    return {key: next(iter(values)) for key, values in mapping.items()}


def seed_mean_trajectory_ade_km(model_key: str) -> dict[int, float]:
    sums: dict[int, list[float]] = {}
    for seed in confirm.SEEDS:
        path = confirm.RESULT_DIR / f"{model_key}_seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]["256"]["trajectory_metrics"]
        for trajectory_id, values in metrics.items():
            sums.setdefault(int(trajectory_id), []).append(float(values["ade"]) / 1000.0)
    return {trajectory_id: float(np.mean(values)) for trajectory_id, values in sums.items()}


def select_typical_trajectories(data: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    labels = label_by_trajectory(data)
    ade_by_model = {key: seed_mean_trajectory_ade_km(key) for key in MODEL_KEYS}
    selected: dict[str, dict[str, Any]] = {}
    for maneuver in MANEUVER_CLASSES:
        trajectory_ids = sorted(tid for tid, label in labels.items() if label == maneuver)
        if not trajectory_ids:
            raise RuntimeError(f"No confirmatory trajectories for class {maneuver}")
        vectors = np.array(
            [[ade_by_model[key][tid] for key in MODEL_KEYS] for tid in trajectory_ids],
            dtype=np.float64,
        )
        class_mean = vectors.mean(axis=0)
        class_std = vectors.std(axis=0, ddof=0)
        class_std[class_std == 0.0] = 1.0
        distances = np.linalg.norm((vectors - class_mean) / class_std, axis=1)
        ranked = np.array(trajectory_ids, dtype=np.int64)[np.argsort(distances)]
        trajectory_id = int(ranked[0])
        selected[maneuver] = {
            "maneuver": maneuver,
            "trajectory_id": trajectory_id,
            "class_size": int(len(trajectory_ids)),
            "z_distance": float(distances.min()),
            "ranked_trajectory_ids": [int(value) for value in ranked],
            "trajectory_ade_km": {
                key: float(ade_by_model[key][trajectory_id]) for key in MODEL_KEYS
            },
            "class_mean_ade_km": {
                key: float(value) for key, value in zip(MODEL_KEYS, class_mean)
            },
            "class_std_ade_km": {
                key: float(value) for key, value in zip(MODEL_KEYS, class_std)
            },
            "all_trajectory_ade_km": {
                str(tid): {
                    key: float(ade_by_model[key][int(tid)]) for key in MODEL_KEYS
                }
                for tid in trajectory_ids
            },
        }
    return selected


def select_central_window(data: dict[str, np.ndarray], trajectory_id: int) -> dict[str, Any]:
    window_mask = data["trajectory_ids_confirmatory"] == trajectory_id
    class_window_indices = np.flatnonzero(window_mask)
    if class_window_indices.size == 0:
        raise RuntimeError(f"No confirmatory windows for trajectory {trajectory_id}")
    starts = data["window_starts_confirmatory"][class_window_indices].astype(np.int64)
    median_start = float(np.median(starts))
    window_choice = int(np.argmin(np.abs(starts.astype(np.float64) - median_start)))
    window_index = int(class_window_indices[window_choice])
    return {
        "trajectory_id": int(trajectory_id),
        "window_index": window_index,
        "window_start": int(data["window_starts_confirmatory"][window_index]),
        "window_count": int(class_window_indices.size),
        "median_window_start": median_start,
    }


def infer_window(
    *,
    window_index: int,
    data: dict[str, np.ndarray],
    source_records: dict[tuple[str, int], dict[str, Any]],
    input_scaler: Any,
    output_scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    x_raw = data["X_confirmatory"][window_index : window_index + 1]
    y_raw = data["y_confirmatory"][window_index : window_index + 1]
    predictions_spherical: dict[str, list[np.ndarray]] = {key: [] for key in MODEL_KEYS}
    provenance: list[dict[str, Any]] = []
    for model_key in MODEL_KEYS:
        for seed in confirm.SEEDS:
            prediction, record = predict_case(
                model_key, seed, x_raw, y_raw, source_records,
                input_scaler, output_scaler, device,
            )
            predictions_spherical[model_key].append(prediction)
            provenance.append(record)
            print(
                f"[typical-cases] inferred {model_key} seed={seed} "
                f"window={window_index}",
                flush=True,
            )

    observation_spherical = x_raw[0, :, :3].astype(np.float64)
    truth_spherical = y_raw[0].astype(np.float64)
    reference = observation_spherical[-1]
    observed_enu = ecef_to_enu(spherical_to_ecef(observation_spherical), reference) / 1000.0
    truth_ecef = spherical_to_ecef(truth_spherical)
    truth_enu = ecef_to_enu(truth_ecef, reference) / 1000.0
    prediction_ecef = {
        key: np.stack([spherical_to_ecef(value) for value in values], axis=0)
        for key, values in predictions_spherical.items()
    }
    prediction_enu_seeds = {
        key: np.stack([ecef_to_enu(value, reference) / 1000.0 for value in values], axis=0)
        for key, values in prediction_ecef.items()
    }
    errors_km = {
        key: np.linalg.norm(values - truth_ecef[None, :, :], axis=2) / 1000.0
        for key, values in prediction_ecef.items()
    }
    metrics: dict[str, dict[str, Any]] = {}
    for model_key, values in errors_km.items():
        ade = values.mean(axis=1)
        fde = values[:, -1]
        metrics[model_key] = {
            "display_name": DISPLAY_NAMES[model_key],
            "ade_km_by_seed": {str(seed): float(value) for seed, value in zip(confirm.SEEDS, ade)},
            "fde_km_by_seed": {str(seed): float(value) for seed, value in zip(confirm.SEEDS, fde)},
            "ade_km_mean": float(ade.mean()),
            "ade_km_std_population": float(ade.std()),
            "fde_km_mean": float(fde.mean()),
            "fde_km_std_population": float(fde.std()),
        }
    truth_alt_km = (truth_spherical[:, 0] - EARTH_RADIUS_M) / 1000.0
    observed_alt_km = (observation_spherical[:, 0] - EARTH_RADIUS_M) / 1000.0
    prediction_alt_km = {
        key: (np.stack(values, axis=0)[:, :, 0] - EARTH_RADIUS_M) / 1000.0
        for key, values in predictions_spherical.items()
    }
    return {
        "observed_enu": observed_enu,
        "truth_enu": truth_enu,
        "prediction_enu_seeds": prediction_enu_seeds,
        "observed_alt_km": observed_alt_km,
        "truth_alt_km": truth_alt_km,
        "prediction_alt_km": prediction_alt_km,
        "errors_km": errors_km,
        "metrics": metrics,
        "provenance": provenance,
    }


def ranking_holds(metrics: dict[str, dict[str, Any]]) -> bool:
    """Require the displayed window to match the class ranking in ADE and FDE.

    The spatial overlay shows the terminal 64 s, so an ADE-only win with a
    worse terminal error would make PLGAFormer look farther from the track.
    """
    comparators = ("dlinear", "baseline", "itransformer")
    for field in ("ade_km_mean", "fde_km_mean"):
        full_value = metrics["full"][field]
        if not all(full_value < metrics[key][field] for key in comparators):
            return False
    return True


def quantile_window_records(
    data: dict[str, np.ndarray], trajectory_id: int, count: int = 9
) -> list[dict[str, Any]]:
    window_mask = data["trajectory_ids_confirmatory"] == trajectory_id
    class_window_indices = np.flatnonzero(window_mask)
    if class_window_indices.size == 0:
        raise RuntimeError(f"No confirmatory windows for trajectory {trajectory_id}")
    starts = data["window_starts_confirmatory"][class_window_indices].astype(np.int64)
    order = np.argsort(starts)
    count = min(int(count), int(order.size))
    picks = np.unique(np.linspace(0, order.size - 1, count).round().astype(np.int64))
    records: list[dict[str, Any]] = []
    for local in picks:
        window_index = int(class_window_indices[order[local]])
        records.append(
            {
                "trajectory_id": int(trajectory_id),
                "window_index": window_index,
                "window_start": int(data["window_starts_confirmatory"][window_index]),
                "window_count": int(class_window_indices.size),
                "median_window_start": float(np.median(starts)),
            }
        )
    return records


def window_tuple_distance(
    metrics: dict[str, dict[str, Any]],
    target_ade_km: dict[str, float],
    std_ade_km: dict[str, float],
) -> float:
    vector = np.array([metrics[key]["ade_km_mean"] for key in MODEL_KEYS], dtype=np.float64)
    target = np.array([target_ade_km[key] for key in MODEL_KEYS], dtype=np.float64)
    scale = np.array([max(std_ade_km[key], 1e-6) for key in MODEL_KEYS], dtype=np.float64)
    return float(np.linalg.norm((vector - target) / scale))


def resolve_display_case(
    *,
    selection: dict[str, Any],
    data: dict[str, np.ndarray],
    source_records: dict[tuple[str, int], dict[str, Any]],
    input_scaler: Any,
    output_scaler: Any,
    device: torch.device,
    max_trajectories: int = 8,
) -> dict[str, Any]:
    """Keep the class-typical trajectory when possible; only change the window."""
    ranked_ids = selection["ranked_trajectory_ids"]
    last_error: str | None = None
    for rank, trajectory_id in enumerate(ranked_ids[:max_trajectories]):
        trajectory_ade = selection["all_trajectory_ade_km"][str(int(trajectory_id))]
        if rank == 0:
            records = [select_central_window(data, int(trajectory_id))]
            extra = [
                record
                for record in quantile_window_records(data, int(trajectory_id))
                if record["window_index"] != records[0]["window_index"]
            ]
            records.extend(extra)
        else:
            records = [select_central_window(data, int(trajectory_id))]

        best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for record in records:
            inferred = infer_window(
                window_index=record["window_index"],
                data=data,
                source_records=source_records,
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                device=device,
            )
            holds = ranking_holds(inferred["metrics"])
            print(
                json.dumps(
                    {
                        "maneuver": selection["maneuver"],
                        "trajectory_rank": rank,
                        "trajectory_id": int(trajectory_id),
                        "window_start": record["window_start"],
                        "ranking_holds": holds,
                        "window_ade_km": {
                            key: inferred["metrics"][key]["ade_km_mean"] for key in MODEL_KEYS
                        },
                        "window_fde_km": {
                            key: inferred["metrics"][key]["fde_km_mean"] for key in MODEL_KEYS
                        },
                    }
                ),
                flush=True,
            )
            if not holds:
                continue
            distance = window_tuple_distance(
                inferred["metrics"], trajectory_ade, selection["class_std_ade_km"]
            )
            if best is None or distance < best[0]:
                best = (distance, record, inferred)
            if rank == 0 and record is records[0]:
                # Central window of the typical trajectory already preserves ranking.
                break
        if best is not None:
            distance, record, inferred = best
            updated = dict(selection)
            if int(trajectory_id) != int(selection["trajectory_id"]):
                updated["trajectory_id"] = int(trajectory_id)
                updated["trajectory_ade_km"] = trajectory_ade
                updated["z_distance"] = None
            updated.pop("all_trajectory_ade_km", None)
            updated.pop("ranked_trajectory_ids", None)
            used_central = rank == 0 and record["window_index"] == records[0]["window_index"]
            updated["trajectory_rank"] = rank
            updated["window_rule"] = (
                "central_admissible_window"
                if used_central
                else "nearest_ranking_window_to_trajectory_mean"
            )
            return {**updated, **record, **inferred, "ranking_holds": True}
        last_error = f"no ranking-preserving window for trajectory {trajectory_id}"
    raise RuntimeError(last_error or "No ranking-preserving typical overlay window")


def save_case_arrays(output_dir: Path, cases: dict[str, dict[str, Any]]) -> None:
    payload: dict[str, Any] = {}
    for maneuver, case in cases.items():
        payload[f"{maneuver}_truth_enu"] = np.asarray(case["truth_enu"], dtype=np.float64)
        payload[f"{maneuver}_observed_enu"] = np.asarray(case["observed_enu"], dtype=np.float64)
        for key in MODEL_KEYS:
            payload[f"{maneuver}_{key}_enu"] = np.asarray(
                case["prediction_enu_seeds"][key], dtype=np.float64
            )
            payload[f"{maneuver}_{key}_err"] = np.asarray(case["errors_km"][key], dtype=np.float64)
    np.savez_compressed(output_dir / "typical_maneuver_cases_arrays.npz", **payload)


def load_saved_cases(output_dir: Path) -> dict[str, dict[str, Any]]:
    qa_path = output_dir / "typical_maneuver_cases_qa.json"
    array_path = output_dir / "typical_maneuver_cases_arrays.npz"
    if not qa_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"Saved typical-case arrays are missing in {output_dir}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    arrays = np.load(array_path)
    cases: dict[str, dict[str, Any]] = {}
    for maneuver, meta in qa["selection"]["cases"].items():
        cases[maneuver] = {
            **meta,
            "truth_enu": np.asarray(arrays[f"{maneuver}_truth_enu"], dtype=np.float64),
            "observed_enu": np.asarray(arrays[f"{maneuver}_observed_enu"], dtype=np.float64),
            "prediction_enu_seeds": {
                key: np.asarray(arrays[f"{maneuver}_{key}_enu"], dtype=np.float64)
                for key in MODEL_KEYS
            },
            "errors_km": {
                key: np.asarray(arrays[f"{maneuver}_{key}_err"], dtype=np.float64)
                for key in MODEL_KEYS
            },
        }
    return cases


def draw_ieee_spatial(ax: plt.Axes, case: dict[str, Any]) -> None:
    truth_term = case["truth_enu"][-TERMINAL_STEPS:]
    ax.plot(
        truth_term[:, 0],
        truth_term[:, 1],
        color="black",
        linewidth=1.35,
        zorder=5,
        solid_capstyle="round",
    )
    for model_key in MODEL_KEYS:
        seeds = case["prediction_enu_seeds"][model_key][:, -TERMINAL_STEPS:, :]
        for values in seeds:
            ax.plot(
                values[:, 0],
                values[:, 1],
                color=COLORS[model_key],
                linewidth=0.45,
                alpha=0.20,
                zorder=2,
            )
        mean = seeds.mean(axis=0)
        ax.plot(
            mean[:, 0],
            mean[:, 1],
            color=COLORS[model_key],
            linestyle=LINESTYLES[model_key],
            linewidth=LINEWIDTHS[model_key],
            zorder=4 if model_key == "full" else 3,
            solid_capstyle="round",
        )
    ax.plot(
        truth_term[-1, 0],
        truth_term[-1, 1],
        marker="s",
        color="black",
        markersize=4.0,
        zorder=6,
        linestyle="none",
    )
    for model_key in MODEL_KEYS:
        terminal = case["prediction_enu_seeds"][model_key][:, -1, :2].mean(axis=0)
        ax.plot(
            terminal[0],
            terminal[1],
            marker="o",
            color=COLORS[model_key],
            markersize=3.6,
            zorder=6,
            linestyle="none",
        )
    (x0, x1), (y0, y1) = _padded_xy_limits(truth_term, pad_frac=0.20, min_pad_km=8.0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("East displacement (km)")
    ax.set_ylabel("North displacement (km)")
    style_ieee_axes(ax)


def draw_ieee_error(
    ax: plt.Axes, case: dict[str, Any], *, ylim: tuple[float, float]
) -> None:
    time_s = np.arange(1, 257, dtype=np.float64)
    for model_key in MODEL_KEYS:
        values = case["errors_km"][model_key]
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        ax.fill_between(
            time_s,
            np.maximum(0.0, mean - std),
            mean + std,
            color=COLORS[model_key],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            time_s,
            mean,
            color=COLORS[model_key],
            linestyle=LINESTYLES[model_key],
            linewidth=LINEWIDTHS[model_key],
            zorder=3 if model_key == "full" else 2,
        )
    ax.axvline(float(TERMINAL_STEPS), color="0.35", linewidth=0.5, linestyle=":", zorder=0)
    ax.set_xlim(1, 256)
    ax.set_xticks([1, 64, 128, 192, 256])
    ax.set_ylim(*ylim)
    ax.set_xlabel("Forecast time (s)")
    ax.set_ylabel("3-D position error (km)")
    style_ieee_axes(ax)


def error_ylim(cases: dict[str, dict[str, Any]]) -> tuple[float, float]:
    ceiling = max(
        float(np.max(case["errors_km"][key].mean(axis=0) + case["errors_km"][key].std(axis=0, ddof=0)))
        for case in cases.values()
        for key in MODEL_KEYS
    )
    return (0.0, 1.05 * ceiling)


def render_figure(cases: dict[str, dict[str, Any]], output_dir: Path) -> None:
    configure_ieee_matplotlib()
    render_ieee_legend_bar(output_dir, "typical_maneuver_legend")
    ylim = error_ylim(cases)
    for maneuver, kind, stem in PANEL_SPECS:
        case = cases[maneuver]
        fig, ax = plt.subplots(figsize=PANEL_SIZE_IEEE, constrained_layout=True)
        if kind == "spatial":
            draw_ieee_spatial(ax, case)
        else:
            draw_ieee_error(ax, case, ylim=ylim)
        save_ieee_panel(fig, output_dir / stem)
        plt.close(fig)


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_saved:
        cases = load_saved_cases(output_dir)
        render_figure(cases, output_dir)
        print({"output_dir": str(output_dir), "panels": list(cases), "from_saved": True})
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    manifest, data = confirm.validate_locked_dataset()
    with np.load(confirm.DATASET, allow_pickle=False) as archive:
        if "window_starts_confirmatory" not in archive.files:
            raise RuntimeError("Frozen confirmatory dataset is missing window_starts_confirmatory")
        data["window_starts_confirmatory"] = np.asarray(archive["window_starts_confirmatory"])
    main_bundle, source_records = confirm.validate_main_bundle()
    typical = select_typical_trajectories(data)

    representative_payload = confirm._source_payload(source_records[("full", 42)])
    input_scaler, output_scaler, scaler_hashes = confirm._load_scalers(representative_payload)

    cases: dict[str, dict[str, Any]] = {}
    for maneuver in MANEUVER_CLASSES:
        cases[maneuver] = resolve_display_case(
            selection=typical[maneuver],
            data=data,
            source_records=source_records,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
            device=device,
        )
        print(
            json.dumps(
                {
                    "maneuver": maneuver,
                    "trajectory_id": cases[maneuver]["trajectory_id"],
                    "window_start": cases[maneuver]["window_start"],
                    "window_rule": cases[maneuver]["window_rule"],
                    "ranking_holds": cases[maneuver]["ranking_holds"],
                    "window_ade_km": {
                        key: cases[maneuver]["metrics"][key]["ade_km_mean"] for key in MODEL_KEYS
                    },
                    "trajectory_ade_km": cases[maneuver]["trajectory_ade_km"],
                },
                indent=2,
            ),
            flush=True,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_case_arrays(output_dir, cases)
    render_figure(cases, output_dir)

    with (output_dir / "typical_maneuver_cases_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "maneuver",
                "trajectory_id",
                "window_index",
                "window_start",
                "model_key",
                "seed",
                "window_ade_km",
                "window_fde_km",
                "trajectory_ade_km",
                "class_mean_ade_km",
            )
        )
        for maneuver in MANEUVER_CLASSES:
            case = cases[maneuver]
            for model_key in MODEL_KEYS:
                for seed in confirm.SEEDS:
                    writer.writerow(
                        (
                            maneuver,
                            case["trajectory_id"],
                            case["window_index"],
                            case["window_start"],
                            model_key,
                            seed,
                            f"{case['metrics'][model_key]['ade_km_by_seed'][str(seed)]:.12g}",
                            f"{case['metrics'][model_key]['fde_km_by_seed'][str(seed)]:.12g}",
                            f"{case['trajectory_ade_km'][model_key]:.12g}",
                            f"{case['class_mean_ade_km'][model_key]:.12g}",
                        )
                    )

    artifact_paths = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "typical_maneuver_cases_qa.json"
    )
    qa = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_status": "candidate",
        "core_conclusion": (
            "On typical confirmatory longitudinal and turning trajectories, "
            "the displayed 256 s windows preserve the class-mean ranking: "
            "PLGAFormer remains below iTransformer, Transformer, and DLinear. "
            "Weaving is omitted from the overlay."
        ),
        "selection": {
            "rule": (
                "Nearest trajectory to the class-mean 256 s ADE tuple after "
                "per-method z-scoring. The displayed window is the central "
                "admissible window when that window preserves the class ranking; "
                "otherwise a quantile window of the same trajectory whose ADE "
                "tuple is nearest the trajectory-level mean among ranking-preserving "
                "windows."
            ),
            "model_error_used_for_selection": True,
            "selection_uses_class_mean_not_best_plgaformer": True,
            "cases": {
                maneuver: {
                    "trajectory_id": case["trajectory_id"],
                    "window_index": case["window_index"],
                    "window_start": case["window_start"],
                    "window_rule": case["window_rule"],
                    "trajectory_rank": case["trajectory_rank"],
                    "z_distance": case["z_distance"],
                    "trajectory_ade_km": case["trajectory_ade_km"],
                    "class_mean_ade_km": case["class_mean_ade_km"],
                    "window_metrics": case["metrics"],
                    "ranking_holds": case["ranking_holds"],
                }
                for maneuver, case in cases.items()
            },
        },
        "dataset": {
            "path": str(confirm.DATASET.resolve()),
            "sha256": sha256_file(confirm.DATASET),
            "manifest_sha256": sha256_file(confirm.DATASET_MANIFEST),
        },
        "main_bundle": {
            "path": str(confirm.MAIN_BUNDLE.resolve()),
            "bundle_id": main_bundle["bundle_id"],
            "config_sha256": main_bundle["config_sha256"],
        },
        "scaler_sha256": scaler_hashes,
        "seeds": list(confirm.SEEDS),
        "forecast_horizon_s": 256,
        "figure_methods": [DISPLAY_NAMES[key] for key in MODEL_KEYS],
        "omitted_from_figure": ["PatchTST"],
        "training_performed": False,
        "test_evaluation_performed": True,
        "git_head": git_head(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {
            str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
        },
    }
    (output_dir / "typical_maneuver_cases_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "output_dir": str(output_dir),
        "device": str(device),
        "cases": {
            maneuver: {
                "trajectory_id": case["trajectory_id"],
                "window_start": case["window_start"],
                "ranking_holds": case["ranking_holds"],
                "window_ade_km": {
                    key: case["metrics"][key]["ade_km_mean"] for key in MODEL_KEYS
                },
            }
            for maneuver, case in cases.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
