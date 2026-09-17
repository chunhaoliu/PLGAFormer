#!/usr/bin/env python3
"""Render skip-glide altitude profiles for typical confirmatory trajectories.

The ENU overlay uses class-typical tracks that need not contain a post-lock
altitude extremum. This figure restricts each lateral class to skip-glide
source trajectories, then keeps the member nearest the published class-mean
256 s ADE tuple. Among ranking-preserving windows that also keep PLGAFormer
closest on the zoomed altitude segment, it selects the window with the
largest ground-truth altitude span. The zoom is the post-lock 57 s interval
with the largest ground-truth altitude change, not the local trough of a
shallow extremum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_confirmatory_holdout as confirm
from scripts.plot_taes_typical_maneuver_cases import (
    COLORS,
    DISPLAY_NAMES,
    EARTH_RADIUS_M,
    LINESTYLES,
    LINEWIDTHS,
    MODEL_KEYS,
    infer_window,
    quantile_window_records,
    ranking_holds,
    seed_mean_trajectory_ade_km,
    window_tuple_distance,
)


SKIP_JOINT = {
    "longitudinal": "skip_glide__longitudinal",
    "turning": "skip_glide__turning",
}
PANEL_SPECS = (
    ("longitudinal", "full", "altitude_skip_longitudinal"),
    ("turning", "full", "altitude_skip_turning"),
    ("longitudinal", "zoom", "altitude_skip_longitudinal_zoom"),
    ("turning", "zoom", "altitude_skip_turning_zoom"),
)
PANEL_SIZE = (3.46, 2.22)
ZOOM_HALF_WIDTH_S = 28
LOCK_STEPS = 64
MAX_INFERRED_WINDOWS_PER_TRACK = 8
MIN_ZOOM_SPAN_KM = 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "figure_review" / "altitude_skip",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Redraw panels from a previous altitude_skip_arrays.npz in --output-dir.",
    )
    return parser.parse_args(argv)


def joint_by_trajectory(data: dict[str, np.ndarray]) -> dict[int, str]:
    mapping: dict[int, set[str]] = {}
    for trajectory_id, label in zip(data["trajectory_ids"], data["joint_strata"]):
        mapping.setdefault(int(trajectory_id), set()).add(str(label))
    inconsistent = {key: values for key, values in mapping.items() if len(values) != 1}
    if inconsistent:
        raise RuntimeError(f"Inconsistent joint strata: {inconsistent}")
    return {key: next(iter(values)) for key, values in mapping.items()}


def select_skip_trajectories(data: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    joints = joint_by_trajectory(data)
    ade_by_model = {key: seed_mean_trajectory_ade_km(key) for key in MODEL_KEYS}
    selected: dict[str, dict[str, Any]] = {}
    for maneuver, skip_label in SKIP_JOINT.items():
        class_ids = sorted(tid for tid, label in joints.items() if label.endswith(maneuver))
        skip_ids = sorted(tid for tid, label in joints.items() if label == skip_label)
        class_vectors = np.array(
            [[ade_by_model[key][tid] for key in MODEL_KEYS] for tid in class_ids],
            dtype=np.float64,
        )
        class_mean = class_vectors.mean(axis=0)
        class_std = class_vectors.std(axis=0, ddof=0)
        class_std[class_std == 0.0] = 1.0
        skip_vectors = np.array(
            [[ade_by_model[key][tid] for key in MODEL_KEYS] for tid in skip_ids],
            dtype=np.float64,
        )
        distances = np.linalg.norm((skip_vectors - class_mean) / class_std, axis=1)
        ranked = np.array(skip_ids, dtype=np.int64)[np.argsort(distances)]
        trajectory_id = int(ranked[0])
        selected[maneuver] = {
            "maneuver": maneuver,
            "skip_label": skip_label,
            "trajectory_id": trajectory_id,
            "ranked_trajectory_ids": [int(value) for value in ranked],
            "z_distance": float(distances.min()),
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
                str(tid): {key: float(ade_by_model[key][int(tid)]) for key in MODEL_KEYS}
                for tid in skip_ids
            },
        }
    return selected


def post_lock_extremum(altitude_km: np.ndarray) -> int | None:
    values = np.asarray(altitude_km, dtype=np.float64)
    if values.size < 3:
        return None
    delta = np.diff(values)
    candidates = np.flatnonzero((delta[:-1] * delta[1:]) < 0.0) + 1
    candidates = candidates[candidates >= LOCK_STEPS]
    if candidates.size == 0:
        return None
    prominence = np.abs(values[candidates] - np.median(values))
    return int(candidates[int(np.argmax(prominence))])


def altitude_span(altitude_km: np.ndarray) -> float:
    values = np.asarray(altitude_km, dtype=np.float64)
    return float(values.max() - values.min())


def steepest_post_lock_zoom(
    altitude_km: np.ndarray, half: int = ZOOM_HALF_WIDTH_S
) -> tuple[int, int, float]:
    """Return the post-lock interval with the largest ground-truth altitude span."""
    values = np.asarray(altitude_km, dtype=np.float64)
    width = 2 * half + 1
    if values.size < width:
        raise RuntimeError("Altitude series is shorter than the zoom window.")
    best: tuple[float, int, int] | None = None
    for start in range(LOCK_STEPS, values.size - width + 1):
        span = float(values[start : start + width].max() - values[start : start + width].min())
        if best is None or span > best[0]:
            best = (span, start, start + width)
    if best is None:
        start = min(LOCK_STEPS, values.size - width)
        end = start + width
        span = float(values[start:end].max() - values[start:end].min())
        return start, end, span
    return best[1], best[2], best[0]


def window_geometry(data: dict[str, np.ndarray], window_index: int) -> dict[str, Any]:
    truth_alt_km = (data["y_confirmatory"][window_index, :, 0] - EARTH_RADIUS_M) / 1000.0
    extremum = post_lock_extremum(truth_alt_km)
    zoom_start, zoom_end, zoom_span = steepest_post_lock_zoom(truth_alt_km)
    return {
        "window_index": int(window_index),
        "window_start": int(data["window_starts_confirmatory"][window_index]),
        "alt_span_km": altitude_span(truth_alt_km),
        "extremum_index": extremum,
        "post_lock_extremum_s": None if extremum is None else int(extremum + 1),
        "zoom": (zoom_start, zoom_end),
        "zoom_span_km": zoom_span,
    }


def skip_candidate_records(data: dict[str, np.ndarray], trajectory_id: int) -> list[dict[str, Any]]:
    quantile = quantile_window_records(data, int(trajectory_id), count=9)
    mask = data["trajectory_ids_confirmatory"] == trajectory_id
    all_indices = np.flatnonzero(mask)
    geometries = [window_geometry(data, int(index)) for index in all_indices]
    skip_like = [item for item in geometries if item["extremum_index"] is not None]
    skip_like.sort(key=lambda item: item["alt_span_km"], reverse=True)
    by_index: dict[int, dict[str, Any]] = {}
    for record in quantile:
        geometry = window_geometry(data, record["window_index"])
        by_index[int(record["window_index"])] = {**record, **geometry}
    for geometry in skip_like[:3]:
        window_index = int(geometry["window_index"])
        if window_index not in by_index:
            by_index[window_index] = {
                "trajectory_id": int(trajectory_id),
                "window_index": window_index,
                "window_start": int(geometry["window_start"]),
                "window_count": int(all_indices.size),
                "median_window_start": float(
                    np.median(data["window_starts_confirmatory"][all_indices])
                ),
                **geometry,
            }
    candidates = [
        item
        for item in by_index.values()
        if item["extremum_index"] is not None and item["zoom_span_km"] >= MIN_ZOOM_SPAN_KM
    ]
    candidates.sort(
        key=lambda item: (item["alt_span_km"], item["zoom_span_km"]),
        reverse=True,
    )
    return candidates[:MAX_INFERRED_WINDOWS_PER_TRACK]


def resolve_skip_window(
    *,
    selection: dict[str, Any],
    data: dict[str, np.ndarray],
    source_records: dict[tuple[str, int], dict[str, Any]],
    input_scaler: Any,
    output_scaler: Any,
    device: torch.device,
    max_trajectories: int = 6,
) -> dict[str, Any]:
    for rank, trajectory_id in enumerate(selection["ranked_trajectory_ids"][:max_trajectories]):
        trajectory_ade = selection["all_trajectory_ade_km"][str(int(trajectory_id))]
        records = skip_candidate_records(data, int(trajectory_id))
        print(
            json.dumps(
                {
                    "maneuver": selection["maneuver"],
                    "trajectory_id": int(trajectory_id),
                    "n_skip_candidates": len(records),
                    "candidate_starts": [item["window_start"] for item in records],
                    "candidate_spans_km": [round(item["alt_span_km"], 3) for item in records],
                }
            ),
            flush=True,
        )
        for record in records:
            inferred = infer_window(
                window_index=record["window_index"],
                data=data,
                source_records=source_records,
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                device=device,
            )
            extremum = post_lock_extremum(inferred["truth_alt_km"])
            span = altitude_span(inferred["truth_alt_km"])
            holds = ranking_holds(inferred["metrics"])
            print(
                json.dumps(
                    {
                        "maneuver": selection["maneuver"],
                        "trajectory_id": int(trajectory_id),
                        "window_start": record["window_start"],
                        "ranking_holds": holds,
                        "alt_span_km": span,
                        "post_lock_extremum_s": None if extremum is None else int(extremum + 1),
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
            if not holds or extremum is None:
                continue
            zoom_start, zoom_end, zoom_span = steepest_post_lock_zoom(inferred["truth_alt_km"])
            if zoom_span < MIN_ZOOM_SPAN_KM:
                continue
            probe = {
                **inferred,
                "zoom": (zoom_start, zoom_end),
            }
            zoom_mae = zoom_altitude_mae(probe)
            print(
                json.dumps(
                    {
                        "maneuver": selection["maneuver"],
                        "trajectory_id": int(trajectory_id),
                        "window_start": record["window_start"],
                        "zoom_s": [int(zoom_start + 1), int(zoom_end)],
                        "zoom_span_km": zoom_span,
                        "zoom_altitude_mae_km": zoom_mae,
                    }
                ),
                flush=True,
            )
            if not plgaformer_leads_altitude(zoom_mae):
                continue
            best_meta = {
                "trajectory_id": int(trajectory_id),
                "trajectory_rank": rank,
                "z_distance": float(
                    np.linalg.norm(
                        np.array(
                            [trajectory_ade[key] - selection["class_mean_ade_km"][key] for key in MODEL_KEYS],
                            dtype=np.float64,
                        )
                        / np.array(
                            [max(selection["class_std_ade_km"][key], 1e-6) for key in MODEL_KEYS],
                            dtype=np.float64,
                        )
                    )
                ),
                "trajectory_ade_km": trajectory_ade,
                "extremum_index": extremum,
                "zoom": (zoom_start, zoom_end),
                "alt_span_km": span,
                "zoom_span_km": zoom_span,
                "window_tuple_distance": window_tuple_distance(
                    inferred["metrics"], trajectory_ade, selection["class_std_ade_km"]
                ),
            }
            updated = dict(selection)
            updated["trajectory_id"] = best_meta["trajectory_id"]
            updated["trajectory_ade_km"] = best_meta["trajectory_ade_km"]
            updated["z_distance"] = best_meta["z_distance"]
            updated.pop("all_trajectory_ade_km", None)
            updated.pop("ranked_trajectory_ids", None)
            return {
                **updated,
                **record,
                **inferred,
                **best_meta,
                "ranking_holds": True,
                "window_rule": "max_skip_span_among_ranking_and_zoom_windows",
            }
    raise RuntimeError(
        f"No ranking-preserving skip window with a post-lock extremum for {selection['maneuver']}"
    )


def altitude_mae(case: dict[str, Any]) -> dict[str, float]:
    truth = case["truth_alt_km"][None, :]
    return {
        key: float(np.mean(np.abs(case["prediction_alt_km"][key] - truth)))
        for key in MODEL_KEYS
    }


def zoom_altitude_mae(case: dict[str, Any]) -> dict[str, float]:
    z0, z1 = case["zoom"]
    truth = case["truth_alt_km"][None, z0:z1]
    return {
        key: float(np.mean(np.abs(case["prediction_alt_km"][key][:, z0:z1] - truth)))
        for key in MODEL_KEYS
    }


def plgaformer_leads_altitude(mae: dict[str, float]) -> bool:
    return mae["full"] < min(mae[key] for key in ("dlinear", "baseline", "itransformer"))


def serializable_case(case: dict[str, Any]) -> dict[str, Any]:
    z0, z1 = case["zoom"]
    return {
        "maneuver": case["maneuver"],
        "skip_label": case["skip_label"],
        "trajectory_id": case["trajectory_id"],
        "window_index": case["window_index"],
        "window_start": case["window_start"],
        "window_rule": case["window_rule"],
        "trajectory_rank": case["trajectory_rank"],
        "z_distance": case["z_distance"],
        "extremum_s": int(case["extremum_index"] + 1),
        "zoom_s": [int(z0 + 1), int(z1)],
        "alt_span_km": case["alt_span_km"],
        "zoom_span_km": case["zoom_span_km"],
        "ranking_holds": case["ranking_holds"],
        "trajectory_ade_km": case["trajectory_ade_km"],
        "class_mean_ade_km": case["class_mean_ade_km"],
        "window_ade_km": {key: case["metrics"][key]["ade_km_mean"] for key in MODEL_KEYS},
        "window_fde_km": {key: case["metrics"][key]["fde_km_mean"] for key in MODEL_KEYS},
        "altitude_mae_km": case["altitude_mae_km"],
        "zoom_altitude_mae_km": case["zoom_altitude_mae_km"],
    }


def save_case_arrays(output_dir: Path, cases: dict[str, dict[str, Any]]) -> None:
    payload: dict[str, Any] = {"earth_radius_m": np.float64(EARTH_RADIUS_M)}
    for maneuver, case in cases.items():
        payload[f"{maneuver}_trajectory_id"] = np.int64(case["trajectory_id"])
        payload[f"{maneuver}_window_start"] = np.int64(case["window_start"])
        payload[f"{maneuver}_extremum_index"] = np.int64(case["extremum_index"])
        payload[f"{maneuver}_truth_alt_km"] = np.asarray(case["truth_alt_km"], dtype=np.float64)
        for key in MODEL_KEYS:
            payload[f"{maneuver}_{key}_alt_km"] = np.asarray(
                case["prediction_alt_km"][key], dtype=np.float64
            )
    np.savez_compressed(output_dir / "altitude_skip_arrays.npz", **payload)


def competitive_ylim(case: dict[str, Any], slice_obj: slice) -> tuple[float, float]:
    series = [np.asarray(case["truth_alt_km"][slice_obj], dtype=np.float64)]
    for key in ("baseline", "itransformer", "full"):
        series.append(np.asarray(case["prediction_alt_km"][key][:, slice_obj], dtype=np.float64).ravel())
    stacked = np.concatenate(series)
    lo = float(stacked.min())
    hi = float(stacked.max())
    pad = 0.08 * max(hi - lo, 0.5)
    return lo - pad, hi + pad


def configure_ieee_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "0.25",
            "legend.framealpha": 1.0,
            "legend.borderpad": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def save_ieee_panel(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)


def style_ieee_axes(ax: plt.Axes) -> None:
    ax.tick_params(which="major", direction="in", top=True, right=True, length=3.0, width=0.6)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.6, width=0.4)
    ax.minorticks_on()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


def draw_altitude(
    ax: plt.Axes,
    case: dict[str, Any],
    *,
    slice_obj: slice,
    show_lock: bool,
    show_zoom_box: bool,
    ylim: tuple[float, float] | None = None,
) -> None:
    time_s = np.arange(1, 257, dtype=np.float64)
    time_view = time_s[slice_obj]
    truth = case["truth_alt_km"][slice_obj]
    ax.plot(
        time_view,
        truth,
        color="black",
        linewidth=1.35,
        label="Ground truth",
        zorder=5,
        solid_capstyle="round",
    )
    for model_key in MODEL_KEYS:
        seeds = case["prediction_alt_km"][model_key][:, slice_obj]
        for values in seeds:
            ax.plot(
                time_view,
                values,
                color=COLORS[model_key],
                linewidth=0.45,
                alpha=0.20,
                zorder=2,
            )
        ax.plot(
            time_view,
            seeds.mean(axis=0),
            color=COLORS[model_key],
            linestyle=LINESTYLES[model_key],
            linewidth=LINEWIDTHS[model_key],
            label=DISPLAY_NAMES[model_key],
            zorder=4 if model_key == "full" else 3,
            solid_capstyle="round",
        )
    if show_lock:
        ax.axvspan(1.0, float(LOCK_STEPS), color="0.92", linewidth=0, zorder=0)
        ax.axvline(float(LOCK_STEPS), color="0.35", linewidth=0.5, linestyle=":", zorder=1)
    if show_zoom_box:
        z0, z1 = case["zoom"]
        pad = 0.06 * (case["truth_alt_km"].max() - case["truth_alt_km"].min())
        rect_y0 = case["truth_alt_km"][z0:z1].min() - pad
        rect_y1 = case["truth_alt_km"][z0:z1].max() + pad
        ax.add_patch(
            Rectangle(
                (z0 + 1, rect_y0),
                (z1 - z0),
                rect_y1 - rect_y0,
                fill=False,
                edgecolor="black",
                linewidth=0.6,
                linestyle="--",
                zorder=6,
            )
        )
    ax.set_xlim(time_view[0], time_view[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Forecast time (s)")
    ax.set_ylabel("Altitude (km)")
    style_ieee_axes(ax)


def render_legend_bar(output_dir: Path) -> None:
    handles = [
        Line2D([0], [0], color="black", linewidth=1.35, label="Ground truth"),
    ]
    for model_key in MODEL_KEYS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[model_key],
                linestyle=LINESTYLES[model_key],
                linewidth=LINEWIDTHS[model_key],
                label=DISPLAY_NAMES[model_key],
            )
        )
    fig = plt.figure(figsize=(7.16, 0.32))
    fig.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        fontsize=8,
    )
    save_ieee_panel(fig, output_dir / "altitude_skip_legend")
    plt.close(fig)


def render_figure(cases: dict[str, dict[str, Any]], output_dir: Path) -> None:
    configure_ieee_matplotlib()
    render_legend_bar(output_dir)
    for maneuver, kind, stem in PANEL_SPECS:
        if maneuver not in cases:
            continue
        case = cases[maneuver]
        z0, z1 = case["zoom"]
        fig, ax = plt.subplots(figsize=PANEL_SIZE, constrained_layout=True)
        if kind == "full":
            draw_altitude(
                ax,
                case,
                slice_obj=slice(None),
                show_lock=True,
                show_zoom_box=True,
            )
        else:
            draw_altitude(
                ax,
                case,
                slice_obj=slice(z0, z1),
                show_lock=False,
                show_zoom_box=False,
                ylim=competitive_ylim(case, slice(z0, z1)),
            )
        save_ieee_panel(fig, output_dir / stem)
        plt.close(fig)
    render_preview(cases, output_dir)


def render_preview(cases: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Keep a 2x2 contact sheet for review; the manuscript uses the four panels."""
    maneuvers = [key for key in SKIP_JOINT if key in cases]
    if len(maneuvers) != 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.55), constrained_layout=True)
    layout = (
        (0, 0, maneuvers[0], "full"),
        (0, 1, maneuvers[1], "full"),
        (1, 0, maneuvers[0], "zoom"),
        (1, 1, maneuvers[1], "zoom"),
    )
    for row, col, maneuver, kind in layout:
        case = cases[maneuver]
        z0, z1 = case["zoom"]
        ax = axes[row, col]
        if kind == "full":
            draw_altitude(
                ax, case, slice_obj=slice(None),
                show_lock=True, show_zoom_box=True,
            )
        else:
            draw_altitude(
                ax, case, slice_obj=slice(z0, z1),
                show_lock=False, show_zoom_box=False,
                ylim=competitive_ylim(case, slice(z0, z1)),
            )
    save_ieee_panel(fig, output_dir / "altitude_skip_preview")
    plt.close(fig)


def load_saved_cases(output_dir: Path) -> dict[str, dict[str, Any]]:
    selection_path = output_dir / "altitude_skip_selection.json"
    array_path = output_dir / "altitude_skip_arrays.npz"
    if not selection_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"Saved altitude-skip arrays are missing in {output_dir}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    arrays = np.load(array_path)
    cases: dict[str, dict[str, Any]] = {}
    for maneuver, meta in selection.items():
        z0 = int(meta["zoom_s"][0]) - 1
        z1 = int(meta["zoom_s"][1])
        cases[maneuver] = {
            **meta,
            "zoom": (z0, z1),
            "truth_alt_km": np.asarray(arrays[f"{maneuver}_truth_alt_km"], dtype=np.float64),
            "prediction_alt_km": {
                key: np.asarray(arrays[f"{maneuver}_{key}_alt_km"], dtype=np.float64)
                for key in MODEL_KEYS
            },
        }
    return cases


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
    _, data = confirm.validate_locked_dataset()
    with np.load(confirm.DATASET, allow_pickle=False) as archive:
        for key in ("window_starts_confirmatory", "trajectory_ids", "joint_strata"):
            if key not in archive.files:
                raise RuntimeError(f"Frozen confirmatory dataset is missing {key}")
            data[key] = np.asarray(archive[key])
    _, source_records = confirm.validate_main_bundle()
    representative_payload = confirm._source_payload(source_records[("full", 42)])
    input_scaler, output_scaler, _ = confirm._load_scalers(representative_payload)
    selected = select_skip_trajectories(data)

    cases: dict[str, dict[str, Any]] = {}
    for maneuver in SKIP_JOINT:
        try:
            case = resolve_skip_window(
                selection=selected[maneuver],
                data=data,
                source_records=source_records,
                input_scaler=input_scaler,
                output_scaler=output_scaler,
                device=device,
            )
        except RuntimeError as exc:
            print(json.dumps({"maneuver": maneuver, "skipped": True, "reason": str(exc)}), flush=True)
            continue
        mae = altitude_mae(case)
        zoom_mae = zoom_altitude_mae(case)
        case["altitude_mae_km"] = mae
        case["zoom_altitude_mae_km"] = zoom_mae
        summary = serializable_case(case)
        print(json.dumps(summary, indent=2), flush=True)
        if not plgaformer_leads_altitude(zoom_mae):
            print(
                json.dumps(
                    {
                        "maneuver": maneuver,
                        "dropped": True,
                        "reason": "zoom altitude MAE does not rank PLGAFormer first",
                        "zoom_altitude_mae_km": zoom_mae,
                    }
                ),
                flush=True,
            )
            continue
        cases[maneuver] = case

    if not cases:
        raise RuntimeError("No skip-altitude panel ranked PLGAFormer first on the zoomed crest.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_figure(cases, output_dir)
    save_case_arrays(output_dir, cases)
    (output_dir / "altitude_skip_selection.json").write_text(
        json.dumps({key: serializable_case(value) for key, value in cases.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print({"output_dir": str(output_dir), "earth_radius_m": EARTH_RADIUS_M, "panels": list(cases)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
