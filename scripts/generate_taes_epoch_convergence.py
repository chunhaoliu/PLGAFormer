#!/usr/bin/env python3
"""Generate formal three-seed training and validation convergence figures for TAES."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "convergence_pilot"
)
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "experiments"
    / "exp1_sota"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "main_run_set_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "taes_submission_artifacts" / "generated" / "formal_v3"
SEEDS = (42, 123, 456)
MAX_EPOCHS = 50
MODEL_ORDER = ("plgaformer", "transformer")
DISPLAY_NAMES = {
    "plgaformer": "PLGAFormer",
    "transformer": "Transformer",
}
COLORS = {
    "plgaformer": "#0F4D92",
    "transformer": "#737A86",
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR,
        help="Historical convergence-pilot directory; requires --allow-legacy-diagnostic.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-legacy-diagnostic", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_epoch_values(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not 1 <= len(array) <= MAX_EPOCHS:
        raise RuntimeError(
            f"{label} must contain 1--{MAX_EPOCHS} actual epoch values."
        )
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{label} contains a non-finite epoch value.")
    if np.any(array <= 0.0):
        raise RuntimeError(f"{label} must remain positive for logarithmic display.")
    return array


def _matches_formal_selection(
    payload: dict[str, Any], model_key: str, seed: int
) -> bool:
    config = payload.get("config", {})
    history = payload.get("training_history", {})
    return (
        payload.get("evidence_tier") == "convergence_pilot"
        and payload.get("test_evaluation_performed") is False
        and payload.get("model_key") == model_key
        and int(payload.get("seed", -1)) == int(seed)
        and int(config.get("epochs", -1)) == MAX_EPOCHS
        and float(config.get("subset_ratio", -1.0)) == 1.0
        and int(config.get("batch_size", -1)) == 128
        and int(config.get("prediction_length", -1)) == 256
        and list(map(int, payload.get("horizons", []))) == [32, 64, 128, 256]
        and int(config.get("label_len", -1)) == 128
        and config.get("train_supervision_protocol")
        == "source_context_pred_window"
        and config.get("eval_protocol") == "source_context_decoder"
        and config.get("eval_ar_seed_mode") == "zero"
        and bool(config.get("amp", False)) is False
        and int(history.get("epochs_requested", -1)) == MAX_EPOCHS
        and int(history.get("epochs_completed", 0)) >= 1
    )


def load_formal_histories(
    selection_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    histories: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model_type: {} for model_type in MODEL_ORDER
    }
    source_audit: dict[str, Any] = {}
    key_map = {
        "plgaformer": "full",
        "transformer": "baseline",
    }
    for model_type in MODEL_ORDER:
        source_audit[DISPLAY_NAMES[model_type]] = {}
        for seed in SEEDS:
            matches: list[tuple[Path, dict[str, Any]]] = []
            for path in selection_dir.glob(
                f"formal_sota_seed{seed}_{key_map[model_type]}_*.json"
            ):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if _matches_formal_selection(payload, key_map[model_type], seed):
                    matches.append((path, payload))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one formal {model_type} history for seed={seed}; "
                    f"found {len(matches)}."
                )
            record_path, record = matches[0]
            history = record.get("training_history", {})
            if history.get("supervision_protocol") != "source_context_pred_window":
                raise RuntimeError(f"{model_type}, seed={seed} has a protocol mismatch.")
            train_values = _finite_epoch_values(
                history.get("train_losses"),
                f"{model_type} seed={seed} training history",
            )
            validation_values = _finite_epoch_values(
                history.get("val_losses"),
                f"{model_type} seed={seed} validation history",
            )
            if len(train_values) != len(validation_values):
                raise RuntimeError(
                    f"{model_type}, seed={seed} train/validation epoch counts differ."
                )
            histories[model_type][str(seed)] = {
                "train": train_values,
                "validation": validation_values,
            }
            checkpoint = Path(record.get("checkpoint", "")).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Formal checkpoint is missing: {checkpoint}")
            source_audit[DISPLAY_NAMES[model_type]][str(seed)] = {
                "record": str(record_path.resolve()),
                "record_sha256": sha256_file(record_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "epochs_completed": len(train_values),
                "best_epoch": int(history.get("best_epoch", -1)),
                "best_validation_objective": float(history.get("best_val_loss")),
            }

    audit = {
        "sources": source_audit,
        "objective": "shared ECEF composite objective (ECEFTrajectoryLoss)",
        "maximum_training_epochs": MAX_EPOCHS,
        "early_stopping_patience": 15,
        "seeds": list(SEEDS),
        "supervision_protocol": "source_context_pred_window",
        "test_metrics_used": False,
    }
    return histories, audit


def load_formal_histories_from_bundle(
    bundle_path: Path,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    """Read training histories from the eligible formal-v3 Main bundle."""
    from utils.formal_evidence import load_evidence_bundle

    bundle = load_evidence_bundle(bundle_path)
    if bundle.get("bundle_kind") != "main" or bundle.get("paper_eligible") is not True:
        raise RuntimeError("Convergence generation requires an eligible formal-v3 Main bundle.")
    key_map = {"plgaformer": "full", "transformer": "baseline"}
    histories: dict[str, dict[str, dict[str, np.ndarray]]] = {
        model_type: {} for model_type in MODEL_ORDER
    }
    source_audit: dict[str, Any] = {
        "bundle_id": bundle.get("bundle_id"),
        "config_sha256": bundle.get("config_sha256"),
        "dataset_sha256": bundle.get("dataset_sha256"),
        "sources": {},
    }
    for model_type in MODEL_ORDER:
        model_key = key_map[model_type]
        source_audit["sources"][DISPLAY_NAMES[model_type]] = {}
        for seed in SEEDS:
            matches = [
                entry
                for entry in bundle.get("source_records", [])
                if str(entry.get("model_key")) == model_key
                and int(entry.get("seed", -1)) == int(seed)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one formal Main history for {model_type}, seed={seed}; found {len(matches)}."
                )
            entry = matches[0]
            record_path = Path(str(entry["source_path"])).resolve()
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            history = payload.get("training_history", {})
            if history.get("supervision_protocol") != "source_context_pred_window":
                raise RuntimeError(f"{model_type}, seed={seed} has a protocol mismatch.")
            train_values = _finite_epoch_values(
                history.get("train_losses"),
                f"{model_type} seed={seed} training history",
            )
            validation_values = _finite_epoch_values(
                history.get("val_losses"),
                f"{model_type} seed={seed} validation history",
            )
            if len(train_values) != len(validation_values):
                raise RuntimeError(f"{model_type}, seed={seed} train/validation epoch counts differ.")
            histories[model_type][str(seed)] = {
                "train": train_values,
                "validation": validation_values,
            }
            checkpoint = Path(str(entry.get("checkpoint", ""))).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Formal checkpoint is missing: {checkpoint}")
            recorded_checkpoint_hash = str(entry.get("checkpoint_sha256", ""))
            observed_checkpoint_hash = sha256_file(checkpoint)
            if not recorded_checkpoint_hash or recorded_checkpoint_hash.lower() != observed_checkpoint_hash.lower():
                raise RuntimeError(f"Formal checkpoint hash mismatch: {checkpoint}")
            source_audit["sources"][DISPLAY_NAMES[model_type]][str(seed)] = {
                "record": str(record_path),
                "record_sha256": sha256_file(record_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed_checkpoint_hash,
                "epochs_completed": len(train_values),
                "best_epoch": int(history.get("best_epoch", -1)),
                "best_validation_objective": float(history.get("best_val_loss")),
            }
    source_audit.update(
        {
            "objective": "shared ECEF composite objective (ECEFTrajectoryLoss)",
            "maximum_training_epochs": MAX_EPOCHS,
            "early_stopping_patience": 15,
            "seeds": list(SEEDS),
            "supervision_protocol": "source_context_pred_window",
            "test_metrics_used": False,
        }
    )
    return histories, source_audit


def aggregate_histories(
    histories: dict[str, dict[str, dict[str, np.ndarray]]]
) -> dict[str, dict[str, np.ndarray]]:
    summary: dict[str, dict[str, np.ndarray]] = {}
    for model_type in MODEL_ORDER:
        if sorted(map(int, histories[model_type])) != list(SEEDS):
            raise RuntimeError(f"{model_type} does not contain the three formal seeds.")
        summary[model_type] = {}
        common_epochs = min(
            len(histories[model_type][str(seed)]["train"]) for seed in SEEDS
        )
        summary[model_type]["common_epochs"] = common_epochs
        for split in ("train", "validation"):
            values = np.stack(
                [
                    histories[model_type][str(seed)][split][:common_epochs]
                    for seed in SEEDS
                ],
                axis=0,
            )
            summary[model_type][f"{split}_mean"] = values.mean(axis=0)
            summary[model_type][f"{split}_std"] = values.std(axis=0, ddof=1)
    return summary


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path_base: Path) -> list[str]:
    outputs = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 600}),
    ):
        path = path_base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.025, **kwargs)
        outputs.append(str(path.resolve()))
    plt.close(fig)
    return outputs


def plot_convergence(
    summary: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> list[str]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    for split, ylabel, filename in (
        ("train", "Training ECEF objective", "fig_epoch_training_objective"),
        (
            "validation",
            "Validation ECEF objective",
            "fig_epoch_validation_objective",
        ),
    ):
        fig, ax = plt.subplots(figsize=(3.30, 2.12))
        for model_type in MODEL_ORDER:
            mean = summary[model_type][f"{split}_mean"]
            std = summary[model_type][f"{split}_std"]
            epochs = np.arange(1, len(mean) + 1)
            linewidth = 2.0 if model_type == "plgaformer" else 1.3
            zorder = 5 if model_type == "plgaformer" else 3
            ax.plot(
                epochs,
                mean,
                color=COLORS[model_type],
                linewidth=linewidth,
                marker="o",
                markersize=2.6 if model_type == "plgaformer" else 2.2,
                markerfacecolor="white",
                markeredgewidth=0.65,
                zorder=zorder,
            )
            selected = sorted(
                {
                    0,
                    min(9, len(mean) - 1),
                    min(24, len(mean) - 1),
                    len(mean) - 1,
                }
            )
            ax.errorbar(
                epochs[selected],
                mean[selected],
                yerr=std[selected],
                fmt="none",
                ecolor=COLORS[model_type],
                elinewidth=0.6,
                capsize=1.5,
                capthick=0.6,
                alpha=0.8,
                zorder=zorder - 1,
            )
        ax.set_yscale("log")
        ax.set_xlim(1, MAX_EPOCHS)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        generated.extend(save_figure(fig, output_dir / filename))

    fig, ax = plt.subplots(figsize=(4.65, 0.22))
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[model_type],
            linewidth=2.0 if model_type == "plgaformer" else 1.3,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            label=DISPLAY_NAMES[model_type],
        )
        for model_type in MODEL_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        handlelength=2.4,
        columnspacing=1.8,
    )
    generated.extend(save_figure(fig, output_dir / "fig_epoch_convergence_legend"))
    return generated


def write_csv(
    histories: dict[str, dict[str, dict[str, np.ndarray]]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "seed", "epoch", "training_objective", "validation_objective"]
        )
        for model_type in MODEL_ORDER:
            for seed in SEEDS:
                current = histories[model_type][str(seed)]
                for index in range(len(current["train"])):
                    writer.writerow(
                        [
                            DISPLAY_NAMES[model_type],
                            seed,
                            index + 1,
                            f"{current['train'][index]:.12g}",
                            f"{current['validation'][index]:.12g}",
                        ]
                    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bundle.is_file():
        histories, evidence_audit = load_formal_histories_from_bundle(args.bundle)
        bundle_id = str(evidence_audit.get("bundle_id"))
    elif args.allow_legacy_diagnostic:
        histories, evidence_audit = load_formal_histories(args.selection_dir)
        bundle_id = "legacy_diagnostic"
        evidence_audit["source_class"] = "legacy_convergence_pilot_diagnostic"
    else:
        raise FileNotFoundError(
            "Formal-v3 Main Results bundle is missing; historical convergence-pilot input "
            "requires --allow-legacy-diagnostic."
        )
    summary = aggregate_histories(histories)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT / "hgv_multiregime_state_v2_1" / bundle_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "epoch_convergence.csv"
    write_csv(histories, csv_path)
    figures = plot_convergence(summary, args.output_dir)

    final_epoch = {
        DISPLAY_NAMES[model_type]: {
            "training_mean": float(summary[model_type]["train_mean"][-1]),
            "training_std": float(summary[model_type]["train_std"][-1]),
            "validation_mean": float(summary[model_type]["validation_mean"][-1]),
            "validation_std": float(summary[model_type]["validation_std"][-1]),
        }
        for model_type in MODEL_ORDER
    }
    transformer_validation = final_epoch["Transformer"]["validation_mean"]
    plga_validation = final_epoch["PLGAFormer"]["validation_mean"]
    comparison = {

        "plgaformer_vs_transformer_final_validation_reduction_percent": float(
            100.0 * (transformer_validation - plga_validation)
            / transformer_validation
        ),
    }
    manifest = {
        "artifact": "taes_epoch_convergence",
        "artifact_kind": "formal_v3_taes_epoch_convergence",
        "bundle_id": evidence_audit.get("bundle_id"),
        "config_sha256": evidence_audit.get("config_sha256"),
        "dataset_sha256": evidence_audit.get("dataset_sha256"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "training_performed": False,
        "plot_contract": {
            "core_conclusion": "PLGAFormer converges to the lowest shared ECEF validation objective under the common 50-epoch maximum.",
            "archetype": "two-panel quantitative grid",
            "statistics": "mean and sample standard deviation across seeds 42, 123, and 456",
            "error_bars": "sample standard deviation at epochs 1, 10, 25, and the last common completed epoch",
            "early_stopping": "curves end at the last epoch completed by all three seeds for each method; no values are imputed",
            "scale": "logarithmic y-axis",
            "review_boundary": "optimization diagnostic; not a replacement for held-out physical test metrics",
        },
        "evidence_audit": evidence_audit,
        "final_epoch_statistics": final_epoch,
        "comparison": comparison,
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "figures": figures,
        "figure_sha256": {
            Path(path).name: sha256_file(Path(path)) for path in figures
        },
    }
    manifest_path = args.output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "final_epoch_statistics": final_epoch,
                "comparison": comparison,
                "figures": figures,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
