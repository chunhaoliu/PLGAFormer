#!/usr/bin/env python3
"""Re-evaluate frozen final PLGAFormer checkpoints for physical ECEF metrics.

This script does not train or alter checkpoints. It supplements the validation-
selected phase-1 records with Cartesian MSE/MAE/RMSE under the exact formal
test protocol used by the TAES main experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.final_plgaformer import (
    FINAL_ARCHITECTURE,
    FINAL_SEEDS,
    MAIN_BUNDLE_PATH,
    final_plgaformer_kwargs,
    resolve_final_plgaformer,
)
from utils.formal_evidence import load_formal_config, validate_evidence_bundle


HORIZONS = [32, 64, 128, 256]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "exp2_ablation"
            / "results"
            / "formal_v3"
            / "hgv_multiregime_state_v2_1"
            / "final"
            / "final_plgaformer_physical_metrics.json"
        ),
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "formal_v3.json")
    parser.add_argument("--bundle", type=Path, default=MAIN_BUNDLE_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args(argv)


def _horizon_result(results: dict[str, Any], horizon: int) -> dict[str, Any]:
    return results.get(str(horizon), results.get(horizon, {}))


def evaluate(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    from utils.console import ensure_utf8_console

    ensure_utf8_console()
    config = load_formal_config(args.config)
    verification = validate_evidence_bundle(args.bundle, config, expected_kind="main")
    if not verification["passed"]:
        codes = sorted({str(item.get("code")) for item in verification.get("blockers", [])})
        raise RuntimeError("Formal-v3 Main Results bundle failed verification: " + ", ".join(codes))
    bundle = verification["bundle"]
    from experiments.exp2_ablation import ablation_study as exp2

    exp2.TRAIN_CONFIG["batch_size"] = int(args.batch_size)
    exp2.TRAIN_CONFIG["dataloader_workers"] = int(args.workers)
    exp2.TRAIN_CONFIG["pin_memory"] = bool(args.workers > 0)
    exp2.ABLATION_SUBSET_RATIO = 1.0
    exp2.ABLATION_PRED_LEN = 256
    exp2.PREDICTION_HORIZONS = list(HORIZONS)
    exp2.TRAIN_SUPERVISION_PROTOCOL = "source_context_pred_window"
    exp2.EVAL_PROTOCOL = "source_context_decoder"
    exp2.EVAL_AR_SEED_MODE = "zero"
    exp2.train_config["label_len"] = 128

    exp2.set_random_seed(FINAL_SEEDS[0])
    train_loader, val_loader, test_loader, x_scaler, y_scaler = (
        exp2.load_and_prepare_data(
            int(args.batch_size),
            subset_ratio=1.0,
            initial_load_seed=FINAL_SEEDS[0],
        )
    )
    del train_loader, val_loader
    if test_loader is None:
        raise RuntimeError("Formal test data could not be loaded.")

    device = exp2.TRAIN_CONFIG["device"]
    scaler_mean = torch.from_numpy(y_scaler.mean_.astype("float32")).to(device)
    scaler_scale = torch.from_numpy(y_scaler.scale_.astype("float32")).to(device)
    source_scaler_mean = np.concatenate(
        [y_scaler.mean_, x_scaler.mean_[3:]]
    ).astype("float32")
    source_scaler_scale = np.concatenate(
        [y_scaler.scale_, x_scaler.scale_[3:]]
    ).astype("float32")
    output_scaler_mean = y_scaler.mean_.astype("float32")
    output_scaler_scale = y_scaler.scale_.astype("float32")

    runs: dict[str, Any] = {}
    for seed in FINAL_SEEDS:
        checkpoint, source_payload, checkpoint_audit = resolve_final_plgaformer(seed, bundle_path=args.bundle)
        exp2.set_random_seed(seed)
        model = exp2.create_model(
            model_type="plgaformer",
            input_dim=6,
            device=device,
            innovations=final_plgaformer_kwargs(),
            model_config_override={"dropout": 0.1},
            input_scaler_mean=source_scaler_mean,
            input_scaler_scale=source_scaler_scale,
            output_scaler_mean=output_scaler_mean,
            output_scaler_scale=output_scaler_scale,
        )
        try:
            state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        eval_results = exp2.run_evaluation(
            model,
            test_loader,
            scaler_mean,
            scaler_scale,
            device,
            HORIZONS,
        )
        compact_results: dict[str, Any] = {}
        for horizon in HORIZONS:
            current = _horizon_result(eval_results, horizon)
            original = _horizon_result(source_payload["eval_results"], horizon)
            for metric in ("trajectory_window_ade", "trajectory_window_fde"):
                if not np.isclose(
                    float(current[metric]),
                    float(original[metric]),
                    rtol=1e-5,
                    atol=1e-3,
                ):
                    raise RuntimeError(
                        f"Frozen-checkpoint reproduction mismatch: seed={seed}, "
                        f"horizon={horizon}, metric={metric}, "
                        f"current={current[metric]}, original={original[metric]}."
                    )
            compact_results[str(horizon)] = {
                metric: float(current[metric])
                for metric in (
                    "mse_cart_m2",
                    "mae_cart_m",
                    "rmse_cart_m",
                    "trajectory_window_ade",
                    "trajectory_window_fde",
                )
            }
        runs[str(seed)] = {
            **checkpoint_audit,
            "source_run_id": source_payload["run_id"],
            "eval_results": compact_results,
        }
        print(
            f"[physical-metrics] seed={seed} "
            f"RMSE@256={compact_results['256']['rmse_cart_m']:.3f} m"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "artifact": "final_plgaformer_physical_metric_augmentation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle["bundle_id"],
        "config_sha256": bundle["config_sha256"],
        "dataset_sha256": bundle["dataset_sha256"],
        "dataset_protocol": bundle.get("dataset_protocol", config["dataset"]["protocol"]),
        "architecture": FINAL_ARCHITECTURE,
        "training_performed": False,
        "protocol": {
            "dataset_protocol": bundle.get("dataset_protocol", config["dataset"]["protocol"]),
            "config_sha256": bundle["config_sha256"],
            "dataset_sha256": bundle["dataset_sha256"],
            "bundle_id": bundle["bundle_id"],
            "subset_ratio": 1.0,
            "batch_size": int(args.batch_size),
            "prediction_horizons": HORIZONS,
            "train_supervision_protocol": "source_context_pred_window",
            "eval_protocol": "source_context_decoder",
            "eval_ar_seed_mode": "zero",
            "label_len": 128,
            "rmse_definition": "ECEF component RMSE over test windows, horizons, and xyz components",
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = evaluate(argv)
    print(
        json.dumps(
            {
                "output": str(parse_args(argv).output),
                "seeds": sorted(payload["runs"]),
                "training_performed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
