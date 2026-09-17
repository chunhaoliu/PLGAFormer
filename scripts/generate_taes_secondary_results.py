#!/usr/bin/env python3
"""Build strict TAES ablation, robustness, and efficiency tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.final_plgaformer import final_evidence_signature
from utils.formal_evidence import (
    load_evidence_bundle,
    load_formal_config,
    normalize_run_record,
    sha256_file,
    validate_evidence_bundle,
)

MANUSCRIPT_OUTPUT = PROJECT_ROOT / "experiments" / "taes_submission_artifacts" / "generated" / "formal_v3"
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
ABLATION_BUNDLE = (
    PROJECT_ROOT
    / "experiments"
    / "exp2_ablation"
    / "results"
    / "formal_v3"
    / "hgv_multiregime_state_v2_1"
    / "final"
    / "ablation_run_set_manifest.json"
)
ABLATION_SUMMARY = ABLATION_BUNDLE
ROBUSTNESS_RESULTS = None
EFFICIENCY_RESULTS = None
EXPECTED_SEEDS = [42, 123, 456]
CAPACITY_EVIDENCE = PROJECT_ROOT / "PublicRelease" / "evidence" / "capacity_control_256s.csv"
CAPACITY_MANIFEST = PROJECT_ROOT / "PublicRelease" / "evidence" / "evidence_manifest.json"

ABLATION_ROWS = [
    (
        "phase4_final_mechanism_controls",
        "spherical_prior",
        "Spherical prior + adaptive fusion",
    ),
    (
        "phase4_final_mechanism_controls",
        "schedule_only",
        "Rotating-Earth prior + fixed schedule",
    ),
    (
        "main_results_bundle",
        "full",
        "PLGAFormer (rotating-Earth prior + adaptive fusion)",
    ),
]

ROBUSTNESS_MODELS = ["Transformer (baseline)", "PLGAFormer (proposed)"]
PAPER_ROBUSTNESS_MODELS = ["transformer", "plgaformer"]
EFFICIENCY_MODELS = [
    "PLGAFormer",
    "Transformer",
    "iTransformer",
    "PatchTST",
    "DLinear",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "formal_v3.json")
    parser.add_argument("--main-bundle", type=Path, default=MAIN_BUNDLE)
    parser.add_argument("--ablation-bundle", type=Path, default=ABLATION_BUNDLE)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS_RESULTS)
    parser.add_argument("--efficiency", type=Path, default=EFFICIENCY_RESULTS)
    parser.add_argument("--registry", type=Path, default=None, help="Legacy diagnostic registry; not used by formal-v3 paper mode.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-existing-output", action="store_true", help="Allow shared formal-paper staging after the directory was initialized.")
    parser.add_argument("--skip-manifest", action="store_true", help="Leave the final artifact_manifest.json to the formal paper coordinator.")
    return parser.parse_args(argv)


def _metric_lookup(rows: list[dict]) -> dict[tuple[str, str, int, str], dict]:
    return {
        (
            str(row["phase"]),
            str(row["model_key"]),
            int(row["horizon"]),
            str(row["metric"]),
        ): row
        for row in rows
    }


def _bundle_ablation_lookup(bundle: dict) -> dict[tuple[str, str, int, str], dict]:
    if bundle.get("bundle_kind") != "ablation":
        raise RuntimeError("Expected an ablation evidence bundle.")
    if bundle.get("paper_eligible") is not True:
        codes = sorted({str(item.get("code")) for item in bundle.get("blockers", [])})
        raise RuntimeError("Formal-v3 ablation bundle is incomplete: " + ", ".join(codes))
    values: dict[tuple[str, str, str], list[float]] = {}
    for entry in bundle.get("source_records", []):
        normalized = normalize_run_record(entry["source_path"])
        model_key = str(normalized.get("model_key", ""))
        if model_key not in {"spherical_prior", "schedule_only"}:
            model_key = model_key if model_key in {"baseline", "full"} else ""
        if not model_key:
            continue
        phase = "phase4_final_mechanism_controls" if model_key in {"spherical_prior", "schedule_only"} else "main_results_bundle"
        item = normalized.get("eval_results", {}).get("256", normalized.get("eval_results", {}).get(256, {}))
        for metric, aliases in {
            "ade": ("ade", "trajectory_window_ade"),
            "fde": ("fde", "trajectory_window_fde"),
            "rmse_cart_m": ("rmse_cart_m",),
        }.items():
            value = next((item.get(alias) for alias in aliases if item.get(alias) is not None), None)
            if value is None:
                raise RuntimeError(f"Ablation source lacks {metric} at 256 s: {entry.get('source_path')}")
            values.setdefault((phase, model_key, metric), []).append(float(value))
    lookup = {}
    for phase, model_key, _ in ABLATION_ROWS:
        for metric in ("ade", "fde", "rmse_cart_m"):
            row_values = values.get((phase, model_key, metric), [])
            if len(row_values) != 3:
                raise RuntimeError(f"Formal-v3 ablation result is not three-seed complete: {phase}:{model_key}:{metric}")
            mean = sum(row_values) / len(row_values)
            variance = sum((value - mean) ** 2 for value in row_values) / (len(row_values) - 1)
            lookup[(phase, model_key, 256, metric)] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "n": len(row_values),
                "seeds": "42,123,456",
            }
    return lookup


def validate_ablation(payload: dict, signature: str) -> dict[tuple[str, str, int, str], dict]:
    """Validate the four-row paper contract from a formal-v3 ablation bundle."""
    if payload.get("bundle_kind") == "ablation":
        return _bundle_ablation_lookup(payload)
    raise RuntimeError(
        "Paper-facing ablation generation requires an ablation_run_set_manifest.json; "
        "historical summaries are diagnostic only."
    )


def load_capacity_control() -> dict[str, dict[str, float]]:
    manifest = json.loads(CAPACITY_MANIFEST.read_text(encoding="utf-8"))
    expected_hash = manifest.get("files", {}).get(CAPACITY_EVIDENCE.name)
    if expected_hash != sha256_file(CAPACITY_EVIDENCE):
        raise RuntimeError("Capacity-control evidence hash mismatch.")
    with CAPACITY_EVIDENCE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {
        str(row["metric"]): {
            "mean": float(row["mean_m"]),
            "std": float(row["sample_std_m"]),
        }
        for row in rows
        if int(row["horizon_s"]) == 256 and int(row["seed_count"]) == 3
    }
    if set(lookup) != {"ade", "fde", "rmse_cart_m"}:
        raise RuntimeError("Capacity-control evidence is not three-metric complete.")
    return lookup


def _ablation_pm_cell(mean_m: float, std_m: float, wrap: str | None = None) -> str:
    cell = (
        rf"\resultnum{{{float(mean_m) / 1000:.3f}}}$\pm$"
        rf"\resultnum{{{float(std_m) / 1000:.3f}}}"
    )
    if wrap == "bf":
        return rf"\textbf{{{cell}}}"
    if wrap == "ul":
        return rf"\underline{{{cell}}}"
    return cell


def _unique_rank_wraps(values: list[float]) -> list[str | None]:
    wraps: list[str | None] = [None] * len(values)
    if not values:
        return wraps
    best = min(values)
    best_idx = [idx for idx, value in enumerate(values) if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)]
    if len(best_idx) == 1:
        wraps[best_idx[0]] = "bf"
        remaining = [values[idx] for idx in range(len(values)) if idx not in best_idx]
        if remaining:
            second = min(remaining)
            second_idx = [
                idx
                for idx, value in enumerate(values)
                if idx not in best_idx and math.isclose(value, second, rel_tol=0.0, abs_tol=1e-12)
            ]
            if len(second_idx) == 1:
                wraps[second_idx[0]] = "ul"
    return wraps


def render_ablation_table(
    lookup: dict[tuple[str, str, int, str], dict],
    capacity_control: dict[str, dict[str, float]] | None = None,
) -> str:
    rows: list[tuple[str, str, dict[str, dict[str, float]]]] = []
    metric_keys = ("ade", "fde", "rmse_cart_m")
    if capacity_control is not None:
        rows.append(
            (
                "PLGAFormer learned-only backbone",
                "learned_only",
                {metric: capacity_control[metric] for metric in metric_keys},
            )
        )
    for phase, model_key, label in ABLATION_ROWS:
        rows.append(
            (
                label,
                model_key,
                {
                    metric: lookup[(phase, model_key, 256, metric)]
                    for metric in metric_keys
                },
            )
        )
    wraps = {
        metric: _unique_rank_wraps([float(stats[metric]["mean"]) for *_, stats in rows])
        for metric in metric_keys
    }

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Ablation of the analytical prior and fusion rule at a 256 s horizon on "
        r"the development test partition. Values are mean$\pm$sample standard deviation "
        r"across three seeds, in kilometers. Best results are bold; second-best results "
        r"are underlined. This table is not pooled with Table~\ref{tab:main_results}.}",
        r"\label{tab:ablation}",
        r"\fontsize{9}{11}\selectfont",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccc@{}}",
        r"\toprule",
        r"Variant & ADE (km) & FDE (km) & RMSE (km) \\",
        r"\midrule",
    ]
    previous_group = None
    for index, (label, model_key, stats) in enumerate(rows):
        group = "full" if model_key in {"full", "prior_only"} else model_key
        if previous_group == "learned_only" or (
            previous_group not in {None, "full"} and group == "full"
        ):
            lines.append(r"\midrule")
        display = rf"\textbf{{{label}}}" if model_key in {"full", "prior_only"} else label
        cells = [display] + [
            _ablation_pm_cell(stats[metric]["mean"], stats[metric]["std"], wraps[metric][index])
            for metric in metric_keys
        ]
        lines.append(" & ".join(cells) + r" \\")
        previous_group = group
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _nested(mapping: dict, key) -> dict:
    if key in mapping:
        return mapping[key]
    text = str(key)
    if text in mapping:
        return mapping[text]
    raise KeyError(key)


def validate_robustness(
    payload: dict,
    signature: str,
    base_signature: str,
    dataset_sha256: str | None = None,
    test_trajectory_count: int | None = None,
    config_sha256: str | None = None,
) -> dict:
    if payload.get("artifact_kind") == "formal_v3_taes_dynamics_shift":
        if config_sha256 and str(payload.get("config_sha256", "")).lower() != config_sha256.lower():
            raise RuntimeError("Robustness result lacks or uses another formal config hash.")
        if str(payload.get("base_exp1_signature", "")) != str(base_signature):
            raise RuntimeError("Robustness result uses another formal Main bundle.")
        if str(payload.get("run_signature", "")) != str(signature):
            raise RuntimeError("Robustness result uses another final-model signature.")
        if int(payload.get("seq_len", -1)) != 256 or int(payload.get("pred_len", -1)) != 256:
            raise RuntimeError("Robustness result is not the formal 256/256 protocol.")
        expected_count = int(test_trajectory_count or payload.get("test_trajectory_count", 0))
        if expected_count <= 0:
            raise RuntimeError("Robustness validation requires the formal test trajectory count.")
        expected_conditions = {"nominal", "aero_shift", "ballistic_shift"}
        results = payload.get("results", {})
        if set(results) != expected_conditions:
            raise RuntimeError("Robustness result lacks the configured dynamics-shift conditions.")
        expected_models = {"transformer", "plgaformer", "rotating_3dof"}
        if set(payload.get("model_types", [])) != expected_models:
            raise RuntimeError("Robustness result lacks the configured comparison models.")
        for condition in expected_conditions:
            if set(results[condition]) != expected_models:
                raise RuntimeError(f"Robustness result has an incomplete model matrix: {condition}.")
            for model_type, row in results[condition].items():
                expected_seeds = [42] if model_type == "rotating_3dof" else EXPECTED_SEEDS
                if row.get("model_seeds") != expected_seeds:
                    raise RuntimeError(
                        f"Robustness result lacks the configured model seeds: {model_type}, {condition}."
                    )
                if int(row.get("trajectory_count", 0)) != expected_count:
                    raise RuntimeError(
                        f"Robustness result does not use {expected_count} independent trajectories: "
                        f"{model_type}, {condition}."
                    )
        audits = payload.get("checkpoint_audits", {})
        for model_type in ("transformer", "plgaformer"):
            if set(audits.get(model_type, {})) != {str(seed) for seed in EXPECTED_SEEDS}:
                raise RuntimeError(f"Robustness checkpoint audit is incomplete: {model_type}.")
        return {
            "__schema__": "formal_v3_dynamics_shift",
            "conditions": sorted(expected_conditions),
            "model_types": ["transformer", "plgaformer", "rotating_3dof"],
            "results": results,
        }
    config = payload.get("config", {})
    observed_dataset_sha256 = str(
        payload.get("dataset_sha256")
        or payload.get("metadata", {}).get("dataset_sha256", "")
        or payload.get("dataset_audit", {}).get("observed_sha256", "")
    )
    if dataset_sha256 and observed_dataset_sha256.lower() != dataset_sha256.lower():
        raise RuntimeError("Robustness result lacks or uses another formal dataset hash.")
    if int(config.get("prediction_length", -1)) != 256 or int(config.get("seq_len", -1)) != 256:
        raise RuntimeError("Robustness result is not the formal 256/256 protocol.")
    expected_trajectory_count = test_trajectory_count
    if expected_trajectory_count is None:
        expected_trajectory_count = payload.get("test_trajectory_count")
    try:
        expected_trajectory_count = int(expected_trajectory_count)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Robustness validation requires the formal test trajectory count.") from exc
    if expected_trajectory_count <= 0:
        raise RuntimeError("Robustness validation requires a positive formal test trajectory count.")
    results = payload.get("results", {})
    conditions = (("noise", 0.0), ("noise", 1.0), ("missing", 0.10), ("input_length", 128))
    for model_name in ROBUSTNESS_MODELS:
        if model_name not in results:
            raise RuntimeError(f"Missing robustness model: {model_name}.")
        for condition, value in conditions:
            row = _nested(results[model_name][condition], value)
            if row.get("model_seeds") != EXPECTED_SEEDS:
                raise RuntimeError(
                    f"Robustness result lacks three model seeds: {model_name}, {condition}={value}."
                )
            if int(row.get("trajectory_count", 0)) != expected_trajectory_count:
                raise RuntimeError(
                    f"Robustness result does not use {expected_trajectory_count} independent trajectories: "
                    f"{model_name}, {condition}={value}."
                )
    audits = payload.get("checkpoint_audits", {})
    if config.get("run_signature") != signature:
        raise RuntimeError("Robustness result uses another final-model signature.")
    for model_name in ROBUSTNESS_MODELS:
        for seed in EXPECTED_SEEDS:
            audit = _nested(audits[model_name], seed)
            if "PLGAFormer" in model_name:
                if audit.get("architecture") is None:
                    raise RuntimeError(
                        f"Robustness final-model audit is missing: {model_name}, seed={seed}."
                    )
            elif audit.get("run_signature") != base_signature:
                raise RuntimeError(
                    f"Robustness checkpoint signature mismatch: {model_name}, seed={seed}."
                )
    return results


CONDITION_ORDER = ("nominal", "aero_shift", "ballistic_shift")
CONDITION_LABELS = {
    "nominal": "Nominal",
    "aero_shift": r"$C_L-10\%$, $C_D+10\%$",
    "ballistic_shift": r"$m+15\%$, $S-10\%$",
}


def _km_pair(mean_m: float, std_m: float, *, bold: bool = False) -> str:
    text = (
        rf"\resultnum{{{mean_m / 1000.0:.3f}}}$\pm$\resultnum{{{std_m / 1000.0:.3f}}}"
    )
    return rf"\textbf{{{text}}}" if bold else text


def render_robustness_table(results: dict) -> str:
    if results.get("__schema__") == "formal_v3_dynamics_shift":
        lines = [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\caption{Frozen-model errors under the specified dynamics shifts at 256 s. Values are mean$\pm$sample standard deviation across three seeds, in kilometers. Each condition uses one central window per trajectory; the nominal row is the matched-subset reference and is not the all-window mean in Table~\ref{tab:main_results}. Best results in each row are bold.}",
            r"\label{tab:robustness}",
            r"\fontsize{9}{11}\selectfont",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}",
            r"\toprule",
            r"Condition & \multicolumn{2}{c}{ADE (km)} & \multicolumn{2}{c}{FDE (km)} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r" & Transformer & PLGAFormer & Transformer & PLGAFormer \\",
            r"\midrule",
        ]
        for condition in CONDITION_ORDER:
            row_t = results["results"][condition]["transformer"]
            row_p = results["results"][condition]["plgaformer"]
            lines.append(
                f"{CONDITION_LABELS[condition]} & "
                f"{_km_pair(float(row_t['ade_m']), float(row_t['ade_std_m']))} & "
                f"{_km_pair(float(row_p['ade_m']), float(row_p['ade_std_m']), bold=True)} & "
                f"{_km_pair(float(row_t['fde_m']), float(row_t['fde_std_m']))} & "
                f"{_km_pair(float(row_p['fde_m']), float(row_p['fde_std_m']), bold=True)} "
                + r"\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
        return "\n".join(lines) + "\n"
    conditions = [
        ("noise", 0.0, "Clean"),
        ("noise", 1.0, "Nominal sensor noise"),
        ("missing", 0.10, "10\\% missing"),
        ("input_length", 128, "128 s context"),
    ]
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Test-time robustness at the 256 s forecast horizon. One central window "
        r"per held-out trajectory is used. Values pool three model seeds and, for stochastic "
        r"perturbations, three shared perturbation seeds (km).}",
        r"\label{tab:robustness}",
        r"\scriptsize",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Condition & Model & ADE & FDE \\",
        r"\midrule",
    ]
    for index, (condition, value, label) in enumerate(conditions):
        if index:
            lines.append(r"\addlinespace[1pt]")
        for model_name in ROBUSTNESS_MODELS:
            row = _nested(results[model_name][condition], value)
            display = "PLGAFormer" if "PLGAFormer" in model_name else "Transformer"
            if display == "PLGAFormer":
                display = r"\textbf{PLGAFormer}"
            lines.append(
                f"{label} & {display} & "
                f"{float(row['ade_m']) / 1000:.3f}$\\pm${float(row['ade_m_std']) / 1000:.3f} & "
                f"{float(row['fde_m']) / 1000:.3f}$\\pm${float(row['fde_m_std']) / 1000:.3f} "
                + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def validate_efficiency(
    payload: dict,
    signature: str,
    base_signature: str | None = None,
    dataset_sha256: str | None = None,
) -> list[dict]:
    metadata = payload.get("metadata", {})
    if not metadata.get("formal"):
        raise RuntimeError("Efficiency result is not a formal timing run.")
    reference_audit = metadata.get("performance_reference_audit", {})
    compatible_signature = reference_audit.get("run_signature") == signature
    compatible_bundle = base_signature and reference_audit.get("bundle_id") == base_signature
    if not (compatible_signature or compatible_bundle):
        raise RuntimeError("Efficiency performance reference uses another formal evidence bundle.")
    observed_dataset_sha256 = str(
        metadata.get("dataset_sha256")
        or reference_audit.get("dataset_sha256", "")
    )
    if dataset_sha256 and observed_dataset_sha256.lower() != dataset_sha256.lower():
        raise RuntimeError("Efficiency result lacks or uses another formal dataset hash.")
    rows = payload.get("results", [])
    names = {row.get("model") for row in rows}
    missing = [name for name in EFFICIENCY_MODELS if name not in names]
    if missing:
        raise RuntimeError(f"Efficiency result lacks models: {missing}.")
    for row in rows:
        for key in (
            "params",
            "flops_mflops",
            "peak_inference_memory_mb",
            "latency_batch1_ms",
            "throughput_trajectories_s",
            "ADE_256_m",
            "FDE_256_m",
            "RMSE_256_m",
        ):
            if row.get(key) is None or not math.isfinite(float(row[key])):
                raise RuntimeError(f"Invalid efficiency field {key} for {row.get('model')}.")
    return rows


def render_efficiency_table(rows: list[dict], metadata: dict) -> str:
    lookup = {row["model"]: row for row in rows}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Measured inference cost on "
        + str(metadata.get("device_name", "the evaluation device")).replace("_", r"\_")
        + r". Batch-one latency includes the complete 256 s forecast.}",
        r"\label{tab:efficiency}",
        r"\scriptsize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Params (M) & MFLOPs & Latency (ms) \\",
        r"\midrule",
    ]
    for name in EFFICIENCY_MODELS:
        row = lookup[name]
        display = rf"\textbf{{{name}}}" if name == "PLGAFormer" else name
        params = float(row["params"]) / 1e6
        lines.append(
            f"{display} & {params:.3f} & {float(row['flops_mflops']):.1f} & "
            f"{float(row['latency_batch1_ms']):.2f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_artifacts(args: argparse.Namespace) -> dict:
    config = load_formal_config(args.config)
    main_verification = validate_evidence_bundle(
        args.main_bundle, config, expected_kind="main"
    )
    main_bundle = main_verification["bundle"]
    if not main_verification["passed"]:
        codes = sorted({str(item.get("code")) for item in main_verification.get("blockers", [])})
        raise RuntimeError("Formal-v3 Main Results bundle failed verification: " + ", ".join(codes))
    ablation_verification = validate_evidence_bundle(
        args.ablation_bundle,
        config,
        expected_kind="ablation",
        main_bundle=main_bundle,
    )
    ablation_bundle = ablation_verification["bundle"]
    if not ablation_verification["passed"]:
        codes = sorted({str(item.get("code")) for item in ablation_verification.get("blockers", [])})
        raise RuntimeError("Formal-v3 Ablation bundle failed verification: " + ", ".join(codes))
    base_signature = str(main_bundle["bundle_id"])
    signature, signature_audit = final_evidence_signature(
        base_signature,
        bundle_path=args.main_bundle,
    )
    ablation_lookup = validate_ablation(ablation_bundle, base_signature)

    optional_sections: dict[str, Any] = {}
    omitted_sections: list[dict[str, str]] = []
    robustness_results = None
    efficiency_rows = None
    efficiency_payload = None
    if args.robustness is not None:
        robustness_payload = json.loads(args.robustness.read_text(encoding="utf-8"))
        robustness_results = validate_robustness(
            robustness_payload,
            signature,
            base_signature,
            str(main_bundle.get("dataset_sha256", "")),
            int(config["split"]["complete_trajectory_counts"]["test"]),
            config_sha256=config.config_sha256,
        )
        optional_sections["robustness"] = str(args.robustness.resolve())
    else:
        omitted_sections.append({"section": "robustness", "reason": "optional protocol-matched result was not supplied"})
    if args.efficiency is not None:
        efficiency_payload = json.loads(args.efficiency.read_text(encoding="utf-8"))
        efficiency_rows = validate_efficiency(
            efficiency_payload,
            signature,
            base_signature,
            str(main_bundle.get("dataset_sha256", "")),
        )
        optional_sections["efficiency"] = str(args.efficiency.resolve())
    else:
        omitted_sections.append({"section": "efficiency", "reason": "optional protocol-matched result was not supplied"})

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = MANUSCRIPT_OUTPUT / str(main_bundle["bundle_id"])
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(f"Paper staging directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = output_dir / "table_ablation.tex"
    capacity_control = load_capacity_control()
    ablation_path.write_text(
        render_ablation_table(ablation_lookup, capacity_control), encoding="utf-8"
    )
    outputs = [ablation_path]
    secondary_path = None
    if robustness_results is not None or efficiency_rows is not None:
        pieces = []
        if robustness_results is not None:
            pieces.append(render_robustness_table(robustness_results))
        if efficiency_rows is not None and efficiency_payload is not None:
            pieces.append(render_efficiency_table(efficiency_rows, efficiency_payload["metadata"]))
        secondary_path = output_dir / "table_robustness_efficiency.tex"
        secondary_path.write_text("\n\n".join(pieces), encoding="utf-8")
        outputs.append(secondary_path)
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_kind": "formal_v3_taes_secondary_results",
        "main_bundle_id": main_bundle["bundle_id"],
        "ablation_bundle_id": ablation_bundle["bundle_id"],
        "config_sha256": main_bundle.get("config_sha256"),
        "dataset_sha256": main_bundle.get("dataset_sha256"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "generator_source": str(Path(__file__).resolve()),
        "generator_source_sha256": sha256_file(Path(__file__).resolve()),
        "run_signature": signature,
        "base_exp1_signature": base_signature,
        "final_signature_audit": signature_audit,
        "inputs": {
            "main_bundle": str(args.main_bundle.resolve()),
            "ablation_bundle": str(args.ablation_bundle.resolve()),
            "capacity_control": str(CAPACITY_EVIDENCE.resolve()),
            **optional_sections,
        },
        "omitted_sections": omitted_sections,
        "outputs": {
            str(path.name): {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in outputs
        },
        "claim_boundary": "matched formal-v3 mechanism evidence on complete simulated HGV trajectories",
    }
    if not args.skip_manifest:
        temp = artifact_manifest_path.with_name(artifact_manifest_path.name + ".tmp")
        temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(artifact_manifest_path)
    manifest["artifact_manifest"] = None if args.skip_manifest else str(artifact_manifest_path)
    manifest["output_files"] = [str(path) for path in outputs]
    manifest["test_trajectory_count"] = int(config["split"]["complete_trajectory_counts"]["test"])
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(write_artifacts(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
