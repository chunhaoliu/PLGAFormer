#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal coordinator for the HGV formal-v3 evidence command family."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "formal_v3.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.formal_evidence import (
    audit_dataset,
    build_evidence_bundle,
    discover_run_records,
    get_formal_study,
    load_formal_config,
    resolve_config_path,
    resolve_study_artifact_root,
    sha256_file,
    validate_ablation_completeness,
    validate_evidence_bundle,
    validate_main_completeness,
)


class FormalPipelineError(RuntimeError):
    """A user-actionable evidence-gate failure."""

    def __init__(self, message: str, *, blockers: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.blockers = blockers or []


def _config_from_args(args: argparse.Namespace):
    return load_formal_config(Path(args.config).expanduser())


def _records_and_audits(config):
    main_root = resolve_config_path(config, "main_results_records")
    ablation_root = resolve_config_path(config, "ablation_records")
    main_records = discover_run_records(main_root, "main")
    ablation_records = discover_run_records(ablation_root, "ablation")
    main_audit = validate_main_completeness(main_records, config)
    ablation_audit = validate_ablation_completeness(ablation_records, main_audit, config)
    return main_records, ablation_records, main_audit, ablation_audit


def build_status(config) -> dict[str, Any]:
    main_records, ablation_records, main_audit, ablation_audit = _records_and_audits(config)
    dataset_audit = audit_dataset(config)
    report = {
        "command": "formal status",
        "config_path": str(config.path),
        "config_sha256": config.config_sha256,
        "dataset": dataset_audit,
        "main_results": main_audit,
        "ablation": ablation_audit,
        "paper_eligible": bool(main_audit["paper_eligible"] and ablation_audit["paper_eligible"]),
        "record_counts": {
            "main": len(main_records),
            "ablation": len(ablation_records),
        },
        "claim_boundary": "complete simulated HGV trajectories; no flight, radar, or deployment validation",
    }
    return report


def _print_report(report: dict[str, Any], as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"formal protocol: {report['dataset']['protocol']}")
    print(f"config sha256: {report['config_sha256']}")
    print(f"dataset: {'pass' if report['dataset']['passed'] else 'fail'}")
    for name in ("main_results", "ablation"):
        audit = report[name]
        print(f"{name}: {'paper-eligible' if audit['paper_eligible'] else 'blocked'}")
        for blocker in audit.get("blockers", []):
            print(f"  [{blocker['code']}] {blocker['message']}")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Frozen formal-v3 configuration (default: {DEFAULT_CONFIG}).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python run.py formal",
        description=(
            "Formal-v3 evidence coordinator for physics-aware long-horizon "
            "hypersonic glide vehicle trajectory prediction."
        ),
    )
    parser.set_defaults(config=DEFAULT_CONFIG)
    _add_config_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Read-only protocol and completeness status.")
    _add_config_argument(status)
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    validate_data = subparsers.add_parser("validate-data", help="Run the strict active-dataset validation path.")
    _add_config_argument(validate_data)
    validate_data.add_argument("--json", action="store_true", help="Emit the validation report as JSON.")

    def add_runner_command(name: str, help_text: str, runner_help: str) -> argparse.ArgumentParser:
        command_parser = subparsers.add_parser(name, help=help_text)
        _add_config_argument(command_parser)
        command_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Describe the formal route without importing a training/evaluation runner.",
        )
        command_parser.add_argument(
            "runner_args",
            nargs=argparse.REMAINDER,
            help=runner_help,
        )
        return command_parser

    add_runner_command(
        "main",
        "Run the registered Overall Prediction formal Main Results study.",
        "Arguments forwarded to run_formal_sota_unit.py.",
    )
    add_runner_command(
        "sota",
        "Explicit compatibility alias for formal main.",
        "Arguments forwarded to run_formal_sota_unit.py.",
    )
    add_runner_command(
        "ablation",
        "Explicit compatibility alias for the phase4 mechanism-control study.",
        "Arguments forwarded to run_formal_ablation_unit.py.",
    )
    add_runner_command(
        "mechanism",
        "Run the registered Mechanism Ablation and Physical Consistency study.",
        "Arguments forwarded to run_formal_ablation_unit.py.",
    )

    robustness = subparsers.add_parser(
        "robustness",
        help="Run the registered Generalization and Robustness study from a Main bundle.",
    )
    _add_config_argument(robustness)
    robustness.add_argument("--main-bundle", type=Path, default=None)
    robustness.add_argument("--output-dir", type=Path, default=None)
    robustness.add_argument("--dry-run", action="store_true")
    robustness.add_argument("runner_args", nargs=argparse.REMAINDER)

    efficiency = subparsers.add_parser(
        "efficiency",
        help="Run the registered Efficiency and Computational Cost study from a Main bundle.",
    )
    _add_config_argument(efficiency)
    efficiency.add_argument("--main-bundle", type=Path, default=None)
    efficiency.add_argument("--output-dir", type=Path, default=None)
    efficiency.add_argument("--dry-run", action="store_true")
    efficiency.add_argument("runner_args", nargs=argparse.REMAINDER)

    aggregate = subparsers.add_parser("aggregate", help="Validate records and write formal-v3 bundle manifests.")
    _add_config_argument(aggregate)
    aggregate.add_argument("--kind", choices=("main", "ablation", "all"), default="all")
    aggregate.add_argument("--force", action="store_true", help="Replace an existing manifest explicitly.")
    aggregate.add_argument("--dry-run", action="store_true", help="Audit eligibility without writing manifests or summaries.")

    paper = subparsers.add_parser("paper", help="Generate staged paper artifacts from eligible bundles only.")
    _add_config_argument(paper)
    paper.add_argument("--robustness", type=Path, default=None)
    paper.add_argument("--efficiency", type=Path, default=None)
    paper.add_argument("--main-bundle", type=Path, default=None)
    paper.add_argument("--ablation-bundle", type=Path, default=None)
    paper.add_argument("--output-dir", type=Path, default=None)
    paper.add_argument("--permutations", type=int, default=20_000)
    paper.add_argument("--bootstrap-resamples", type=int, default=10_000)
    paper.add_argument("--dry-run", action="store_true", help="Check eligibility without creating staged artifacts.")
    paper.add_argument("--json", action="store_true", help="Emit a machine-readable blocker or success report.")

    audit = subparsers.add_parser("audit", help="Audit the full data-to-evidence chain; nonzero on blockers.")
    _add_config_argument(audit)
    audit.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    audit.add_argument("--dry-run", action="store_true", help="Keep the audit explicitly read-only.")
    return parser


def _runner_args(args: argparse.Namespace) -> list[str]:
    values = list(getattr(args, "runner_args", []) or [])
    if values and values[0] == "--":
        values = values[1:]
    return values


def _run_main(args: argparse.Namespace, config) -> int:
    from scripts.run_formal_sota_unit import main as run_main

    forwarded = _runner_args(args)
    if getattr(args, "dry_run", False):
        print(json.dumps(_study_dry_run_report(config, "main"), indent=2, ensure_ascii=False))
        return 0
    if "--config" not in forwarded:
        forwarded = ["--config", str(config.path), *forwarded]
    return int(run_main(forwarded))


def _run_ablation(args: argparse.Namespace, config, *, study_name: str = "mechanism") -> int:
    from scripts.run_formal_ablation_unit import main as run_ablation

    forwarded = _runner_args(args)
    if getattr(args, "dry_run", False):
        return _run_bundle_study_dry_run(config, study_name)
    if "--config" not in forwarded:
        forwarded = ["--config", str(config.path), *forwarded]
    if "--phase" not in forwarded:
        forwarded = ["--phase", str(get_formal_study(config, "mechanism")["phase"]), *forwarded]
    return int(run_ablation(forwarded))


def _default_main_bundle_path(config) -> Path:
    return resolve_study_artifact_root(config, "main") / "main_run_set_manifest.json"


def _verified_main_bundle(config, requested: Path | None = None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = (requested or _default_main_bundle_path(config)).expanduser().resolve()
    if not path.is_file():
        return None, [{"code": "MAIN_BUNDLE_MISSING", "message": f"Main bundle is missing: {path}"}]
    verification = validate_evidence_bundle(path, config, expected_kind="main")
    return verification.get("bundle"), list(verification.get("blockers", []))


def _study_dry_run_report(config, study_name: str) -> dict[str, Any]:
    study = get_formal_study(config, study_name)
    return {
        "command": f"formal {study_name} --dry-run",
        "study_id": study["study_id"],
        "runner": study["runner"],
        "artifact_root": str(resolve_study_artifact_root(config, study_name)),
        "input_bundle": study.get("input_bundle"),
        "retrain": bool(study.get("retrain", False)),
        "paper_facing": bool(study.get("paper_facing", False)),
        "config_sha256": config.config_sha256,
        "dataset_protocol": config["dataset"]["protocol"],
        "dataset_sha256": config["dataset"]["sha256"],
    }


def _run_bundle_study_dry_run(config, study_name: str, requested: Path | None = None) -> int:
    bundle, blockers = _verified_main_bundle(config, requested)
    report = _study_dry_run_report(config, study_name)
    report["paper_eligible_input_bundle"] = bool(bundle and not blockers)
    report["blockers"] = blockers
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not blockers else 1


def _run_robustness(args: argparse.Namespace, config) -> int:
    bundle_path = (args.main_bundle or _default_main_bundle_path(config)).expanduser().resolve()
    if args.dry_run:
        return _run_bundle_study_dry_run(config, "robustness", bundle_path)
    bundle, blockers = _verified_main_bundle(config, bundle_path)
    if blockers or bundle is None:
        raise FormalPipelineError("Robustness requires an eligible formal Main bundle.", blockers=blockers)
    from scripts import run_taes_dynamics_shift

    forwarded = ["--config", str(config.path), "--registry", str(bundle_path)]
    results_dir = args.output_dir or resolve_study_artifact_root(config, "robustness")
    forwarded.extend(["--results-dir", str(results_dir.expanduser().resolve())])
    forwarded.extend(_runner_args(args))
    return int(run_taes_dynamics_shift.main(forwarded))


def _run_efficiency(args: argparse.Namespace, config) -> int:
    bundle_path = (args.main_bundle or _default_main_bundle_path(config)).expanduser().resolve()
    if args.dry_run:
        return _run_bundle_study_dry_run(config, "efficiency", bundle_path)
    bundle, blockers = _verified_main_bundle(config, bundle_path)
    if blockers or bundle is None:
        raise FormalPipelineError("Efficiency requires an eligible formal Main bundle.", blockers=blockers)
    from experiments.exp6_efficiency import efficiency_experiment

    forwarded = [
        "--config", str(config.path),
        "--performance-results", str(bundle_path),
    ]
    output_dir = args.output_dir or resolve_study_artifact_root(config, "efficiency")
    forwarded.extend(["--output-dir", str(output_dir.expanduser().resolve())])
    forwarded.extend(_runner_args(args))
    return 0 if efficiency_experiment.main(forwarded) is not None else 1


def _write_summary(path: Path, audit: dict[str, Any], manifest: dict[str, Any]) -> None:
    payload = {
        "bundle_id": manifest["bundle_id"],
        "manifest_path": manifest["manifest_path"],
        "paper_eligible": manifest["paper_eligible"],
        "config_sha256": manifest.get("config_sha256"),
        "record_count_discovered": manifest["record_count_discovered"],
        "matrix": audit.get("matrix", {}),
        "blockers": audit.get("blockers", []),
        "claim_boundary": audit.get("claim_boundary"),
    }
    target = path.with_name("main_results_summary.json" if audit.get("kind") == "main" else "formal_ablation_summary.json")
    temp = target.with_name(target.name + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)


def _aggregate(config, kind: str, force: bool) -> dict[str, Any]:
    main_records, ablation_records, main_audit, _ = _records_and_audits(config)
    output: dict[str, Any] = {"config_sha256": config.config_sha256, "bundles": []}
    main_basis: dict[str, Any] = main_audit
    if kind in {"main", "all"}:
        manifest_path = resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json"
        manifest = build_evidence_bundle(main_records, main_audit, manifest_path, config=config, force=force)
        _write_summary(manifest_path, main_audit, manifest)
        output["bundles"].append(manifest)
        main_basis = manifest
    if kind in {"ablation", "all"}:
        if kind == "ablation":
            main_path = resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json"
            main_basis, main_blockers = _bundle_or_audit(
                main_path, main_audit, config, expected_kind="main"
            )
            if main_blockers:
                raise FormalPipelineError(
                    "Ablation aggregation requires a valid Main Results bundle.",
                    blockers=main_blockers,
                )
        ablation_audit = validate_ablation_completeness(
            ablation_records, main_basis, config
        )
        manifest_path = resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json"
        manifest = build_evidence_bundle(ablation_records, ablation_audit, manifest_path, config=config, force=force)
        _write_summary(manifest_path, ablation_audit, manifest)
        output["bundles"].append(manifest)
    return output


def _bundle_or_audit(
    path: Path,
    audit: dict[str, Any],
    config,
    *,
    expected_kind: str,
    main_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        return audit, list(audit.get("blockers", []))
    verification = validate_evidence_bundle(
        path,
        config,
        expected_kind=expected_kind,
        main_bundle=main_bundle,
    )
    bundle = verification["bundle"]
    return bundle, list(verification.get("blockers", []))


def _write_unified_artifact_manifest(
    output_dir: Path,
    *,
    artifact_id: str,
    config,
    main_bundle: dict[str, Any],
    ablation_bundle: dict[str, Any],
    optional_evidence: dict[str, Any],
    main_result: dict[str, Any],
    secondary_result: dict[str, Any],
) -> Path:
    required = {
        "table_main_results.tex",
        "table_significance.tex",
        "table_maneuver_results.tex",
        "main_results_summary.json",
        "table_ablation.tex",
    }
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    names = {path.name for path in files}
    missing = sorted(required - names)
    if missing:
        raise FormalPipelineError(
            "Formal paper staging is incomplete; required files are missing: " + ", ".join(missing)
        )
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = {
        "schema_version": 2,
        "artifact_kind": "formal_v3_taes_complete_bundle",
        "bundle_id": artifact_id,
        "main_bundle_id": main_bundle["bundle_id"],
        "ablation_bundle_id": ablation_bundle["bundle_id"],
        "optional_evidence": optional_evidence,
        "config_sha256": config.config_sha256,
        "dataset_sha256": config["dataset"]["sha256"],
        "dataset_protocol": config["dataset"]["protocol"],
        "test_trajectory_count": int(config["split"]["complete_trajectory_counts"]["test"]),
        "study_registry": {
            name: {
                "study_id": config["studies"][name]["study_id"],
                "runner": config["studies"][name]["runner"],
                "input_bundle": config["studies"][name].get("input_bundle"),
            }
            for name in ("main", "mechanism", "robustness", "efficiency")
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generators": {
            "main_results": main_result,
            "secondary_results": secondary_result,
        },
        "omitted_sections": list(secondary_result.get("omitted_sections", [])),
        "files": {
            path.name: {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path)}
            for path in files
        },
        "claim_boundary": "complete simulated HGV trajectories; simulation-only evidence",
    }
    temp = manifest_path.with_name(manifest_path.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, manifest_path)
    return manifest_path


def _paper(config, args: argparse.Namespace) -> int:
    _, _, main_audit, ablation_audit = _records_and_audits(config)
    dataset_audit = audit_dataset(config)
    main_path = args.main_bundle or (resolve_config_path(config, "main_results_records") / "main_run_set_manifest.json")
    ablation_path = args.ablation_bundle or (resolve_config_path(config, "ablation_records") / "ablation_run_set_manifest.json")
    main_bundle, main_bundle_blockers = _bundle_or_audit(
        main_path, main_audit, config, expected_kind="main"
    )
    ablation_bundle, ablation_bundle_blockers = _bundle_or_audit(
        ablation_path,
        ablation_audit,
        config,
        expected_kind="ablation",
        main_bundle=main_bundle if main_path.is_file() else None,
    )
    blockers: list[dict[str, Any]] = []
    if not dataset_audit.get("passed"):
        blockers.append({"code": "DATASET_AUDIT_FAILED", "message": "Frozen dataset hash validation failed.", "context": dataset_audit})
    blockers.extend(main_bundle_blockers)
    blockers.extend(ablation_bundle_blockers)
    if main_bundle.get("paper_eligible") is not True:
        blockers.extend(main_bundle.get("blockers", []))
    if ablation_bundle.get("paper_eligible") is not True:
        blockers.extend(ablation_bundle.get("blockers", []))
    unique_blockers: list[dict[str, Any]] = []
    seen_blockers: set[str] = set()
    for blocker in blockers:
        fingerprint = json.dumps(blocker, sort_keys=True, ensure_ascii=False)
        if fingerprint not in seen_blockers:
            seen_blockers.add(fingerprint)
            unique_blockers.append(blocker)
    blockers = unique_blockers
    if blockers:
        raise FormalPipelineError(
            "Formal paper artifacts are blocked until Main Results and matched ablation bundles are complete.",
            blockers=blockers,
        )
    optional_evidence: dict[str, Any] = {}
    for label, value in (("robustness", args.robustness), ("efficiency", args.efficiency)):
        if value is None:
            continue
        source = value.expanduser().resolve()
        if not source.is_file():
            raise FormalPipelineError(f"Optional {label} evidence is missing: {source}")
        optional_evidence[label] = {"path": str(source), "sha256": sha256_file(source)}
    from scripts import generate_taes_main_results as main_generator
    from scripts import generate_taes_secondary_results as secondary_generator
    generator_sources = {
        "main_results": sha256_file(main_generator.__file__),
        "secondary_results": sha256_file(secondary_generator.__file__),
    }


    identity_payload = {
        "config_sha256": config.config_sha256,
        "main_bundle_id": main_bundle["bundle_id"],
        "ablation_bundle_id": ablation_bundle["bundle_id"],
        "optional_evidence_sha256": {
            label: item["sha256"]
            for label, item in sorted(optional_evidence.items())
        },
        "generator_sources": generator_sources,
        "permutations": int(args.permutations),
        "bootstrap_resamples": int(args.bootstrap_resamples),
    }
    from utils.formal_evidence import canonical_json, sha256_bytes
    artifact_id = sha256_bytes(canonical_json(identity_payload).encode("utf-8"))
    if args.dry_run:
        print(json.dumps({
            "paper_eligible": True,
            "artifact_id": artifact_id,
            "main_bundle": str(main_path),
            "ablation_bundle": str(ablation_path),
            "optional_evidence": optional_evidence,
        }, indent=2))
        return 0
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = resolve_config_path(config, "generated_root") / artifact_id
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FormalPipelineError(f"Paper staging destination already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{artifact_id[:12]}-", dir=output_dir.parent))

    try:
        main_args = [
            "--config", str(config.path),
            "--input", str(main_path),
            "--output-dir", str(staging_dir),
            "--permutations", str(args.permutations),
            "--bootstrap-resamples", str(args.bootstrap_resamples),
            "--skip-manifest",
        ]
        main_return = int(main_generator.main(main_args))
        if main_return != 0:
            raise FormalPipelineError("Main Results generator did not complete successfully.")
        secondary_args = [
            "--config", str(config.path),
            "--main-bundle", str(main_path),
            "--ablation-bundle", str(ablation_path),
            "--output-dir", str(staging_dir),
            "--allow-existing-output",
            "--skip-manifest",
        ]
        for label, value in (("robustness", args.robustness), ("efficiency", args.efficiency)):
            if value is not None:
                secondary_args.extend([f"--{label}", str(value.expanduser().resolve())])
        secondary_return = int(secondary_generator.main(secondary_args))
        if secondary_return != 0:
            raise FormalPipelineError("Secondary artifact generator did not complete successfully.")
        omitted_sections = [
            {"section": label, "reason": "optional protocol-matched result was not supplied"}
            for label in ("robustness", "efficiency")
            if label not in optional_evidence
        ]
        main_result = {
            "returncode": main_return,
            "source_path": str(Path(main_generator.__file__).resolve()),
            "source_sha256": sha256_file(main_generator.__file__),
        }
        secondary_result = {
            "returncode": secondary_return,
            "source_path": str(Path(secondary_generator.__file__).resolve()),
            "source_sha256": sha256_file(secondary_generator.__file__),
            "omitted_sections": omitted_sections,
        }
        _write_unified_artifact_manifest(
            staging_dir,
            artifact_id=artifact_id,
            config=config,
            main_bundle=main_bundle,
            ablation_bundle=ablation_bundle,
            optional_evidence=optional_evidence,
            main_result=main_result,
            secondary_result=secondary_result,
        )
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    manifest_path = output_dir / "artifact_manifest.json"
    print(json.dumps({
        "paper_eligible": True,
        "artifact_id": artifact_id,
        "main_bundle": str(main_path),
        "ablation_bundle": str(ablation_path),
        "staging": str(output_dir),
        "artifact_manifest": str(manifest_path),
    }, indent=2))
    return 0


def _validate_data(config, as_json: bool) -> int:
    from data_provider.validation import validate_hgv_dataset
    from scripts.validate_multiregime_dataset import main as validate_multiregime

    report = validate_hgv_dataset(require_trajectory_level=True)
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("passed"):
        return 1
    dataset_path = resolve_config_path(config, "main_results_records")
    del dataset_path
    return int(
        validate_multiregime(
            [
                "--dataset",
                str(config.resolve(config["dataset"]["relative_path"])),
                "--no-write",
                "--require-formal-count",
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed, unknown = parser.parse_known_args(argv)
    if unknown and parsed.command not in {"main", "sota", "ablation", "mechanism"}:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    config = _config_from_args(parsed)
    if parsed.command == "status":
        _print_report(build_status(config), as_json=bool(parsed.json))
        return 0
    if parsed.command == "validate-data":
        return _validate_data(config, bool(parsed.json))
    if parsed.command in {"main", "sota"}:
        if unknown:
            parsed.runner_args.extend(unknown)
        return _run_main(parsed, config)
    if parsed.command in {"ablation", "mechanism"}:
        if unknown:
            parsed.runner_args.extend(unknown)
        return _run_ablation(parsed, config, study_name="mechanism")
    if parsed.command == "robustness":
        try:
            return _run_robustness(parsed, config)
        except FormalPipelineError as exc:
            print(json.dumps({"paper_eligible": False, "error": str(exc), "blockers": exc.blockers}, indent=2, ensure_ascii=False))
            return 1
    if parsed.command == "efficiency":
        try:
            return _run_efficiency(parsed, config)
        except FormalPipelineError as exc:
            print(json.dumps({"paper_eligible": False, "error": str(exc), "blockers": exc.blockers}, indent=2, ensure_ascii=False))
            return 1
    if parsed.command == "aggregate":
        if parsed.dry_run:
            report = build_status(config)
            report["command"] = "formal aggregate --dry-run"
            _print_report(report, as_json=True)
            return 0
        print(json.dumps(_aggregate(config, parsed.kind, parsed.force), indent=2, ensure_ascii=False))
        return 0
    if parsed.command == "audit":
        report = build_status(config)
        _print_report(report, as_json=bool(parsed.json))
        return 0 if report["paper_eligible"] and report["dataset"]["passed"] else 1
    if parsed.command == "paper":
        try:
            return _paper(config, parsed)
        except FormalPipelineError as exc:
            report = {
                "paper_eligible": False,
                "error": str(exc),
                "blockers": exc.blockers,
            }
            if getattr(parsed, "json", False):
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(f"formal paper artifacts blocked: {exc}")
                for blocker in exc.blockers:
                    print(f"  [{blocker.get('code', 'BLOCKED')}] {blocker.get('message', blocker)}")
            return 1
        except (OSError, ValueError, RuntimeError) as exc:
            report = {
                "paper_eligible": False,
                "error": str(exc),
                "blockers": [],
            }
            if getattr(parsed, "json", False):
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                print(f"formal paper artifacts blocked: {exc}")
            return 1
    parser.error(f"unsupported formal command: {parsed.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
