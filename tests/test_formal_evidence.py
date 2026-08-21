import json
from pathlib import Path

import pytest

from utils.formal_evidence import (
    FINAL_MODEL_FLAGS,
    REQUIRED_HORIZONS,
    build_evidence_bundle,
    discover_run_records,
    canonical_json,
    load_formal_config,
    normalize_run_record,
    resolve_bundle_checkpoint,
    sha256_bytes,
    resolve_config_path,
    validate_ablation_completeness,
    validate_evidence_bundle,
    validate_main_completeness,
    validate_protocol_core,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "formal_v3.json"


def _payload(tmp_path: Path, config, *, model_key="baseline", seed=42, metrics=None, af_root=None):
    checkpoint = tmp_path / f"{model_key}_{seed}.pth"
    checkpoint.write_bytes(f"{model_key}:{seed}".encode("utf-8"))
    core = {
        "dataset_protocol": config["dataset"]["protocol"],
        "dataset_sha256": config["dataset"]["sha256"],
        "sampling_interval_s": config["dataset"]["sampling_interval_s"],
        "subset_ratio": 1.0,
        "epochs": config["training"]["epochs_max"],
        "batch_size": config["training"]["batch_size"],
        "prediction_length": config["task"]["pred_len"],
        "prediction_horizons": list(REQUIRED_HORIZONS),
        "train_supervision_protocol": config["task"]["train_supervision_protocol"],
        "eval_protocol": config["task"]["eval_protocol"],
        "eval_ar_seed_mode": config["task"]["eval_ar_seed_mode"],
        "label_len": config["task"]["label_len"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_epochs": config["training"]["warmup_epochs"],
        "patience": config["training"]["patience"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        "mixed_precision": config["training"]["amp"],
        "mixed_precision_dtype": config["training"]["precision"],
        "cache_physics_prior": config["training"]["cache_physics_prior"],
        "physics_prior_cache_batch_size": config["training"]["physics_prior_cache_batch_size"],
    }
    test_count = int(config["split"]["complete_trajectory_counts"]["test"])
    total_count = int(config["dataset"]["complete_trajectories"])
    test_trajectory_ids = range(total_count - test_count, total_count)
    eval_results = {
        str(horizon): {
            "ade": float(horizon),
            "fde": float(horizon + 1),
            "rmse_cart_m": float(horizon + 2),
            "trajectory_metrics": {
                str(trajectory_id): {
                    "ade": float(horizon + trajectory_id),
                    "fde": float(horizon + trajectory_id + 1),
                }
                for trajectory_id in test_trajectory_ids
            },
            "trajectory_count": test_count,
        }
        for horizon in REQUIRED_HORIZONS
    }
    if metrics:
        eval_results["256"].update(metrics)
    model_config = {
        "model_type": (
            "plgaformer"
            if model_key == "full"
            else "transformer"
            if model_key == "baseline"
            else model_key
        )
    }
    if model_key == "full":
        model_config["innovations"] = ["C: identified rotating-Earth 3-DOF prior with adaptive fusion"]
        model_config.update(FINAL_MODEL_FLAGS)
    display_names = {
        "baseline": "Transformer (baseline)",
        "full": "PLGAFormer (proposed)",
        "pit": "PIT",
        "kinematic": "Spherical kinematics",
        "rotating_3dof": "Rotating-Earth 3-DOF",
        "dlinear": "DLinear",
        "patchtst": "PatchTST",
        "itransformer": "iTransformer",
        "af_ciln": "AF-CILN",
    }
    payload = {
        "schema_version": 2,
        "formal_config_sha256": config.config_sha256,
        "run_id": f"run_{model_key}_{seed}",
        "timestamp": "20260808_000000",
        "seed": seed,
        "model": display_names.get(model_key, model_key),
        "model_key": model_key,
        "model_config": model_config,
        "horizons": list(REQUIRED_HORIZONS),
        "eval_results": eval_results,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "evidence_tier": "final",
        "test_evaluation_performed": True,
        "protocol_identity": core | {"run_signature": f"signature-{model_key}-{seed}"},
        "dataset_identity": {
            "dataset_protocol": config["dataset"]["protocol"],
            "dataset_sha256": config["dataset"]["sha256"],
            "sampling_interval_s": config["dataset"]["sampling_interval_s"],
        },
    }
    if af_root is not None:
        payload["config"] = {"af_ciln_root": str(af_root)}
    public_source_keys = set(config["main_results"].get("public_source_model_keys", []))
    if model_key in public_source_keys:
        from models.public_baselines import TSLIB_FILE_SHA256, _required_files

        source_model_type = "transformer" if model_key == "baseline" else model_key
        payload["external_source_provenance"] = {
            "kind": "TSLib",
            "repository": config["external_sources"]["tslib"]["repository"],
            "root": str(tmp_path / "Time-Series-Library"),
            "commit": config["external_sources"]["tslib"]["commit"],
            "model_types": [source_model_type],
            "source_files": {
                relative: TSLIB_FILE_SHA256[relative]
                for relative in _required_files([source_model_type])
            },
            "license": "MIT",
            "adapter_config": config["external_sources"]["tslib"]["adapter_config"],
        }
    return payload


def _write_record(tmp_path: Path, payload: dict, name: str) -> dict:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return normalize_run_record(path, payload)


def _complete_main_records(tmp_path: Path, config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = []
    af_root = tmp_path / "AF-CILN"
    af_root.mkdir(exist_ok=True)
    for model_key in config["main_results"]["trainable_model_keys"]:
        for seed in config["training"]["seeds"]:
            payload = _payload(
                tmp_path,
                config,
                model_key=model_key,
                seed=seed,
                af_root=af_root if model_key == "af_ciln" else None,
            )
            records.append(_write_record(tmp_path, payload, f"formal_sota_{model_key}_{seed}.json"))
    for model_key in config["main_results"]["analytical_models"]:
        records.append(_write_record(tmp_path, _payload(tmp_path, config, model_key=model_key, seed=42), f"formal_sota_{model_key}_42.json"))
    return records


def test_config_hash_and_nested_record_normalization(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    payload = _payload(tmp_path, config, model_key="full", seed=42)
    record = _write_record(tmp_path, payload, "formal_sota_full_42.json")

    assert config.config_sha256
    assert record["dataset_protocol"] == "hgv_multiregime_state_v2_1"
    assert record["scientific_protocol_core"]["prediction_horizons"] == list(REQUIRED_HORIZONS)
    assert record["model_config_sha256"]
    assert record["run_signature"] == "signature-full-42"


def test_wrong_dataset_hash_is_reported_without_rewriting_source(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    payload = _payload(tmp_path, config)
    payload["protocol_identity"]["dataset_sha256"] = "bad-hash"
    source = tmp_path / "formal_sota_baseline_42.json"
    source.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = source.read_bytes()
    audit = validate_protocol_core([normalize_run_record(source)], config)

    assert not audit["passed"]
    assert any(item["code"] == "PROTOCOL_CORE_MISMATCH" for item in audit["blockers"])
    assert source.read_bytes() == before


def test_main_completeness_reports_missing_seed_and_missing_checkpoint(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    records = _complete_main_records(tmp_path, config)
    records = [record for record in records if not (record["model_key"] == "full" and record["seed"] == 456)]
    audit = validate_main_completeness(records, config)

    assert not audit["paper_eligible"]
    assert any(item["code"] == "MISSING_REQUIRED_RUN" and item["context"]["model_key"] == "full" and item["context"]["seed"] == 456 for item in audit["blockers"])

    missing_checkpoint = _payload(tmp_path, config, model_key="baseline", seed=42)
    missing_checkpoint["checkpoint"] = str(tmp_path / "not-there.pth")
    missing_checkpoint["checkpoint_sha256"] = "deadbeef"
    record = _write_record(tmp_path, missing_checkpoint, "formal_sota_baseline_missing.json")
    missing_audit = validate_main_completeness([record], config)
    assert any(item["code"] == "MISSING_CHECKPOINT" for item in missing_audit["blockers"])


def test_duplicate_trainable_records_are_not_selected_by_timestamp(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    records = _complete_main_records(tmp_path, config)
    duplicate = _write_record(
        tmp_path,
        _payload(tmp_path, config, model_key="full", seed=42, metrics={"ade": 999.0}),
        "formal_sota_full_42_duplicate.json",
    )
    audit = validate_main_completeness(records + [duplicate], config)

    assert not audit["paper_eligible"]
    assert any(item["code"] == "AMBIGUOUS_DUPLICATE_RECORDS" for item in audit["blockers"])


def test_identical_analytical_duplicates_are_allowed_as_one_deterministic_row(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    records = _complete_main_records(tmp_path, config)
    duplicate = _write_record(
        tmp_path,
        _payload(tmp_path, config, model_key="kinematic", seed=123),
        "formal_sota_kinematic_123.json",
    )
    audit = validate_main_completeness(records + [duplicate], config)

    assert not any(item["code"] == "AMBIGUOUS_DUPLICATE_RECORDS" for item in audit["blockers"])
    assert len([row for row in audit["selected_records"] if row["model_key"] == "kinematic"]) == 1


def test_bundle_manifest_is_hashable_and_resolves_exact_checkpoint(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    records = _complete_main_records(tmp_path, config)
    audit = validate_main_completeness(records, config)
    manifest = build_evidence_bundle(records, audit, tmp_path / "bundle", config=config)

    assert manifest["bundle_id"]
    assert manifest["paper_eligible"] is True
    from utils.formal_evidence import load_evidence_bundle

    bundle = load_evidence_bundle(manifest["manifest_path"])
    checkpoint = resolve_bundle_checkpoint(bundle, "full", 456)
    assert checkpoint.is_file()


def test_af_ciln_is_rejected_from_paper_main_matrix(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    payload = _payload(tmp_path, config, model_key="af_ciln", seed=42, af_root=tmp_path / "missing-af-ciln")
    record = _write_record(tmp_path, payload, "formal_sota_af_ciln_42.json")
    audit = validate_main_completeness([record], config)

    assert any(item["code"] == "UNEXPECTED_MODEL_RECORD" for item in audit["blockers"])

def test_tampered_bundle_source_is_rejected(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    records = _complete_main_records(tmp_path, config)
    audit = validate_main_completeness(records, config)
    manifest = build_evidence_bundle(records, audit, tmp_path / "bundle", config=config)
    from utils.formal_evidence import load_evidence_bundle

    source = Path(manifest["source_records"][0]["source_path"])
    original = source.read_bytes()
    source.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError, match="source record hash mismatch"):
            load_evidence_bundle(manifest["manifest_path"])
    finally:
        source.write_bytes(original)


@pytest.mark.parametrize("mutation", ["empty_innovations", "missing_flag", "wrong_flag"])
def test_final_model_requires_explicit_nonempty_correct_flags(tmp_path, mutation):
    config = load_formal_config(CONFIG_PATH)
    payload = _payload(tmp_path, config, model_key="full", seed=42)
    if mutation == "empty_innovations":
        payload["model_config"]["innovations"] = []
    elif mutation == "missing_flag":
        payload["model_config"].pop("prior_blend_mode")
    else:
        payload["model_config"]["use_channel_residual"] = True
    record = _write_record(tmp_path, payload, f"formal_sota_full_42_{mutation}.json")
    audit = validate_main_completeness([record], config)
    assert any(item["code"] == "WRONG_FINAL_MODEL_CONFIGURATION" for item in audit["blockers"])


def test_empty_bundle_source_records_cannot_be_promoted(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="empty source_records"):
        build_evidence_bundle(
            [],
            {
                "kind": "main",
                "paper_eligible": True,
                "config_sha256": config.config_sha256,
                "record_count": 0,
                "selected_records": [],
            },
            tmp_path / "empty.json",
            config=config,
        )



def _build_synthetic_eligible_bundles(tmp_path: Path):
    config = load_formal_config(CONFIG_PATH)
    main_records = _complete_main_records(tmp_path / "main_sources", config)
    main_audit = validate_main_completeness(main_records, config)
    assert main_audit["paper_eligible"] is True
    main_manifest = build_evidence_bundle(
        main_records,
        main_audit,
        tmp_path / "main_bundle.json",
        config=config,
    )

    ablation_root = tmp_path / "ablation_sources"
    ablation_root.mkdir()
    ablation_records = []
    for model_key in ("spherical_prior", "schedule_only"):
        for seed in config["ablation"]["required_seeds"]:
            payload = _payload(tmp_path / "ablation_sources", config, model_key=model_key, seed=seed)
            payload["phase"] = config["ablation"]["phase"]
            ablation_records.append(
                _write_record(
                    ablation_root,
                    payload,
                    f"formal_{model_key}_{seed}.json",
                )
            )
    ablation_audit = validate_ablation_completeness(ablation_records, main_audit, config)
    assert ablation_audit["paper_eligible"] is True
    ablation_audit["main_bundle_id"] = main_manifest["bundle_id"]
    ablation_audit["main_bundle_path"] = main_manifest["manifest_path"]
    ablation_manifest = build_evidence_bundle(
        ablation_records,
        ablation_audit,
        tmp_path / "ablation_bundle.json",
        config=config,
    )
    return config, main_manifest, ablation_manifest


def test_formal_paper_synthetic_success_stages_complete_bundle(monkeypatch, tmp_path):
    config, main_manifest, ablation_manifest = _build_synthetic_eligible_bundles(tmp_path)
    from scripts import formal_pipeline
    from scripts import generate_taes_main_results

    monkeypatch.setattr(
        formal_pipeline,
        "audit_dataset",
        lambda _config: {
            "passed": True,
            "protocol": config["dataset"]["protocol"],
            "observed_sha256": config["dataset"]["sha256"],
        },
    )
    monkeypatch.setattr(
        generate_taes_main_results,
        "trajectory_maneuver_map",
        lambda: {
            trajectory_id: ("longitudinal", "turning", "weaving")[trajectory_id % 3]
            for trajectory_id in range(1440, 1800)
        },
    )
    output_dir = tmp_path / "staged"
    result = formal_pipeline.main(
        [
            "paper",
            "--config",
            str(config.path),
            "--main-bundle",
            str(main_manifest["manifest_path"]),
            "--ablation-bundle",
            str(ablation_manifest["manifest_path"]),
            "--output-dir",
            str(output_dir),
            "--permutations",
            "20",
            "--bootstrap-resamples",
            "20",
        ]
    )
    assert result == 0
    required = {
        "table_main_results.tex",
        "table_maneuver_results.tex",
        "main_results_summary.json",
        "table_ablation.tex",
        "artifact_manifest.json",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})
    artifact_manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert artifact_manifest["artifact_kind"] == "formal_v3_taes_complete_bundle"
    assert artifact_manifest["main_bundle_id"] == main_manifest["bundle_id"]
    assert artifact_manifest["ablation_bundle_id"] == ablation_manifest["bundle_id"]
    assert artifact_manifest["omitted_sections"] == [
        {"section": "robustness", "reason": "optional protocol-matched result was not supplied"},
        {"section": "efficiency", "reason": "optional protocol-matched result was not supplied"},
    ]
    assert "artifact_manifest.json" not in artifact_manifest["files"]


def test_formal_paper_rejects_wrong_bundle_type_hashes_empty_records_and_mismatch(tmp_path):
    config, main_manifest, ablation_manifest = _build_synthetic_eligible_bundles(tmp_path)
    from scripts import formal_pipeline

    def rewrite(source: Path, target: Path, mutate):
        payload = json.loads(source.read_text(encoding="utf-8"))
        mutate(payload)
        hashes = sorted(str(item.get("source_sha256", "")) for item in payload.get("source_records", []))
        payload["source_record_sha256"] = hashes
        payload["bundle_id"] = sha256_bytes(
            canonical_json(
                {
                    "config_sha256": payload.get("config_sha256"),
                    "source_record_sha256": hashes,
                    "kind": payload.get("bundle_kind"),
                }
            ).encode("utf-8")
        )
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cases = [
        ("wrong_type", lambda payload: payload.__setitem__("bundle_kind", "ablation"), False),
        ("wrong_config", lambda payload: payload.__setitem__("config_sha256", "0" * 64), False),
        ("wrong_dataset", lambda payload: payload.__setitem__("dataset_sha256", "1" * 64), False),
        ("empty_records", lambda payload: payload.update({"source_records": []}), False),
    ]
    for name, mutate, _ in cases:
        main_path = tmp_path / f"{name}_main.json"
        rewrite(Path(main_manifest["manifest_path"]), main_path, mutate)
        result = formal_pipeline.main(
            [
                "paper",
                "--json",
                "--dry-run",
                "--config",
                str(config.path),
                "--main-bundle",
                str(main_path),
                "--ablation-bundle",
                str(ablation_manifest["manifest_path"]),
            ]
        )
        assert result == 1, name

    mismatch_path = tmp_path / "mismatch_ablation.json"
    rewrite(
        Path(ablation_manifest["manifest_path"]),
        mismatch_path,
        lambda payload: payload.__setitem__("main_bundle_id", "not-the-selected-main"),
    )
    result = formal_pipeline.main(
        [
            "paper",
            "--json",
            "--dry-run",
            "--config",
            str(config.path),
            "--main-bundle",
            str(main_manifest["manifest_path"]),
            "--ablation-bundle",
            str(mismatch_path),
        ]
    )
    assert result == 1


@pytest.mark.skipif(
    not (CONFIG_PATH.parent.parent / "data_generation" / "data" / "processed" / "hgv_multiregime_dataset_v2_1.npz").is_file(),
    reason="complete local formal evidence is excluded from public Git",
)
def test_real_formal_v3_records_are_complete_and_hash_attested():
    config = load_formal_config(CONFIG_PATH)
    records = discover_run_records(resolve_config_path(config, "main_results_records"), "main")
    audit = validate_main_completeness(records, config)

    minimum_required = (
        len(config["main_results"]["trainable_model_keys"])
        * len(config["training"]["seeds"])
        + len(config["main_results"]["analytical_models"])
    )
    assert len(records) >= minimum_required
    assert audit["passed"] is True
    assert audit["paper_eligible"] is True
    assert audit["blockers"] == []
    assert audit["record_issues"] == {}

def test_incomplete_trajectory_population_is_not_paper_eligible(tmp_path):
    config = load_formal_config(CONFIG_PATH)
    payload = _payload(tmp_path, config, model_key="baseline", seed=42)
    payload["eval_results"]["256"]["trajectory_metrics"].pop("1440")
    payload["eval_results"]["128"]["trajectory_metrics"]["1440"]["ade"] = float("nan")
    payload["eval_results"]["64"]["rmse_cart_m"] = float("inf")
    record = _write_record(tmp_path, payload, "formal_sota_baseline_42.json")
    audit = validate_main_completeness([record], config)
    issues = audit["record_issues"][record["source_path"]]
    codes = {item["code"] for item in issues}

    assert "INCOMPLETE_TEST_TRAJECTORY_COVERAGE" in codes
    assert "TEST_TRAJECTORY_ID_SET_MISMATCH" in codes
    assert audit["paper_eligible"] is False
    assert "INCOMPLETE_TRAJECTORY_METRIC" in codes
    assert "MISSING_REQUIRED_METRIC" in codes


def test_aggregate_all_binds_ablation_to_fresh_main_bundle(monkeypatch, tmp_path):
    config = load_formal_config(CONFIG_PATH)
    main_records = _complete_main_records(tmp_path / "main_sources", config)
    main_audit = validate_main_completeness(main_records, config)
    ablation_root = tmp_path / "ablation_sources"
    ablation_root.mkdir()
    ablation_records = []
    for model_key in ("spherical_prior", "schedule_only"):
        for seed in config["ablation"]["required_seeds"]:
            payload = _payload(ablation_root, config, model_key=model_key, seed=seed)
            payload["phase"] = config["ablation"]["phase"]
            ablation_records.append(_write_record(
                ablation_root, payload, f"formal_{model_key}_{seed}.json"
            ))
    initial_ablation_audit = validate_ablation_completeness(
        ablation_records, main_audit, config
    )
    from scripts import formal_pipeline

    roots = {
        "main_results_records": tmp_path / "main_bundle",
        "ablation_records": tmp_path / "ablation_bundle",
    }
    monkeypatch.setattr(
        formal_pipeline,
        "_records_and_audits",
        lambda _: (main_records, ablation_records, main_audit, initial_ablation_audit),
    )
    monkeypatch.setattr(
        formal_pipeline,
        "resolve_config_path",
        lambda _, key: roots[key],
    )
    result = formal_pipeline._aggregate(config, "all", force=False)

    assert len(result["bundles"]) == 2
    main_manifest, ablation_manifest = result["bundles"]
    assert ablation_manifest["main_bundle_id"] == main_manifest["bundle_id"]
    assert ablation_manifest["paper_eligible"] is True


def test_paper_failure_removes_transactional_staging(monkeypatch, tmp_path):
    config, main_manifest, ablation_manifest = _build_synthetic_eligible_bundles(tmp_path)
    from scripts import formal_pipeline
    from scripts import generate_taes_main_results
    from scripts import generate_taes_secondary_results

    monkeypatch.setattr(
        generate_taes_main_results,
        "trajectory_maneuver_map",
        lambda: {
            trajectory_id: ("longitudinal", "turning", "weaving")[trajectory_id % 3]
            for trajectory_id in range(1440, 1800)
        },
    )
    monkeypatch.setattr(
        generate_taes_secondary_results,
        "main",
        lambda _: (_ for _ in ()).throw(RuntimeError("synthetic secondary failure")),
    )
    output_dir = tmp_path / "published"
    result = formal_pipeline.main([
        "paper",
        "--config", str(config.path),
        "--main-bundle", str(main_manifest["manifest_path"]),
        "--ablation-bundle", str(ablation_manifest["manifest_path"]),
        "--output-dir", str(output_dir),
        "--permutations", "20",
        "--bootstrap-resamples", "20",
    ])
    assert result == 1

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".*-*"))
