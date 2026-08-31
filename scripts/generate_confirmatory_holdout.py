#!/usr/bin/env python3
"""Generate the preregistered, all-confirmatory HGV holdout exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generation.multiregime_protocol import (
    CONTROL_PARAMETER_NAMES,
    MANEUVER_TYPES,
    VERTICAL_REGIMES,
    MultiregimeHGVSimulator,
    sample_maneuver_profile,
)
from scripts.generate_multiregime_dataset import (
    DESIGN_DIMENSION,
    STATE_FEATURE_NAMES,
    TARGET_FEATURE_NAMES,
    _build_split,
    _sample_initial_state,
    _simulate_candidate,
    _space_filling_design,
)


PROTOCOL = "hgv_multiregime_confirmatory_holdout"
GENERATION_SEED = 20260831
TRAJECTORY_COUNT = 360
TRAJECTORY_ID_START = 1800
POINTS = 1000
SAMPLING_INTERVAL_S = 1.0
SEQ_LEN = 256
PRED_LEN = 256
EVAL_WINDOW_STRIDE = 5
INTEGRATION_METHOD = "DOP853"
INTEGRATION_RTOL = 1e-7
INTEGRATION_ATOL = 1e-9
INTEGRATION_MAX_STEP_S = 4.0
MAX_ATTEMPT_FACTOR = 6
OUTPUT = (
    PROJECT_ROOT
    / "data_generation"
    / "data"
    / "processed"
    / "hgv_confirmatory_holdout.npz"
)
PROTOCOL_DOCUMENT = PROJECT_ROOT / "docs" / "CONFIRMATORY_HOLDOUT_PROTOCOL.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confirmatory_design_rows() -> list[tuple[str, str, np.ndarray]]:
    per_stratum = TRAJECTORY_COUNT // (len(VERTICAL_REGIMES) * len(MANEUVER_TYPES))
    rows: list[tuple[str, str, np.ndarray]] = []
    for vertical_index, vertical in enumerate(VERTICAL_REGIMES):
        for maneuver_index, maneuver in enumerate(MANEUVER_TYPES):
            design = _space_filling_design(
                per_stratum,
                seed=GENERATION_SEED,
                split_seed=GENERATION_SEED,
                split_index=3,
                vertical_index=vertical_index,
                maneuver_index=maneuver_index,
            )
            rows.extend((vertical, maneuver, unit_sample) for unit_sample in design)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "Confirmatory generation is non-overwriting; existing artifact must be "
            f"audited, not replaced: {output}"
        )
    if not PROTOCOL_DOCUMENT.is_file():
        raise FileNotFoundError(f"Preregistered protocol is missing: {PROTOCOL_DOCUMENT}")

    requested_design = confirmatory_design_rows()
    if len(requested_design) != TRAJECTORY_COUNT:
        raise RuntimeError("Confirmatory design does not contain exactly 360 rows.")

    rng = np.random.default_rng(GENERATION_SEED)
    simulator = MultiregimeHGVSimulator()
    states_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    bank_command_rows: list[np.ndarray] = []
    bank_achieved_rows: list[np.ndarray] = []
    dynamic_pressure_rows: list[np.ndarray] = []
    load_factor_rows: list[np.ndarray] = []
    heat_proxy_rows: list[np.ndarray] = []
    maneuver_labels: list[str] = []
    vertical_labels: list[str] = []
    joint_labels: list[str] = []
    control_rows: list[np.ndarray] = []
    initial_rows: list[np.ndarray] = []
    extrema_counts: list[int] = []

    attempts = 0
    maximum_attempts = TRAJECTORY_COUNT * MAX_ATTEMPT_FACTOR
    started = time.perf_counter()
    for vertical_regime, maneuver_type, design_sample in requested_design:
        accepted = False
        local_attempt = 0
        while not accepted and attempts < maximum_attempts:
            attempts += 1
            local_attempt += 1
            candidate_sample = (
                design_sample if local_attempt == 1 else rng.random(DESIGN_DIMENSION)
            )
            profile = sample_maneuver_profile(
                rng,
                vertical_regime=vertical_regime,
                maneuver_type=maneuver_type,
                unit_sample=candidate_sample[:15],
            )
            initial_state = _sample_initial_state(
                simulator,
                vertical_regime=vertical_regime,
                qeg_gamma_reference_deg=profile.qeg_gamma_reference_deg,
                unit_sample=candidate_sample[15:],
            )
            candidate = _simulate_candidate(
                simulator,
                initial_state,
                profile,
                points=POINTS,
                sampling_interval_s=SAMPLING_INTERVAL_S,
                integration_method=INTEGRATION_METHOD,
                integration_rtol=INTEGRATION_RTOL,
                integration_atol=INTEGRATION_ATOL,
                integration_max_step_s=INTEGRATION_MAX_STEP_S,
            )
            if candidate is None:
                continue
            states, diagnostics, extrema_count = candidate
            states_rows.append(states.astype(np.float32))
            alpha_rows.append(diagnostics["alpha"].astype(np.float32))
            bank_command_rows.append(diagnostics["bank_command"].astype(np.float32))
            bank_achieved_rows.append(diagnostics["bank"].astype(np.float32))
            dynamic_pressure_rows.append(
                diagnostics["dynamic_pressure_pa"].astype(np.float32)
            )
            load_factor_rows.append(diagnostics["load_factor_g"].astype(np.float32))
            heat_proxy_rows.append(diagnostics["heat_rate_proxy"].astype(np.float32))
            maneuver_labels.append(maneuver_type)
            vertical_labels.append(vertical_regime)
            joint_labels.append(f"{vertical_regime}__{maneuver_type}")
            control_rows.append(profile.parameter_vector())
            initial_rows.append(initial_state)
            extrema_counts.append(extrema_count)
            accepted = True
        if not accepted:
            raise RuntimeError(
                f"Accepted {len(states_rows)}/{TRAJECTORY_COUNT} trajectories after "
                f"{attempts} attempts; protocol review is required."
            )

    states = np.stack(states_rows)
    maneuver_arr = np.asarray(maneuver_labels)
    vertical_arr = np.asarray(vertical_labels)
    joint_arr = np.asarray(joint_labels)
    trajectory_ids = np.arange(
        TRAJECTORY_ID_START,
        TRAJECTORY_ID_START + TRAJECTORY_COUNT,
        dtype=np.int64,
    )
    windows = _build_split(
        states,
        maneuver_arr,
        vertical_arr,
        trajectory_ids,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        stride=EVAL_WINDOW_STRIDE,
    )
    expected_windows = TRAJECTORY_COUNT * (
        ((POINTS - SEQ_LEN - PRED_LEN) // EVAL_WINDOW_STRIDE) + 1
    )
    if len(windows["X"]) != expected_windows:
        raise RuntimeError(
            f"Expected {expected_windows} windows, got {len(windows['X'])}."
        )

    elapsed_s = time.perf_counter() - started
    payload = {
        "dataset_protocol": np.asarray(PROTOCOL),
        "parent_training_protocol": np.asarray("hgv_multiregime_state_v2_1"),
        "evidence_role": np.asarray("untouched_confirmatory_holdout"),
        "observation_protocol": np.asarray("complete_simulator_state_v2_1"),
        "input_source": np.asarray("clean_simulated_hgv_state"),
        "target_source": np.asarray("clean_simulated_hgv_position"),
        "state_feature_names": np.asarray(STATE_FEATURE_NAMES),
        "target_feature_names": np.asarray(TARGET_FEATURE_NAMES),
        "maneuver_taxonomy": np.asarray(MANEUVER_TYPES),
        "vertical_regime_taxonomy": np.asarray(VERTICAL_REGIMES),
        "control_parameter_names": np.asarray(CONTROL_PARAMETER_NAMES),
        "sampling_interval_s": np.asarray(SAMPLING_INTERVAL_S),
        "points_per_trajectory": np.asarray(POINTS),
        "trajectory_duration_s": np.asarray((POINTS - 1) * SAMPLING_INTERVAL_S),
        "seq_len": np.asarray(SEQ_LEN),
        "pred_len": np.asarray(PRED_LEN),
        "window_stride": np.asarray(EVAL_WINDOW_STRIDE),
        "generation_seed": np.asarray(GENERATION_SEED),
        "sampling_design": np.asarray("confirmatory_joint_stratum_latin_hypercube"),
        "trajectory_ids": trajectory_ids,
        "trajectory_labels": maneuver_arr,
        "vertical_regimes": vertical_arr,
        "joint_strata": joint_arr,
        "clean_trajectories": states,
        "alpha_commands": np.stack(alpha_rows),
        "bank_commands": np.stack(bank_command_rows),
        "bank_achieved": np.stack(bank_achieved_rows),
        "dynamic_pressure_pa": np.stack(dynamic_pressure_rows),
        "load_factor_g": np.stack(load_factor_rows),
        "heat_rate_proxy": np.stack(heat_proxy_rows),
        "control_parameters": np.stack(control_rows),
        "initial_states": np.stack(initial_rows).astype(np.float32),
        "vertical_extrema_counts": np.asarray(extrema_counts, dtype=np.int64),
        "integration_method": np.asarray(INTEGRATION_METHOD),
        "integration_rtol": np.asarray(INTEGRATION_RTOL),
        "integration_atol": np.asarray(INTEGRATION_ATOL),
        "integration_max_step_s": np.asarray(INTEGRATION_MAX_STEP_S),
        "generation_attempts": np.asarray(attempts),
        "X_confirmatory": windows["X"],
        "y_confirmatory": windows["y"],
        "maneuver_labels_confirmatory": windows["maneuver_labels"],
        "vertical_regimes_confirmatory": windows["vertical_regimes"],
        "joint_strata_confirmatory": windows["joint_strata"],
        "trajectory_ids_confirmatory": windows["trajectory_ids"],
        "window_starts_confirmatory": windows["window_starts"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)

    stratum_counts = {
        str(label): int(np.count_nonzero(joint_arr == label))
        for label in np.unique(joint_arr)
    }
    manifest = {
        "schema_version": 1,
        "artifact": str(output),
        "sha256": sha256_file(output),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_document": str(PROTOCOL_DOCUMENT),
        "protocol_document_sha256": sha256_file(PROTOCOL_DOCUMENT),
        "dataset_protocol": PROTOCOL,
        "parent_training_protocol": "hgv_multiregime_state_v2_1",
        "evidence_role": "untouched_confirmatory_holdout",
        "generation_seed": GENERATION_SEED,
        "trajectory_count": TRAJECTORY_COUNT,
        "trajectory_id_range": [TRAJECTORY_ID_START, TRAJECTORY_ID_START + TRAJECTORY_COUNT - 1],
        "joint_stratum_counts": stratum_counts,
        "window_count": int(len(windows["X"])),
        "points_per_trajectory": POINTS,
        "sampling_interval_s": SAMPLING_INTERVAL_S,
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "window_stride": EVAL_WINDOW_STRIDE,
        "integration": {
            "method": INTEGRATION_METHOD,
            "rtol": INTEGRATION_RTOL,
            "atol": INTEGRATION_ATOL,
            "max_step_s": INTEGRATION_MAX_STEP_S,
        },
        "generation_attempts": attempts,
        "acceptance_rate": TRAJECTORY_COUNT / attempts,
        "elapsed_s": elapsed_s,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
