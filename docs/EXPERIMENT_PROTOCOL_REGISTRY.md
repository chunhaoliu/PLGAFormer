# Experiment Protocol Registry

This file is the routing authority for result artifacts. A result directory is
not evidence by name alone; every claim still requires its run JSON, dataset
hash, checkpoint hash, seeds, and frozen evaluation record.

## Active route: `hgv_multiregime_state_v2_1`

- Dataset: `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz`
- SHA-256:
  `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
- Observation interval: 1 s
- Main task input/forecast: 256/256 steps = 256/256 s
- Reporting checkpoints: 32/64/128/256 steps within the same directly generated
  256-step forecast; these are not separately trained horizon-specific models
- Complete trajectories: 1,800, with 300 in each of the six vertical-regime
  and maneuver strata
- Complete split: 1,260/180/360 train/validation/test, with 210/30/60 from
  every joint stratum
- Sampling: split-wise, joint-stratum Latin-hypercube candidate designs over
  initial states and controls, followed by physical-constraint rejection and
  resampling when a candidate is infeasible
- Bank response: ideal commands and 1.5--3.0 deg/s rate-limited achieved
  histories are stored separately
- Seeds: 42, 123, 456
- Selection: validation only, at most 50 epochs with early stopping
- Test: one frozen evaluation after selection
- Data status: formal audit passed; the 120-trajectory strict-integration audit
  also passed with a 22.04 m maximum ECEF discrepancy
- Manuscript status: data protocol and figures synchronized on 2026-07-31;
  performance results remain intentionally absent
- Training status: the three-seed FP32 validation-selection ceiling check
  completed on 2026-08-04 without test evaluation; paper-facing Main Results
  and matched ablations remain pending
- Execution profile: deterministic physics-prior caching is enabled, the
  measured Windows default is `workers=0`, and BF16 remains an explicitly
  identified opt-in mode; see `docs/FORMAL_TRAINING_EFFICIENCY.md`

New Main Results and ablations must write to:

```text
experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/<tier>/
experiments/exp1_sota/trained_models/formal_v3/hgv_multiregime_state_v2_1/<tier>/
experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/<tier>/
experiments/exp2_ablation/trained_models/formal_v3/hgv_multiregime_state_v2_1/<tier>/
```

Scaler artifacts are isolated under:

```text
data_generation/data/processed/protocol_artifacts/
  hgv_multiregime_state_v2_1/<dataset-hash-prefix>/<experiment>/
```

## Length-protocol diagnostics

Observation-length sensitivity uses `64/128/256 -> 256` while preserving the
same complete trajectories, split membership, 1 Hz sampling, training-origin
policy, and evaluation stride. A bounded internal diagnostic separately trains
`256 -> 64/128/256` and compares each endpoint with the corresponding prefix
of the main direct 256-step forecast. Prefix and horizon-specific results must
not be pooled or described as the same training task. Derived NPZ files record
the exact parent SHA-256 and are window views, not newly simulated datasets.
See `docs/LENGTH_PROTOCOL_DIAGNOSTICS.md`.

## Rejected diagnostic route: `pit_aligned_radar_v1`

The 1,000-trajectory artifact is retained for provenance, but its single-site
radar layer fails the visibility gate: only 12.96% of ideal measurements have
positive elevation and every trajectory starts below the radar horizon. Do not
launch formal training from it or describe it as continuous radar tracking.
See `docs/DATASET_REFERENCE_GAP_AUDIT_20260730.md`.

## Frozen predecessor: `hgv_multiregime_state_v2`

The 1,200-trajectory predecessor remains available for provenance.  It used
fixed geographic initial longitude/latitude, ideal step bank commands, and a
speed-law low branch that was not materially exercised.  Do not combine its
results, scalers, or figures with v2.1.

## Frozen legacy route: `trajectory_level_v1`

The following are retained for provenance and for rebuilding the current
pre-PIT manuscript snapshot:

- `experiments/exp1_sota/results/formal/`
- `experiments/exp1_sota/results/formal_v2/`
- `experiments/exp2_ablation/results/formal_v2/`
- corresponding historical checkpoint and generated-table artifacts

These directories are read-only evidence in normal work. Do not append new
multi-regime runs or combine their statistics with
`hgv_multiregime_state_v2_1`.

## Experiment-evidence gate

The manuscript data protocol, dataset figures, and experimental design now
match the active artifact. Numerical performance tables and claims remain
withheld until all of the following exist for the active route:

1. complete three-seed Main Results;
2. complete matched ablations;
3. frozen test evaluation records;
4. protocol-aware tables and figures generated only from the current frozen evidence artifacts;
5. a manuscript-wide replacement of timing, dataset, table, and limitation
   statements followed by a clean compile and visual QA.

Until then, do not insert or reinterpret any performance claim as evidence
from the multi-regime protocol.

## Four registered paper-level studies

The current formal configuration registers exactly four paper-level studies;
the old numbered directories remain implementation/diagnostic locations:

| Study | Formal command | Input | Registered implementation |
|---|---|---|---|
| Overall Prediction Performance | `python run.py formal main` | frozen dataset | `experiments/overall_prediction/main_results.py` |
| Mechanism Ablation and Physical Consistency | `python run.py formal mechanism` | eligible Main bundle | `experiments/mechanism_analysis/ablation_study.py` + `physics_consistency.py` |
| Generalization and Robustness | `python run.py formal robustness` | eligible Main bundle | `experiments/generalization_robustness/dynamics_shift.py` |
| Efficiency and Computational Cost | `python run.py formal efficiency` | eligible Main bundle | `experiments/efficiency/efficiency_experiment.py` |

The registry is logical and does not move or copy the preserved evidence roots.
`exp5_missing_data`, `exp7_longterm`, old ablation phases, and old
`exp3_robustness` routes remain explicit legacy diagnostics. Formal missing
evidence is a blocker; no study may fall back to pilot, partial, quick, smoke,
or historical protocol outputs.

## Current evidence bundles

The machine-readable orchestration contract is
`configs/formal_v3.json`; its current canonical SHA-256 is
  `37209254645c1bfacea975371f40be9b602dcfa03904091587f6a98ff557a539`.

Run records remain immutable under the preserved protocol roots. The
read-only status/audit layer normalizes them through
`utils/formal_evidence.py`. Aggregation may add only:

- `experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final/main_run_set_manifest.json`
- `experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final/ablation_run_set_manifest.json`

Paper staging is bundle-specific under
`experiments/taes_submission_artifacts/generated/formal_v3/hgv_multiregime_state_v2_1/<artifact_id>/`.
The artifact identity binds Main, Ablation, optional evidence, generator source
hashes, and inferential settings rather than reusing the Main bundle ID.
No manifest or staged artifact is paper-eligible until the complete Main Results
and phase4 matched-ablation matrices pass their recorded hash/checkpoint/metric
gates.

The implementation audit on 2026-08-08 reads 23 Main Results JSON records and
reports missing `full:456` and `pit:42/123/456`; no phase4 ablation bundle is
complete. Therefore no current numerical paper claim is promoted.
