# Confirmatory Holdout Protocol

Status: preregistered before dataset generation and before model inference.

## Purpose

This protocol defines one untouched simulation-only holdout for confirming the
existing PLGAFormer paper result. It is part of the sole PLGAFormer mainline. It
does not define a new model, a new training protocol, or a competing result
version.

## Frozen data contract

- Protocol identifier: `hgv_multiregime_confirmatory_holdout`
- Generator seed: `20260831`
- Complete trajectories: `360`
- Trajectory IDs: integers `1800` through `2159`, inclusive
- Joint strata: quasi-equilibrium/skip-glide crossed with longitudinal/turning/
  weaving
- Per-stratum quota: exactly `60` complete trajectories
- Samples per trajectory: `1000`
- Sampling interval: `1 s`
- Input length: `256`
- Prediction length: `256`
- Evaluation window stride: `5`
- Windows per trajectory: `98`
- Total evaluation windows: `35,280`
- Integration: DOP853, relative tolerance `1e-7`, absolute tolerance `1e-9`,
  maximum step `4 s`
- Simulator, state variables, control sampling, physical-envelope filters, and
  acceptance criteria: identical to `hgv_multiregime_state_v2_1`
- Output path:
  `data_generation/data/processed/hgv_confirmatory_holdout.npz`

The output path is non-overwriting. If either the dataset or its manifest
already exists, generation must stop. The generated manifest must bind this
document by SHA-256 and record the dataset SHA-256.

## Frozen model-evaluation contract

- Methods: PLGAFormer, Transformer, DLinear, PatchTST, and iTransformer only
- Checkpoint seeds: `42`, `123`, and `456`
- Checkpoint authority: the already-eligible Main Results bundle
- Scalers: the already-fitted Main Results input and output scalers; refitting is
  forbidden
- PLGAFormer inference policy: lock steps `64`, time constant `450 s`, decay
  power `2.5`
- Horizons: `32`, `64`, `128`, and `256 s`
- Metrics: trajectory-level ADE, FDE, and RMSE, using the existing formal metric
  implementation
- Additional breakdown: registered maneuver classes

No model training, checkpoint replacement, policy adjustment, scaler fitting,
trajectory removal, or seed replacement is permitted after generation. The
result is retained regardless of whether PLGAFormer ranks first.

## Evidence boundary

The holdout is independently generated under the same simulator family and is
therefore confirmatory for simulation-based generalization only. It is not
evidence of real-flight, cross-simulator, sensor-degraded, or operational
deployment performance.
