# Experiment Protocol Registry

This file is the routing authority for current PLGAFormer experiments. A folder
name is not evidence by itself; paper-facing results must match the frozen data,
configuration, source, seed, checkpoint, metric, and test-evaluation identities.

## Active Protocol

- Protocol: `hgv_multiregime_state_v2_1`.
- Dataset SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`.
- Formal configuration: `configs/formal_v3.json`.
- Configuration SHA-256: `967310425a89d5a0acdb098c77939647aa22548335719846cdf274a8dcf04165`.
- Complete trajectories: 1,800, balanced across six vertical-regime and
  maneuver strata.
- Source-trajectory split: 1,260/180/360 train/validation/test.
- Sampling: 1 Hz; 1,000 states per trajectory.
- Input/direct forecast: 256 s / 256 s.
- Reporting horizons: 32, 64, 128, and 256 s from the same direct forecast.
- Learned-model seeds: 42, 123, and 456.
- Selection/test policy: validation checkpoint selection followed by one frozen
  test evaluation.
- Statistical unit: held-out source trajectory, not an overlapping window.

The simulator, sampling design, physical checks, split construction, and derived
window counts are specified in `docs/MULTIREGIME_V21_DATA_PROTOCOL.md`.

## Paper-Level Studies

| Study | Command | Role |
|---|---|---|
| Overall Prediction Performance | `python run.py formal main` | Main accuracy and public baseline comparison |
| Mechanism Ablation and Physical Consistency | `python run.py formal mechanism` | Matched prior and fusion controls |
| Generalization and Robustness | `python run.py formal robustness` | Frozen dynamics-shift evaluation |
| Efficiency and Computational Cost | `python run.py formal efficiency` | Parameters, FLOPs, memory, latency, throughput |

PIT, AF-CILN, missing-observation diagnostics, shortened-history diagnostics,
convergence pilots, smoke runs, and historical numbered outputs are not
paper-facing fallbacks.

## Current Evidence State

- Main bundle: `ff4a707476fa98c8270880bbc2f7be23c9a73b8ad89462c18cf8a6a434d208ac`.
- Ablation bundle: `afa2867d0a7a5d0f9e0d38e6b3691268e5cbdae836642053192029637bd6026f`.
- Complete paper bundle: `ebbcec4b2321dafeb825878b873eddf710ade15d5a242673ee170a494f8585a6`.
- Main records: 21; complete and paper-eligible in the local workspace.
- Ablation records: 6; complete and paper-eligible in the local workspace.
- Robustness and efficiency artifacts are bound to the same dataset,
  configuration, and Main bundle.

Run the read-only gate before using any local artifact:

```bash
python run.py formal status --json
python run.py formal audit --json
```

The exact supported and unsupported manuscript statements are maintained in
`docs/PAPER_EVIDENCE_MAP.md`.

## Local and Public Artifact Boundary

Generated datasets, checkpoints, per-run trajectory JSON, checkpoint-bound
manifests, training histories, pilots, logs, and machine-specific staging remain
local. Git publishes active source and the path-free compact evidence under
`PublicRelease/evidence`.

A public clone is therefore not expected to report `paper_eligible=true` until
its user regenerates or restores all hash-matched local evidence. The committed
public evidence manifest records the identities of the complete workspace used
to produce the paper tables.

## Legacy Protocols

`hgv_multiregime_state_v2`, `trajectory_level_v1`, and
`pit_aligned_radar_v1` remain provenance-only identities. Their results,
scalers, figures, and checkpoints must never be pooled with the active protocol.
Creating an incompatible successor requires an explicit protocol decision and a
fresh evidence chain; it must not silently overwrite the current results.
