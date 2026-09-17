# Public Release Boundary

This repository publishes the active PLGAFormer source code, frozen protocol definitions, formal experiment entrypoints, tests, and a compact evidence snapshot. It does not publish the complete local compute workspace.

## Current Scope

PLGAFormer predicts hypersonic glide vehicle trajectories by fusing a Transformer proposal with an online-identified rotating-Earth three-degree-of-freedom analytical proposal. The verified evidence is simulation only; it does not establish flight, radar, hardware-in-the-loop, or deployment performance.

The frozen experiment protocol contains 1,800 complete trajectories, each with 1,000 samples at 1 Hz, divided into six joint motion strata. Complete source trajectories are split into 1,260/180/360 training/validation/test sets before window extraction. The input and maximum direct forecast are both 256 s, and learned methods use seeds 42, 123, and 456.

The primary learning-method comparison additionally uses a disjoint 360-trajectory confirmatory holdout. It contains 60 trajectories in each joint motion stratum, uses trajectory IDs 1800--2159 and generator seed `20260831`, and is evaluated with the frozen seeds, checkpoints, scalers, and inference policy. The protocol is stored in `docs/CONFIRMATORY_HOLDOUT_PROTOCOL.md`; its SHA-256 is bound into the generated dataset manifest and the compact audit in `PublicRelease/evidence/confirmatory_holdout.json`.

## Included

- active model, data, experiment, and evidence-generation code;
- the pinned public TSLib adapter and source provenance checks;
- frozen text manifests, configuration, and validation metadata;
- tests for protocol, model interfaces, evidence gates, and public artifacts;
- compact machine-readable tables under `PublicRelease/evidence`;
- the confirmatory-holdout protocol, generator, evaluator, audit, and tests;
- plotting code for the Information Fusion figures and their source-data checks;
- documentation for supported commands and evidence boundaries.

## Excluded

- generated NPZ datasets and scalers;
- checkpoints and trained weights;
- per-run trajectory-level JSON records and training histories;
- convergence pilots, smoke outputs, logs, caches, and temporary workspaces;
- manuscript workspace files and machine-specific paths.

These assets remain in the local research workspace or recoverable archive. Their exclusion keeps the Git repository focused and does not change the frozen protocol or reported values.

## Reproduction Modes

A public clone can inspect the implementation, validate lightweight fixtures, regenerate the dataset, and rerun experiments after installing the required dependencies and obtaining the pinned TSLib checkout.

A complete local experiment workspace can additionally run:

```bash
python run.py formal status --json
python run.py formal audit --json
python run.py formal paper --dry-run
python scripts/generate_release_evidence.py
python scripts/generate_confirmatory_holdout.py --help
python scripts/evaluate_confirmatory_holdout.py --preflight-only
python scripts/evaluate_confirmatory_holdout.py --audit-only
```

`formal status` is paper-eligible only when the local per-run records and their hash-matched checkpoints are present. Their absence in a public clone is an expected release boundary, not evidence that the published summaries were generated without provenance checks.

## Public Evidence

`PublicRelease/evidence/evidence_manifest.json` binds the compact tables to the frozen dataset, formal configuration, Main bundle, Ablation bundle, complete paper bundle, and retained diagnostic analyses. The CSV files omit local paths and trajectory-level arrays while retaining the reported means, sample standard deviations, mechanism controls, and dynamics-shift results. The adaptive-gate paired analysis and computational-cost measurements are retained for traceability only and are not current manuscript claims.

`PublicRelease/evidence/confirmatory_holdout.json` records the registered methods, seeds, horizons, trajectory counts, protocol and dataset hashes, checkpoint identities, aggregate errors, and maneuver breakdowns used by the confirmatory evaluation. It does not contain generated trajectories or model weights.
