# HGVTP-PLGAFormer Current Work Plan

## Objective

Maintain one paper and code mainline for simulation-based long-horizon hypersonic glide vehicle trajectory prediction. The method combines a Transformer proposal with an online-identified rotating-Earth three-degree-of-freedom proposal through bounded adaptive fusion.

## Frozen Scientific Contract

- Dataset protocol: `hgv_multiregime_state_v2_1`.
- Complete trajectories: 1,800; 1,000 samples per trajectory at 1 Hz.
- Source-trajectory split: 1,260/180/360 train/validation/test.
- Input and maximum forecast: 256 s / 256 s.
- Reporting horizons: 32, 64, 128, and 256 s.
- Learned-model seeds: 42, 123, and 456.
- Selection: validation checkpoint selection followed by one frozen test evaluation.
- Statistical unit: held-out source trajectory, not overlapping windows.
- Claim boundary: simulation only; no flight, radar, hardware-in-the-loop, or deployment validation.

Dataset splits, preprocessing, metrics, seeds, and final result values must not change without an explicit new protocol decision.

## Repository Boundary

Git contains active source code, formal configuration, tests, compact bundle summaries, manifests, and path-free public evidence tables. Generated datasets, checkpoints, per-run trajectory records, training histories, pilots, logs, and manuscript workspace files remain local or archived.

The public baseline matrix uses pinned official TSLib source for Transformer, DLinear, PatchTST, and iTransformer. PIT and AF-CILN are not paper-facing baselines because a complete protocol-matched official-source comparison is unavailable.

## Verified Evidence

- Main and mechanism-ablation bundles are paper-eligible in the complete local workspace.
- Dynamics-shift robustness and efficiency artifacts are bound to the same frozen protocol and Main bundle.
- The adaptive gate has seed-and-trajectory-robust gains for selected short and intermediate metrics, while its 256 s marginal advantage over the fixed schedule is not seed-robust.
- PLGAFormer should not be described as uniformly superior to the rotating-Earth analytical propagator.

## Remaining Work

1. Finish the repository release boundary and validate public-clone behavior.
2. Review and split the current implementation into focused local commits.
3. Audit every manuscript number and claim against the compact evidence package.
4. Refine the main tables and figures without adding unsupported experiments.
5. Run an editor-style submission audit, compile the final PDF, and prepare the upload package.

## Acceptance Criteria

- The complete local workspace retains paper-eligible formal status.
- Public Git contains no dataset, checkpoint, pilot, raw per-run record, or machine-specific path.
- Core tests pass in both complete-local and public-release modes.
- Manuscript claims match the frozen evidence and state limitations explicitly.
- The final Git diff is reviewable and the remote is not updated until the user approves the commits.
