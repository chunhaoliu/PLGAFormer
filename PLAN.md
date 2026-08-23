# HGVTP-PLGAFormer Current Work Plan

## Objective

Maintain one paper and code mainline for simulation-based long-horizon hypersonic glide vehicle trajectory prediction. The method combines a Transformer proposal with an online-identified rotating-Earth three-degree-of-freedom proposal through bounded adaptive fusion.

## Current Editorial Pass (2026-08-23)

Synchronize the active IEEE TAES manuscript with the promoted paper-facing PLGAFormer result while preserving the frozen scientific contract. The edit is limited to reproducibility wording, claim boundaries, and manuscript-facing evidence records; it must not retrain models, regenerate data, change reported values, or alter the formal configuration.

Planned changes:

- expose the inference-only policy and its no-parameter-update boundary in the abstract and formal protocol table;
- state explicitly that the performance claim is restricted to the listed public learning-based baselines;
- refresh the path-free public evidence snapshot and evidence map to the promoted Main/Ablation/robustness artifacts; historical efficiency and paired-gate files remain separate from manuscript claims;
- compile and audit the revised PDF and record the new manuscript hashes in a separate editorial-sync manifest.

Validation:

- verify the formal status remains passing and the promoted Main/Ablation bundles remain unchanged;
- check that all main, ablation, and robustness numbers remain identical to the promoted evidence tables;
- compile the manuscript and inspect the log for errors, undefined references, and overfull boxes.

## Submission Readiness Pass (2026-08-23)

Audit the active IEEE TAES manuscript for structural submission readiness after the paper-facing evidence synchronization. Reuse only figures that describe the current PLGAFormer and generate any quantitative figure from the current path-free evidence tables. Do not reuse historical comparison plots or add unsupported experiments.

Completed:

- verified that the prior manuscript inserted no scientific figures and screened the existing figure directory for current-model provenance;
- inserted the verified PLGAFormer architecture figure and linked it from the method section;
- generated the current multi-horizon learning-based performance figure with Python from `PublicRelease/evidence/main_results.csv`, retaining 60 trainable rows and excluding 24 analytical-model rows from the ranking plot;
- compiled the final nine-page manuscript with zero overfull boxes, undefined references, or LaTeX errors;
- recorded the updated method, figure, PDF, and figure-QA hashes in the editorial-sync manifest.

Validation completed:

- Python backend selected and source preflight passed with 20 checks and no warnings or failures;
- performance figure uses three-seed mean plus sample standard deviation and makes no statistical-significance claim;
- final PDF font audit found no text below 5.0 pt in the generated performance figure, and all 11 manifest file hashes match their current files.

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
