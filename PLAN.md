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

## Figure Evidence Pass (2026-08-25)

Add manuscript figures one at a time after a reference-paper and evidence-gap review. The first approved addition is a six-panel motion-regime figure generated from the frozen `hgv_multiregime_state_v2_1` dataset.

Figure contract:

- claim: the formal dataset covers distinct quasi-equilibrium and skip-glide behavior across longitudinal, turning, and weaving motion;
- evidence: one deterministic representative from each of the six joint strata, selected without using prediction error;
- layout: six independent Python-rendered panels assembled by LaTeX in a 2-by-3 double-column figure;
- integrity: verify the frozen dataset hash, record representative trajectory IDs, preserve all 1,000 samples of each selected trajectory, and export a path-free QA manifest;
- boundary: this is dataset/task evidence only and does not claim prediction superiority.

Validation:

- run strict Python source preflight and PDF text-size audits for every panel;
- compile and render the active IEEE TAES manuscript;
- require zero LaTeX errors, undefined references, and overfull boxes before completion.
## Single-Mainline Cleanup (2026-08-28)

Treat the current `main` branch as the only authoritative paper and experiment mainline. Clean the working repository without changing the frozen data protocol, trained-model contents, formal result values, or manuscript claims.

Scope:

- create a recoverable Git tag at the current clean `main` commit;
- derive an active-artifact whitelist from the paper-eligible Main and Ablation manifests;
- retain the frozen dataset, protocol scalers, final result records, and every checkpoint referenced by the active manifests;
- move inactive checkpoints, pre-promotion results, learned-only capacity-control artifacts, historical data products, and generated legacy figures into one dated archive directory;
- delete only reproducible caches and confirmed temporary files after their exact paths are validated;
- preserve the divergent `mainline-unification` worktree until its useful cleanup changes have been reviewed; do not merge or delete it during the artifact pass.

Integrity gates:

- record original path, archived path, byte size, and SHA-256 for every moved file;
- verify all archived files after the move and require no missing active artifacts;
- rerun `formal status --json` and `formal audit --json` after cleanup;
- require Main and Ablation to remain paper-eligible and the dataset/config hashes to remain unchanged;
- run the focused formal-evidence and pipeline tests before committing the cleanup tooling and documentation.

Completed validation:

- safety tag: `before-mainline-cleanup-20260828` at `c2f7e11`;
- archive: 215 files (510.84 MiB) under `D:\Research\HGV_Code\Archive\HGVTP_mainline_20260828`, with zero archive hash failures and zero source remnants;
- active evidence: 23 manifest-referenced checkpoints present with matching SHA-256 values;
- formal evidence: overall paper eligibility retained, with 17 Main records and 6 Ablation records;
- frozen hashes retained: config `967310425a89d5a0acdb098c77939647aa22548335719846cdf274a8dcf04165` and dataset `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`;
- focused validation: 38 tests passed and 3 were skipped by design.

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
