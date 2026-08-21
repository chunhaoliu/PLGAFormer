# HGV Project Mainline Unification Design

## Objective

Consolidate the repository around one active HGV trajectory-prediction line:

- one audited dataset protocol: `hgv_multiregime_state_v2_1`;
- one proposed model: the validation-selected PLGAFormer;
- one active experiment contract and one command entrypoint;
- one paper-facing Main Results bundle and one mechanism-ablation bundle;
- one isolated optimization area for candidates that cannot overwrite formal evidence.

The consolidation must preserve all current paper-eligible results, hashes, checkpoints, and provenance. Historical material is moved to `D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main` when it is no longer required by the active dependency graph. Nothing uncertain is deleted.

## Scope decomposition

This work is split into two sequential projects.

1. **Mainline consolidation**: remove competing active routes, centralize configuration and paths, isolate historical data/code/results, and prove that the current formal evidence still validates.
2. **PLGAFormer performance optimization**: design and evaluate new PLGAFormer candidates against the frozen mainline after consolidation passes all acceptance criteria.

This specification covers project 1. Project 2 receives a separate design and experiment plan so that engineering cleanup cannot silently alter the scientific comparison protocol.

## Current-state diagnosis

The repository already has a valid formal core, but several historical layers remain visible as if they were equally active.

- `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz` is the active dataset selected by `data_generation/data_paths.py`.
- The same directory also contains predecessor, pilot, radar-aligned, raw, and derived artifacts.
- `configs/formal_v3.json` is the only standalone JSON experiment contract, while `models/__init__.py`, experiment modules, and command-line defaults duplicate part of its training configuration.
- `models/plgaformer.py` contains the selected model and retained historical candidate modules in one file.
- `run.py` exposes both the paper-facing `formal` route and legacy `exp1` through `exp7` routes.
- `experiments` contains current formal artifacts together with legacy, duplicate, optional, and manuscript-generation directories.
- The repository currently contains 16 NPZ files, 56 checkpoints, and 63 JSON files; these artifacts do not all belong to the active paper line.

## Canonical mainline

### Dataset authority

The only dataset allowed for new Main Results, ablation, and model-promotion experiments is:

`data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz`

Its SHA-256 identity, trajectory-level split, manifest, audit, convergence report, and protocol-scoped scalers remain mandatory. The artifact is not renamed during the first consolidation because existing formal records and audit files bind its identity. Version strings remain internal provenance and are not exposed as manuscript method versions.

Other NPZ files are classified before movement:

- **required provenance**: retained read-only if an active audit or result bundle references the exact file hash;
- **reproducible derivative**: regenerated from the canonical dataset and excluded from the active root;
- **historical protocol**: moved to the external Archive with a manifest containing original path, size, SHA-256, and reason;
- **temporary view**: removed from the active dependency graph and regenerated on demand under `tmp`.

Environment overrides such as `HGV_DATASET_PATH` remain available for tests and diagnostics but paper-facing commands must reject an identity that differs from the canonical dataset contract.

### Model authority

`models.model_factory.create_registered_model("plgaformer", ...)` is the only supported construction route for the proposed model. The active constructor resolves to the selected PLGAFormer configuration:

- standard Transformer encoder-decoder backbone;
- rotating-Earth 3-DOF prior;
- adaptive bounded prior fusion;
- multi-head trajectory output;
- sparse physics attention disabled;
- physics corrector disabled;
- channel residual disabled.

Public comparison adapters remain in the repository because they are required to reproduce the Main Results matrix. PIT, AF-CILN compatibility code, obsolete candidate-search code, and inactive PLGAFormer variants are not active model versions. They are either isolated behind explicit diagnostic imports or archived after dependency tests prove that formal Main and Ablation do not import them.

The selected model flags are defined once in a dedicated active-model contract and consumed by the factory, formal evidence validator, formal runner, checkpoint resolver, and ablation runner. Tests must fail if any consumer reconstructs a different architecture.

### Configuration authority

The current content of `configs/formal_v3.json` remains the immutable evidence contract for existing results. Consolidation introduces a stable mainline loading API rather than allowing modules to read duplicated constants independently.

The loading API provides:

- dataset identity and split;
- training seeds, epochs, batch size, optimizer, scheduler, and early stopping;
- sequence, decoder-context, prediction-horizon, and evaluation protocols;
- active PLGAFormer flags;
- registered public baselines;
- artifact roots and evidence requirements.

`HGVConfig` may retain physical constants and compatibility methods, but paper-facing training values must be derived from the loaded mainline contract. Command-line arguments may select a registered unit or resource setting; they may not silently override frozen data, split, metric, horizon, or test-policy fields.

Candidate optimization configurations are stored outside the formal result roots and are explicitly marked diagnostic until promotion. A candidate cannot mutate the active evidence contract in place.

### Entrypoint authority

`python run.py formal ...` remains the single active experiment interface during compatibility migration. Its supported paper-facing operations are:

- data validation and status audit;
- Main Results units and aggregation;
- mechanism-ablation units and aggregation;
- optional registered secondary studies;
- model-optimization candidates in an isolated artifact root.

The `--task exp1` through `--task exp7`, `all`, and `enhancements` routes are removed from the default help and active dependency graph after equivalent formal commands and tests exist. Legacy wrappers are archived rather than kept as competing entrypoints.

### Experiment and result authority

The active evidence set is:

- Main Results: current verified `exp1_sota` formal bundle and exact checkpoints;
- Mechanism Ablation: current verified `exp2_ablation` formal bundle and exact checkpoints;
- Secondary studies: retained only when explicitly registered and protocol-matched;
- Optimization: a new isolated candidate area with no permission to overwrite Main or Ablation.

Existing formal artifact paths are not moved until a relocation test demonstrates that bundle records, source records, checkpoints, hashes, and audit commands still resolve. Historical experiment source directories and nonformal results are moved in small batches, each with a generated archive manifest and a Git commit.

## Target logical structure

```text
HGVTP_PLGAformer-main/
  configs/
    formal_v3.json                 # immutable current evidence contract
  data_generation/
    data/processed/                # canonical data plus required provenance only
  models/
    plgaformer.py                  # selected model and necessary ablation switches
    model_factory.py               # only proposed-model construction route
    public_baselines.py            # pinned public comparison adapters
    baseline_models.py             # active analytical prior implementation
  experiments/
    overall_prediction/            # shared Main training/evaluation engine
    mechanism_analysis/            # shared ablation engine
    model_optimization/            # isolated candidate artifacts and reports
    secondary/                     # registered optional studies only
  scripts/
    formal_pipeline.py             # single orchestration surface
    run_formal_sota_unit.py
    run_formal_ablation_unit.py
  tests/
  run.py
```

The physical directory migration may preserve compatibility paths temporarily, but README documentation, help output, and new code refer only to the logical structure above.

## Archive policy

The external archive root is:

`D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main`

Every archive batch contains a JSON manifest with:

- archive timestamp;
- source repository commit;
- original relative path;
- archived relative path;
- file size and SHA-256 for material artifacts;
- classification and reason;
- known importing files before migration;
- restoration instructions.

Tracked files are moved with Git-aware operations after imports are updated. Generated binaries are moved with explicit absolute paths after their hashes are recorded. No recursive deletion is used.

## Migration sequence

1. Create and verify a Git recovery tag at the clean pre-migration commit.
2. Generate an active dependency and artifact inventory without changing evidence.
3. Add mainline contract tests for dataset, model flags, configuration, entrypoint, and formal bundle resolution.
4. Centralize active configuration reads and remove conflicting paper-facing defaults.
5. Make the formal entrypoint the only documented and default route.
6. Isolate inactive model adapters and obsolete candidate-search code.
7. Archive historical datasets and temporary views after dependency and hash checks.
8. Archive legacy experiment routes and nonformal result artifacts in small, reversible batches.
9. Run the full test suite, data audit, formal status, bundle verification, and Git diff review.
10. Freeze the consolidated mainline before starting the separate PLGAFormer optimization project.

## Safety and scientific-integrity rules

- Do not change dataset contents, trajectory IDs, split membership, normalization, horizons, metrics, seeds, or test policy during consolidation.
- Do not alter or regenerate current Main or Ablation JSON records and checkpoints.
- Do not use test results to select an optimization candidate.
- Do not mix candidate artifacts with paper-eligible formal artifacts.
- Do not claim that an archive movement improves model performance.
- Do not push commits or tags without separate user authorization.
- Stop a migration batch if any formal bundle loses a source record, checkpoint, hash, or protocol identity.

## Validation strategy

Each migration batch must pass the narrow tests affected by that batch. The completed consolidation must pass all of the following:

1. Repository worktree contains only intentional changes.
2. Full automated test suite passes with the explicit project Python environment.
3. Canonical dataset validation passes and reports the expected dataset and test-ID hashes.
4. `formal status` reports the existing Main and Ablation bundles as paper-eligible.
5. Every existing final record still has `evidence_tier=final` and `test_evaluation_performed=true` where required.
6. Dataset, formal-config, model-config, and checkpoint hashes still verify.
7. The proposed model reconstructs identically for seeds 42, 123, and 456.
8. Default help and README expose one active experiment route.
9. No active source imports an archived module or references an archived dataset.
10. Archive manifests permit exact restoration of every moved material artifact.

## Acceptance criteria

Mainline consolidation is complete only when:

- one dataset protocol is active for all new experiments;
- one PLGAFormer constructor and one selected flag set are active;
- one formal configuration loading path controls paper-facing runs;
- one command surface is documented and tested;
- current Main and Ablation evidence remains unchanged and eligible;
- old datasets, models, scripts, and results are absent from the active dependency graph;
- all uncertain material is preserved in the external Archive with hashes;
- the repository is ready for a separate, protocol-frozen PLGAFormer optimization cycle.
