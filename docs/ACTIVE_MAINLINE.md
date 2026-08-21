# Active HGV Paper Mainline

## Authority and scope

`python run.py formal ...` is the only supported experiment command surface.
It governs one frozen HGV dataset, one selected PLGAFormer construction, the
registered comparison matrix, and the Main/Mechanism evidence chain. Retained
legacy source is historical provenance only. Inactive generated material is
preserved reversibly under
`D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main`; it is not an
alternative active project version.

This consolidation changes command routing and documentation only. It does not
change the frozen configuration, data, splits, metrics, checkpoints, result
records, or manuscript values.

## Frozen dataset identity and protocol

| Field | Frozen value |
|---|---|
| Protocol | `hgv_multiregime_state_v2_1` |
| Dataset | `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz` |
| Dataset SHA-256 | `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7` |
| Complete trajectories | 1,800 |
| Train/validation/test | 1,260 / 180 / 360 complete trajectories |
| Split unit | complete trajectory ID, before window generation |
| Test-ID SHA-256 | `c5851880f81e334717527bec68e91dccab59e78a0ef99d609b04926b5ad3e070` |
| Sampling | 1 Hz; 1,000 points per trajectory |
| Input/decoder/prediction | 256 / 128 / 256 steps |
| Reporting horizons | 32, 64, 128, 256 s |
| Training origins / evaluation stride | 16 / 5 |
| Seeds / maximum epochs / batch size | 42, 123, 456 / 50 / 128 |
| Test policy | one frozen test evaluation after validation selection |

The internal evidence contract remains `configs/formal_v3.json`. The filename
is intentionally unchanged because existing evidence records are bound to its
configuration identity. Version-like internal paths are provenance identifiers,
not competing manuscript method versions.

## Active model contract

The exact registered model types are:

```text
transformer
kinematic
rotating_3dof
dlinear
plgaformer
patchtst
itransformer
```

The trainable Main matrix is Transformer, PLGAFormer, DLinear, PatchTST, and
iTransformer at seeds 42, 123, and 456. The spherical kinematic and
rotating-Earth 3-DOF models are deterministic analytical anchors.

The selected PLGAFormer is reconstructed only through
`models.model_factory.create_registered_model("plgaformer", ...)` with:

| Setting | Active value |
|---|---|
| sparse physics attention | disabled |
| physics corrector | disabled |
| multi-head trajectory output | enabled |
| analytical-prior fusion | enabled |
| channel residual | disabled |
| prior | rotating-Earth 3-DOF |
| fusion | adaptive bounded fusion |

## Innovation-aligned mechanism controls

The active interpretation uses three controlled concepts:

1. **Learned-only capacity control:** an architecture-matched Transformer
   backbone with analytical-prior, physics-attention, physics-correction, and
   residual-head paths disabled.
2. **Prior control:** `spherical_prior` replaces the rotating-Earth 3-DOF
   prior with spherical constant-velocity kinematics while retaining adaptive
   fusion.
3. **Fusion control:** `schedule_only` retains the rotating-Earth 3-DOF prior
   but removes learned state/disagreement gating from the fusion weight.

The required formal Mechanism bundle has the paper rows `baseline`,
`spherical_prior`, `schedule_only`, and `full`; baseline and full are
reused from the Main bundle. The learned-only capacity control is a separate
hash-bound control and is not automatically part of the required bundle.

## Supported command surface

```bash
# Read-only state and data checks
python run.py formal status --json
python run.py formal validate-data --json
python run.py formal audit --json

# Formal units; inspect each runner's frozen options before execution
python run.py formal main --help
python run.py formal mechanism --help

# Read-only aggregation checks
python run.py formal aggregate --kind main --dry-run
python run.py formal aggregate --kind ablation --dry-run

# Manifest writes after all required units validate
python run.py formal aggregate --kind main
python run.py formal aggregate --kind ablation

# Registered optional secondary studies
python run.py formal robustness --dry-run
python run.py formal efficiency --dry-run

# Paper staging gate
python run.py formal paper --dry-run --json
```

There is no active `--task exp1` through `--task exp7` route and no duplicate
`sota` or `ablation` command alias. Mechanism ablation is invoked through
`formal mechanism`; its evidence is aggregated with
`formal aggregate --kind ablation`.
`main --help` and `mechanism --help` are delegated to the corresponding
downstream runner parsers. Runner options such as `--models` and
`--selection-only` are forwarded in their original order without requiring a
`--` separator.

## Output and evidence boundaries

- `status`, `validate-data`, and `audit` are read-only.
- `validate-data --json` emits exactly one JSON envelope with frozen
  config/dataset provenance and separate trajectory/multiregime validator
  reports and exit codes, including on failure.
- Every study command with `--dry-run` is read-only.
- `aggregate --dry-run` is a fail-closed eligibility gate scoped by
  `--kind`; it returns nonzero when the requested matrix is incomplete or
  ineligible and never writes manifests.
- `main` and `mechanism` can write training records/checkpoints only for
  explicitly requested units that satisfy the frozen runtime contract.
- `aggregate` without `--dry-run` writes or validates run-set manifests; it
  does not create missing experimental evidence.
- `paper` accepts only eligible, provenance-matched Main and Mechanism
  bundles. Optional robustness/efficiency artifacts are revalidated separately.
- Candidate, historical, single-seed, incomplete, or protocol-mismatched output
  is diagnostic evidence and cannot replace verified paper evidence.
- A timeout, background launch, log line, CSV row, or checkpoint without its
  matched JSON/hash chain is not a passing result.

## Protected evidence

The following roots are protected from movement, regeneration, or overwrite:

```text
configs/formal_v3.json
data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz
experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp1_sota/trained_models/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/final
experiments/exp2_ablation/trained_models/formal_v3/hgv_multiregime_state_v2_1/final
```

A record is paper-facing only after the configured evidence gates verify its
dataset, configuration, model, checkpoint, required metrics,
`evidence_tier=final`, and `test_evaluation_performed=true`. Main and
Mechanism bundles remain separate from compact release evidence under
`PublicRelease/evidence`, local detailed artifacts, and future optimization
candidates.

## Historical and archival material

Legacy numbered experiments, inactive model integrations, predecessor datasets,
candidate searches, and temporary protocol views may remain temporarily while
their dependencies are audited. Their presence does not make them active.
The retained `run_all_experiments.py` and `run_all_enhancements.py` paths
are fail-fast archival shims that execute no experiments or artifact
generation.
Archival movement is a later, separate, hash-attested task; this command-surface
task does not move or delete them.
