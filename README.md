# HGVTP-PLGAformer

This repository contains the active research code and evidence pipeline for
long-horizon hypersonic glide vehicle trajectory prediction with PLGAFormer.
There is one supported paper mainline:

```bash
python run.py formal --help
```

Numbered experiments, candidate searches, PIT/AF-CILN integrations, and other
legacy launch routes are historical provenance, not active paper choices. Their
retained source or externally archived copies must not be used to overwrite the
formal evidence described below.

## Frozen data and protocol

The only dataset authorized for new Main Results, mechanism ablation, or model
promotion is:

- protocol: `hgv_multiregime_state_v2_1`
- file: `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz`
- SHA-256: `526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
- complete trajectories: 1,800, split 1,260/180/360 by complete trajectory ID
- test-ID SHA-256: `c5851880f81e334717527bec68e91dccab59e78a0ef99d609b04926b5ad3e070`
- sampling: 1 Hz, 1,000 points per trajectory
- task: 256-step input, 128-step decoder context, 256-step prediction
- reporting horizons: 32, 64, 128, and 256 s
- training: seeds 42/123/456, at most 50 epochs, batch size 128
- test policy: one frozen test evaluation after validation selection

The immutable evidence contract remains `configs/formal_v3.json`; its filename
is retained because existing records are hash-bound to that internal identity.
Do not regenerate or replace the frozen dataset for ordinary formal runs.

## Active models and mechanism controls

The exact active registry is `transformer`, `kinematic`,
`rotating_3dof`, `dlinear`, `plgaformer`, `patchtst`, and
`itransformer`. The selected PLGAFormer uses a Transformer
encoder-decoder, rotating-Earth 3-DOF prior, adaptive bounded prior fusion, and
multi-head trajectory output. Sparse physics attention, the physics corrector,
and the channel residual are disabled.

The three innovation-aligned control concepts are:

1. learned-only matched-capacity backbone, which removes analytical-prior paths;
2. spherical-prior replacement, which isolates the rotating-Earth 3-DOF prior;
3. schedule-only fusion, which removes learned state/disagreement gating.

The required formal mechanism bundle uses the spherical-prior and schedule-only
controls together with the reused Transformer and full-PLGAFormer Main records.
The learned-only capacity control remains separately hash-bound and is not
silently promoted into that required bundle.

## Supported commands

```bash
python run.py formal status --json
python run.py formal validate-data --json
python run.py formal main --help
python run.py formal mechanism --help
python run.py formal aggregate --kind main --dry-run
python run.py formal aggregate --kind ablation --dry-run
python run.py formal audit --json
python run.py formal robustness --dry-run
python run.py formal efficiency --dry-run
python run.py formal paper --dry-run --json
```

`main --help` and `mechanism --help` delegate to their downstream frozen
runner parsers, so documented options can be passed directly without a `--`
separator. `validate-data --json` emits one JSON envelope containing config,
dataset, trajectory-validator, and multiregime-validator provenance.
`status`, `validate-data`, `audit`, and all `--dry-run` routes are
read-only. `aggregate --dry-run` is also a fail-closed gate: it returns
nonzero when the requested `--kind` is incomplete or ineligible.
`main` and `mechanism` may train only explicitly selected missing units
under the frozen contract. `aggregate` without `--dry-run` writes evidence
manifests; `paper` without `--dry-run` stages artifacts only from eligible
bundles. No command may turn diagnostic, single-seed, historical, or
protocol-mismatched output into paper evidence.

## Evidence and archive boundary

The protected active records and checkpoints are under:

- `experiments/exp1_sota/{results,trained_models}/formal_v3/hgv_multiregime_state_v2_1/final`
- `experiments/exp2_ablation/{results,trained_models}/formal_v3/hgv_multiregime_state_v2_1/final`

Compact public evidence is under `PublicRelease/evidence`. Local datasets,
checkpoints, logs, candidate artifacts, and detailed histories are separate
reproducibility material and are not manuscript claims by themselves.
Historical material is preserved reversibly under
`D:\Research\HGV_Code\Archive\HGVTP_PLGAformer-main`.
Inactive numbered launchers, candidate-search sources, predecessor model
integrations, and their tests are removed from the active tree after being
copied to the external archive with hash manifests. They are not public
entrypoints; use the formal command surface above.

See [docs/ACTIVE_MAINLINE.md](docs/ACTIVE_MAINLINE.md) for the complete active
contract and [PublicRelease/README.md](PublicRelease/README.md) for release
boundaries.

## Environment and layout

Use Python 3.10+ with a compatible PyTorch environment:

```bash
pip install -r requirements.txt
pip install -r requirements-public-baselines.txt
```

The core layout is `data_generation/`, `data_provider/`, `models/`,
`experiments/`, `scripts/`, `tests/`, `docs/`, and
`PublicRelease/`. Run commands from the repository root.
