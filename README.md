# HGVTP-PLGAformer

`HGVTP-PLGAformer` is a research repository for hypersonic glide vehicle trajectory prediction. It combines:

- physics-aware sequence modeling with `PLGAFormer`
- HGV trajectory data generation and validation
- one formal CLI for Overall Prediction, Mechanism/Physics, Generalization/Robustness, and Efficiency
- paper artifact generation for figures and tables

The repository now uses a TSLib-like top-level layout while keeping small compatibility wrappers for release tooling.

## Environment

Recommended:

- Python 3.10+
- PyTorch-compatible environment with GPU support if you plan to run full experiments

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
# Required for the paper-facing public baseline matrix:
pip install -r requirements-public-baselines.txt
```

## Current Paper Mainline

Run from the repository root:

```bash
python run.py formal status --json
python run.py formal validate-data
python run.py formal audit --json
```

These commands inspect the frozen formal evidence chain and do not launch
training. Running `python run.py` without a command prints help and performs no
experiment.

The numbered `python run.py --task ...` workflow remains available for existing
experiments and compatibility. It is separate from the paper-facing formal
evidence route; its future status will be decided after the project structure
review.

The active formal dataset follows a reference-audited multi-regime HGV framing:

- `1800` complete simulated HGV trajectories
- `1000` points per trajectory at `1 Hz`
- three primary maneuver classes: longitudinal glide, turning glide, and weaving glide
- two balanced vertical regimes inside every primary class:
  quasi-equilibrium and skip glide
- per-trajectory randomized attack-angle and bank-angle profiles
- split-wise, joint-stratum Latin-hypercube initial-condition/control sampling
- ideal bank commands plus rate-limited achieved bank histories
- trajectory-level train/validation/test split before sliding-window construction
- default windows: `seq_len=256`, `pred_len=256`; 16 fixed training origins and
  a five-step evaluation stride
- formal artifact:
  `data_generation/data/processed/hgv_multiregime_dataset_v2_1.npz`

The dataset is already frozen. Regenerate it only when intentionally creating a
new, incompatible protocol generation. For ordinary formal work, validate the
existing artifact:

```bash
python run.py formal validate-data
```

Do not use numbered experiment outputs in the manuscript unless the formal
audit accepts their dataset, protocol, checkpoint, seed, and metric identity.

## Layout

- `run.py`: canonical unified experiment entrypoint
- `requirements.txt`: canonical dependency list
- `data_generation/`: simulator and generated dataset pipeline
- `data_provider/`: strict trajectory-level dataset loader and checks
- `experiments/`: experiment implementations plus isolated result generations
- `scripts/`: auxiliary and compatibility experiment wrappers
- `models/`: model implementations and architecture defaults
- `tests/`: protocol and artifact validation
- `docs/`: audits, planning notes, and manuscript/release readiness records
- `PublicRelease/`: release boundary and path-free machine-readable evidence

## Experiment Status

- `overall_prediction/`: logical study registry for the `exp1_sota` Main Results implementation.
- `mechanism_analysis/`: logical phase4 registry for `exp2_ablation` plus the physics evaluator.
- `generalization_robustness/`: logical registry for frozen-model dynamics/noise/history tests.
- `efficiency/`: logical registry for the non-training computational-cost benchmark.
- `legacy/`: explicit index for retained `exp1`--`exp7` compatibility paths; no legacy result is a formal fallback.

## Dataset Protocol

The active new-evidence route is `hgv_multiregime_state_v2_1`. It retains the
trajectory-level split contract while adding balanced vertical regimes,
randomized controls, and stored physical diagnostics:

- generate complete physics-based trajectories first
- save the complete trajectories as `raw_hgv_trajectories.npz` for dataset coverage, 2D/3D trajectory, operational-envelope, and initial-condition figures
- split complete trajectories into train, validation, and test before windowing
- generate sliding windows only inside each split
- avoid leakage from neighboring windows of the same trajectory
- store `sampling_interval_s`, `points_per_trajectory`, `trajectory_duration_s`, `seq_len`, `pred_len`, and `window_stride` in the processed `.npz`

The clean complete-trajectory artifact is the paper-facing dataset evidence.
The window arrays are training/evaluation tensor caches. Do not describe
window count as trajectory count in manuscripts; report complete trajectories
and supervised windows separately. Sensor noise and missing observations are
separate robustness protocols rather than part of the formal state benchmark.

Validate the processed data with:

```bash
python run.py --task validate-data
```

The processed `.npz` stores both step counts and physical timing. Under the
active 1 s protocol, the four reporting horizons are 32/64/128/256 steps =
32/64/128/256 s. Historical `trajectory_level_v1` and rejected
`pit_aligned_radar_v1` results remain available for provenance but must not be
combined with this protocol.

See `docs/EXPERIMENT_PROTOCOL_REGISTRY.md` before running or assembling tables.

## Formal Evidence Commands

The paper-facing route is the explicit formal command family. It targets
physics-aware long-horizon prediction of complete simulated HGV trajectories and
never launches full training implicitly:

```bash
python run.py formal status --json
python run.py formal validate-data
python run.py formal main --tslib-root "PATH/TO/Time-Series-Library" --models transformer,plgaformer,kinematic,rotating_3dof,dlinear,patchtst,itransformer
python run.py formal mechanism
python run.py formal robustness --dry-run
python run.py formal efficiency --dry-run
python run.py formal ablation --phase phase4_final_mechanism_controls  # explicit compatibility alias
python run.py formal aggregate --dry-run
python run.py formal aggregate
python run.py formal audit --json
python run.py formal paper --dry-run
python run.py formal paper --robustness PATH.json --efficiency PATH.json
```

`status`, the four study `--dry-run` routes, `audit --dry-run`, and
`aggregate --dry-run` are read-only. `aggregate` writes
`main_run_set_manifest.json` and `ablation_run_set_manifest.json` beside the
formal run records; incomplete manifests remain explicitly
`paper_eligible=false`. `paper` accepts only eligible Main Results and matched
phase4 ablation bundles. Optional robustness and efficiency evidence is accepted
only through the explicit flags above and is revalidated by the secondary
generator. Artifacts are generated transactionally under an identity derived
from both required bundle IDs, optional evidence hashes, generator hashes, and
statistical settings.
A missing seed, protocol/hash mismatch, checkpoint mismatch, or missing required
metric blocks paper artifact generation.

The paper-facing main matrix uses the MIT-licensed TSLib checkout for the
Transformer, DLinear, PatchTST, and iTransformer baselines. Pass its pinned
checkout with `--tslib-root` (or set `HGV_TSLIB_ROOT`); every run record stores
the repository, commit, model/layer hashes, and adapter protocol. The local
reimplementations remain compatibility paths only.

PIT and AF-CILN are not part of the paper-facing main matrix. PIT remains
available only for explicit isolated analysis. AF-CILN is accepted only when
`--af-ciln-root` points to the audited external checkout; its run record must
retain the source root, commit, source hash, and observation protocol.

Recovery is explicit and resumable: run `formal status`, then pass only the
missing model/seed units to `formal main --models ... --seeds ...` or
`formal ablation --phase phase4_final_mechanism_controls --seeds ...`. Completed
formal units remain untouched; run `formal aggregate` only after the matrix
has been re-audited.

`experiments/generate_paper_artifacts_v2.py` remains a historical diagnostic
compatibility generator. It is not the active formal source for paper
numbers and must not be used to promote current claims.

## Agent-Facing Files

- `README.md`: human-facing project overview
- `AGENTS.md`: authoritative agent instructions
- `CLAUDE.md`: compatibility entrypoint for Claude-style tooling

## Public Release and Reproducibility Boundary

This repository contains research code, paper artifacts, and local experiment structure. It does not guarantee that every internal dataset, checkpoint, or cache should be published.

Before packaging a public snapshot, read:

- `PublicRelease/README.md`

That document defines:

- what is intended to be public
- what is intentionally excluded
- what a reproducer should expect to rebuild locally
- which path-free CSV tables and hashes support the current paper

The compact evidence snapshot is under `PublicRelease/evidence`. Regenerate it
from a complete local experiment workspace with:

```bash
python scripts/generate_release_evidence.py
```
