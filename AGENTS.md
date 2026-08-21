# AGENTS.md

This file is the single source of truth for agent-facing instructions in this repository.

## Mission

Work on `HGVTP-PLGAformer`, a research codebase for hypersonic glide vehicle trajectory prediction with a physics-aware transformer variant named `PLGAFormer`.

The repository mixes:

- core model code
- data generation and validation
- formal experiment runners
- paper artifact generation

Agents should preserve this split and avoid collapsing research, release, and manuscript concerns into one script or one folder.

## Authority

If multiple agent-facing files exist, follow this precedence:

1. `AGENTS.md`
2. task-specific files explicitly referenced by `AGENTS.md`
3. compatibility entrypoints such as `CLAUDE.md`

`CLAUDE.md` is only a compatibility shim and should not become the primary instruction file again.

## Working Style

- Prefer minimal, targeted changes.
- Preserve backward compatibility unless the user explicitly approves a breaking cleanup.
- Treat experiment reproducibility, protocol clarity, and release boundaries as first-class concerns.
- Do not silently change scientific claims, reported metrics, or experiment scope.

## Repository Map

- `run.py`: top-level unified experiment entrypoint
- `requirements.txt`: canonical dependency list
- `data_generation/`: simulator and generated dataset pipeline
- `data_provider/`: strict dataset loader and validation utilities
- `experiments/`: formal experiment code and paper artifacts
- `scripts/`: active formal coordination plus retained archival utilities
- `models/`: source-of-truth model implementations and config
- `tests/`: protocol, metrics, and artifact tests
- `docs/`: audits, planning notes, and manuscript/release readiness records
- `PublicRelease/`: public release policy, reproducibility scope, and release boundaries

## Current Formal Commands

Run from repository root unless the user explicitly asks otherwise.

- Install deps: `pip install -r requirements.txt`
- Command help: `python run.py formal --help`
- Read-only status: `python run.py formal status --json`
- Validate frozen data: `python run.py formal validate-data --json`
- Read-only evidence audit: `python run.py formal audit --json`
- Main Results runner: `python run.py formal main [explicit runner options]`
- Mechanism/physics runner: `python run.py formal mechanism [explicit runner options]`
- Optional secondary robustness inspection: `python run.py formal robustness --dry-run`
- Optional secondary efficiency inspection: `python run.py formal efficiency --dry-run`
- Main evidence eligibility gate: `python run.py formal aggregate --kind main --dry-run`
- Mechanism evidence eligibility gate: `python run.py formal aggregate --kind ablation --dry-run`
- Paper staging check: `python run.py formal paper --dry-run --json`

Running `python run.py` without a command prints help and performs no work.
The only supported public experiment entrypoint is `python run.py formal ...`.
Numbered experiment routes and duplicate Main/Mechanism aliases are retired;
retained legacy modules are archival provenance rather than public commands.
`main --help` and `mechanism --help` delegate to their downstream runner
parsers, and runner options are forwarded directly in their original order.
`validate-data --json` must emit one JSON envelope on success or failure.
`aggregate --dry-run` is read-only but fail-closed for the requested
`--kind`; a missing or ineligible matrix must return nonzero.

The default formal protocol is `hgv_multiregime_state_v2_1`: `1800`
trajectories, `1000` points/trajectory, `1 Hz`, `seq_len=256`, `pred_len=256`,
16 fixed training origins, and evaluation stride 5. Regenerating the frozen
dataset requires an explicit protocol-change decision.

## Project Conventions

### Data protocol

- Prefer `hgv_multiregime_state_v2_1`; retain `trajectory_level_v1` and
  `hgv_multiregime_state_v2` as frozen provenance only.
- Treat `longitudinal`, `turning`, and `weaving` as the primary three-class maneuver taxonomy unless the user explicitly approves an unseen/mixed-maneuver extension set.
- Preserve complete generated trajectories inside the active NPZ before
  constructing supervised windows.
- Use the raw long-trajectory artifact, not training windows, as the source for dataset coverage, 2D/3D trajectory distribution, operational-envelope, and initial-condition figures.
- Split complete trajectories before generating windows.
- Do not introduce random-window leakage across train, validation, and test.
- Preserve timing metadata in generated `.npz` files: `sampling_interval_s`, `points_per_trajectory`, `trajectory_duration_s`, `seq_len`, `pred_len`, and `window_stride`.
- In manuscripts and summaries, never describe window count as trajectory count; report complete trajectories and supervised windows separately.

### Paths

- Use project-root-relative paths based on `Path(__file__).resolve()`.
- Do not rely on `os.getcwd()` for experiment-critical paths.

### Model and configuration handling

- Keep `configs/formal_v3.json` as the experiment and evidence contract for
  current formal work.
- Keep model implementations and architecture defaults under `models/`;
  `HGVConfig` must not silently override the formal-v3 protocol.
- Construct the proposed model only through
  `create_registered_model("plgaformer", ...)`; do not introduce a competing
  registry, candidate constructor, or method/version branch into paper work.

### Results and artifacts

- Keep path-free compact evidence under `PublicRelease/evidence`. Local per-run records, trajectory arrays, training histories, and checkpoint-bound manifests stay out of public Git.
- Do not commit large transient caches, trained weights, or generated datasets unless the user explicitly asks.
- Numbered experiment source and outputs are archival/non-active provenance.
  Do not expose their wrappers through `run.py`, regenerate them as pending
  paper work, or promote them into the active evidence chain.
- Candidate searches, retired model integrations, and competing method/version
  branches are diagnostic or archival unless a separate, protocol-frozen
  promotion plan explicitly replaces the active mainline.

### Experiment status conventions

- Main Results and Mechanism Ablation are the required paper-evidence bundles.
  They must match the active v2.1 dataset, configuration, seed, checkpoint, and
  metric identities.
- Registered robustness and efficiency studies are optional secondary evidence.
  Their presence does not replace or relax Main/Mechanism eligibility gates.
- Historical numbered outputs remain archival/non-active even if individual
  files are readable or happen to share a dataset hash.
- Generate current paper tables only through `python run.py formal paper` and
  eligible evidence bundles. `experiments/generate_paper_artifacts_v2.py` is a
  historical diagnostic generator and must not promote manuscript claims.

## Release Boundary

Before publishing or preparing a public snapshot, check `PublicRelease/README.md`.

Assume the following are private or too large for default publication unless explicitly approved:

- full generated datasets
- large model checkpoints
- local logs and caches
- ad hoc manuscript workspace clutter

## When Updating Documentation

- `README.md` is for humans.
- `AGENTS.md` is for agents and is the authoritative instruction file.
- `CLAUDE.md` should only point Claude-compatible tooling back to `AGENTS.md`.
- `PublicRelease/README.md` should define what is public, what is excluded, and what a reproducer can expect.
