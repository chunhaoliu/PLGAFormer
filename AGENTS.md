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
- `scripts/`: auxiliary and compatibility experiment wrappers
- `models/`: source-of-truth model implementations and config
- `tests/`: protocol, metrics, and artifact tests
- `docs/`: audits, planning notes, and manuscript/release readiness records
- `PublicRelease/`: public release policy, reproducibility scope, and release boundaries

## Current Formal Commands

Run from repository root unless the user explicitly asks otherwise.

- Install deps: `pip install -r requirements.txt`
- Read-only status: `python run.py formal status --json`
- Validate frozen data: `python run.py formal validate-data`
- Read-only evidence audit: `python run.py formal audit --json`
- Main Results runner: `python run.py formal main [explicit runner options]`
- Mechanism/physics runner: `python run.py formal mechanism [explicit runner options]`
- Generalization/robustness runner: `python run.py formal robustness [explicit runner options]`
- Efficiency runner: `python run.py formal efficiency [explicit runner options]`
- Explicit ablation compatibility alias: `python run.py formal ablation --phase phase4_final_mechanism_controls [explicit runner options]`
- Evidence aggregation: `python run.py formal aggregate --dry-run`
- Paper staging: `python run.py formal paper --dry-run`

Running `python run.py` without a command prints help and performs no work.
The numbered `python run.py --task ...` workflow remains available only as an
explicit compatibility/legacy route; it is not an implicit paper-evidence
route.

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
- Preserve the existing model-construction paths until their later structure
  review; do not create another registry or configuration source meanwhile.

### Results and artifacts

- Keep lightweight JSON, PNG, PDF, and release-safe tables under version control when they are intended as reproducibility artifacts.
- Do not commit large transient caches, trained weights, or generated datasets unless the user explicitly asks.
- Keep `experiments/exp5_missing_data/run_missing_data.py` and `experiments/exp6_efficiency/run_benchmark.py` as compatibility wrappers only; the canonical implementations are `missing_data_experiment.py` and `efficiency_experiment.py`.
- Prefer `main(argv=None)` for experiment scripts that are called from `run.py`, so the root CLI can pass `--quick` safely.

### Experiment status conventions

- Treat Exp1-Exp4 committed outputs as historical unless their run identity
  matches the active v2.1 dataset hash.
- Treat historical Exp1 outputs as older-protocol evidence only. Do not claim
  v2.1 performance until matched multi-seed Main Results and ablations exist.
- Generate current paper tables only through `python run.py formal paper` and
  eligible evidence bundles. `experiments/generate_paper_artifacts_v2.py` is a
  historical diagnostic generator and must not promote manuscript claims.
- Treat Exp5 and Exp7 as runnable but pending formal full-scale regeneration until their `results/` directories contain final JSON/CSV/figure artifacts from the target compute environment.
- Treat Exp6 as a lightweight benchmark that can be regenerated quickly, but do not replace manuscript performance claims without checking the formal Exp1 source values.

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
