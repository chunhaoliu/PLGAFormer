# Single-mainline path audit

Date: 2026-08-09  
Repository: `HGVTP_PLGAformer-main`  
Audit commit: `98c1c64`  
Scope: read-only path and import audit; no training, dataset generation, or
result movement was performed.

## Decision

The repository should expose four paper-level studies only:

1. Overall Prediction Performance
2. Mechanism Ablation and Physical Consistency
3. Generalization and Robustness
4. Efficiency and Deployment Analysis

The numbered experiment directories are historical implementation names, not
additional paper studies. Existing result and checkpoint roots remain
untouched until every reader, manifest, and runner has a verified canonical
replacement.

## Current baseline

- Git worktree was clean before this audit.
- `main` was ahead of `origin/main` by four local commits.
- The frozen dataset, configuration, result records, and retained checkpoints
  were not opened for writing.
- The four logical target directories already exist, but three are currently
  documentation shells and one contains the canonical efficiency source.

## Source and entrypoint mapping

| Paper study | Target source directory | Current implementation | Current formal entry | Migration decision |
|---|---|---|---|---|
| Overall Prediction Performance | `experiments/overall_prediction/` | `experiments/exp1_sota/SOTA_comparison.py` (2,582 lines) | `scripts/run_formal_sota_unit.py` and `run.py formal main` | Move one implementation, preserve the old import as a thin shim, then update the formal runner and tests. |
| Mechanism Ablation and Physical Consistency | `experiments/mechanism_analysis/` | `experiments/exp2_ablation/ablation_study.py` (2,575 lines) plus `mechanism_analysis/physics_consistency.py` | `scripts/run_formal_ablation_unit.py` and `run.py formal mechanism` | Put the ablation engine beside the canonical evaluator; keep old phase and result identities in compatibility metadata. |
| Generalization and Robustness | `experiments/generalization_robustness/` | `scripts/run_taes_dynamics_shift.py` (516 lines) | `run.py formal robustness` | Move the dynamics-shift implementation into the study directory and leave a thin script wrapper. Keep `experiments/exp3_robustness/` as legacy diagnostics. |
| Efficiency and Deployment Analysis | `experiments/efficiency/` | `experiments/efficiency/efficiency_experiment.py` (canonical) | `run.py formal efficiency` | Already canonicalized. Update the formal pipeline import after the other study migrations are stable. |

## Compatibility callers to update

The following active callers still import numbered paths and must be changed
only after the corresponding canonical source has been verified:

- Main: `scripts/run_formal_sota_unit.py`,
  `scripts/evaluate_formal_sota_selection.py`,
  `scripts/run_regression_chain_smoke.py`, and the Main protocol tests.
- Mechanism: `scripts/run_formal_ablation_unit.py`,
  `scripts/evaluate_formal_ablation_selection.py`,
  `scripts/evaluate_final_plgaformer_physical_metrics.py`,
  `scripts/run_plgaformer_candidate_search.py`, and the ablation tests.
- Robustness: `scripts/formal_pipeline.py`, the dynamics-shift test, and the
  old robustness diagnostic tests.
- Efficiency: `scripts/formal_pipeline.py` and the compatibility protocol
  test; the old Exp6 module is already a shim.

Shared readers also contain historical experiment namespaces:
`utils/experiment_io.py`, `utils/experiment_summary.py`,
`utils/formal_evidence.py`, and `utils/final_plgaformer.py`. They should be
updated only with explicit backward-compatible aliases, because they read
existing record schemas and checkpoint metadata.

## Legacy boundary

The following are not new paper studies and should not be reintroduced into
the main CLI or new result tables:

- old robustness, physics-consistency, missing-data, and long-term experiment
  implementations;
- old all-in-one experiment launchers and result visualizers;
- historical paper-artifact generators and quick/smoke/pilot outputs.

They remain available until a zero-reference audit proves that a particular
file can be archived. No deletion is part of this audit step.

## Migration order

1. Canonicalize the Main source and import it through the formal runner.
2. Canonicalize the ablation engine beside the physical evaluator.
3. Canonicalize the dynamics-shift implementation and its script wrapper.
4. Update shared readers, tests, documentation, and the formal pipeline.
5. Run static/import/unit/CLI checks and verify that existing result and
   checkpoint manifests are byte-for-byte unchanged.
6. Only then decide whether the old source directories can be moved to the
   outer archive; compatibility shims stay until the zero-reference audit.

## Non-negotiable boundaries

- Do not retrain, regenerate data, or overwrite existing result artifacts as a
  consequence of a directory migration.
- Do not create a second copy of a large implementation; a canonical move must
  reduce active source duplication.
- Do not change tensor shapes, model construction, data splits, metrics, seeds,
  evaluation horizons, or checkpoint selection.
- Do not mix archived diagnostics with the single paper mainline.

## Acceptance for the next step

The next source migration is ready only when the Main canonical module can be
imported directly, the old path still imports through a thin shim, all formal
unit tests pass, and `git diff` shows no result or checkpoint changes.
