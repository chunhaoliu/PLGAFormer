# Single-mainline migration status

Date: 2026-08-09
Repository: `HGVTP_PLGAformer-main`

The source migration is complete for the four paper studies. The active map is:

| Study | Canonical source | Compatibility boundary |
|---|---|---|
| Overall prediction | `experiments/overall_prediction/main_results.py` | `experiments/exp1_sota/SOTA_comparison.py` |
| Mechanism analysis | `experiments/mechanism_analysis/ablation_study.py` and `physics_consistency.py` | `experiments/exp2_ablation/` and `exp4_physics_consistency/` |
| Generalization and robustness | `experiments/generalization_robustness/dynamics_shift.py` | `scripts/run_taes_dynamics_shift.py` |
| Efficiency | `experiments/efficiency/efficiency_experiment.py` | `experiments/exp6_efficiency/` |

The formal configuration and pipeline now point to the canonical sources. The
old result roots, checkpoint identities, dataset protocol, and run metadata
were not moved or rewritten.

Local checkpoints created during this work:

- `7e84066` — Overall Prediction source and production imports
- `b648dc3` — Mechanism Analysis source and production imports
- `842eecf` — Robustness source, pipeline route, and configuration runner
- `99c1b4b` — Efficiency route and canonical documentation

Validation completed:

- full test suite: `192 passed, 24 warnings`
- Main regression subset: `21 passed`
- mechanism regression subset: `14 passed`
- robustness subset: `8 passed`
- efficiency and formal CLI subset: `12 passed`
- no training, dataset generation, result movement, checkpoint movement, or
  remote Git operation was performed during this source migration

The remaining numbered directories are not yet external-archive candidates:
tests, result readers, visualizers, and legacy smoke routes still reference
some of them. They remain compatibility/history boundaries until a separate
zero-reference audit is complete.
