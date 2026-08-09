# Experiment layout

The paper-facing work is organized as four studies. The numbered directories
are retained as compatibility implementations because the formal runners,
protocol tests, or legacy CLI still import them.

| Paper-level study | Formal command | Current implementation | Formal output boundary |
|---|---|---|---|
| Overall prediction | `python run.py formal main` | `exp1_sota/` | `exp1_sota/results/formal_v3/` and `exp1_sota/trained_models/formal_v3/` |
| Mechanism / physical consistency | `python run.py formal mechanism` | `exp2_ablation/` plus the physical evaluator | `exp2_ablation/results/formal_v3/` |
| Generalization / robustness | `python run.py formal robustness` | `scripts/run_taes_dynamics_shift.py` | `exp5_ood_dynamics/results/formal_v3/` |
| Efficiency | `python run.py formal efficiency` | `exp6_efficiency/efficiency_experiment.py` | `exp6_efficiency/results/formal_v3/` |

## Numbered compatibility implementations

- `exp1_sota/` and `exp2_ablation/` contain the shared engines used by the
  current formal Main and mechanism runners.
- `exp3_robustness/` and `exp4_physics_consistency/` remain available to the
  regression and diagnostic compatibility routes.
- `exp5_missing_data/` and `exp7_longterm/` are legacy diagnostic studies;
  they are not evidence fallbacks for the formal-v3 paper route.
- `exp5_ood_dynamics/` is the current dynamics-shift result root, while
  `exp6_efficiency/` is the current efficiency implementation.
- `legacy/` records the evidence boundary for the numbered directories.

Historical result files, generated paper bundles, and transient logs are kept
outside the active tree under `D:\Research\HGV_Code\Archive\` with their
original relative paths. The active tree keeps only source, formal-v3 evidence,
and explicitly retained compatibility artifacts.

Use `python run.py formal status --json` to inspect evidence completeness before
using any output in a manuscript. Do not combine archived historical outputs
with formal-v3 records.
