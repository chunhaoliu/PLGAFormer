# Legacy experiment index

The numbered directories below are retained only where tests, result readers,
visualizers, or historical commands still depend on them. They do not define a
second paper route.

| Legacy path | Role |
|---|---|
| `exp1_sota/`, `exp2_ablation/`, `exp6_efficiency/` | Thin compatibility boundaries for the canonical studies |
| `exp3_robustness/` | Older noise, missingness, and short-history diagnostic |
| `exp4_physics_consistency/` | Compatibility boundary for the mechanism evaluator |
| `exp5_missing_data/`, `exp7_longterm/` | Development and diagnostic studies excluded from the paper mainline |
| `exp5_ood_dynamics/` | Preserved robustness result root, not an active source directory |

No historical source or result is deleted during the source migration. A move
to an external archive directory requires a fresh zero-reference audit and a
separate review of tests, summaries, visualizers, and manifests.
