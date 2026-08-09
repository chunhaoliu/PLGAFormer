# Legacy archive audit

Date: 2026-08-09
Scope: source tree only; no deletion or result movement performed.

The four paper studies now have canonical source directories. The remaining
numbered directories cannot yet be moved to the outer Archive because they
still have live compatibility or evidence-reader references.

| Path | Current references | Decision |
|---|---|---|
| `experiments/exp1_sota/` | Main tests, old wrappers, preserved result/checkpoint roots | Keep the thin source shim and result roots |
| `experiments/exp2_ablation/` | Mechanism tests, formal readers, preserved result/checkpoint roots | Keep the thin source shim and result roots |
| `experiments/exp3_robustness/` | Legacy protocol tests, visualizer test, regression smoke, summary readers | Keep as a diagnostic compatibility boundary |
| `experiments/exp4_physics_consistency/` | Direct fixture loader, regression smoke, summary readers | Keep the physics compatibility shim |
| `experiments/exp5_missing_data/` | Missing-data protocol test, legacy task runner, artifact generator | Keep as excluded diagnostic source |
| `experiments/exp5_ood_dynamics/` | Canonical robustness output root and downstream artifact readers | Keep as a result root; do not rename |
| `experiments/exp6_efficiency/` | Efficiency protocol test, legacy benchmark wrapper, preserved output root | Keep the thin compatibility boundary |
| `experiments/exp7_longterm/` | Legacy task runner and enhancement summary | Keep as excluded development source |

The safe cleanup already completed is source de-duplication: the active Main,
mechanism, robustness, and efficiency implementations each have one canonical
copy. The old active imports now resolve through compatibility aliases where
needed.

Archive criteria for a future pass:

1. no test, summary reader, visualizer, wrapper, manifest, or documentation
   depends on the path;
2. preserved outputs have an explicit replacement reader and checksum record;
3. the move is performed as a recoverable operation into the user-specified
   outer Archive;
4. the full test suite passes after the move.

Until all four criteria hold, deleting or moving these directories would reduce
reproducibility rather than clean the project.
