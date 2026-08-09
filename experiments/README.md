# Experiment layout

The paper has exactly four study entries. Their active implementations are
kept in the following canonical directories:

| Paper study | Formal command | Canonical implementation |
|---|---|---|
| Overall prediction | `python run.py formal main` | `experiments/overall_prediction/main_results.py` |
| Mechanism analysis | `python run.py formal mechanism` | `experiments/mechanism_analysis/ablation_study.py` and `physics_consistency.py` |
| Generalization and robustness | `python run.py formal robustness` | `experiments/generalization_robustness/dynamics_shift.py` |
| Efficiency | `python run.py formal efficiency` | `experiments/efficiency/efficiency_experiment.py` |

The numbered directories are compatibility or historical diagnostic boundaries,
not additional paper studies. Existing result and checkpoint roots retain their
original paths so historical evidence remains traceable while source imports
move to the canonical directories.

Use `python run.py formal status --json` before treating any artifact as
paper evidence. Do not combine compatibility, pilot, smoke, or historical
outputs with the current study records.
