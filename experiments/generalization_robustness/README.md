# Generalization and robustness

This directory contains the paper's single frozen-model generalization and robustness study.

- Canonical dynamics-shift evaluator: `experiments/generalization_robustness/dynamics_shift.py`
- Formal entry point: `python run.py formal robustness`
- Core conditions: nominal, aerodynamic-parameter shift, and ballistic-parameter shift
- Inputs: an eligible Main evidence bundle and frozen checkpoints; the evaluator does not retrain models
- Existing result roots, checkpoint identities, and artifact metadata remain unchanged during this source migration.

The older `experiments/exp3_robustness/` implementation is retained only as a historical diagnostic boundary. It is not an additional paper study.
