# Generalization and Robustness

Paper-level frozen-model study for HGV dynamics shifts and the configured
sensor-noise/short-history stress tests. The current dynamics implementation
is retained at `scripts/run_taes_dynamics_shift.py` until a later physical
directory migration is approved.

- CLI: `python run.py formal robustness`
- Input: an eligible formal Main bundle; `retrain=false`
- Core scenarios: nominal, aerodynamic shift, and ballistic-parameter shift
- Auxiliary scenarios: configured sensor noise and shortened histories;
  random missingness is disabled for the formal route
- Paper mapping: Generalization / Robustness

Every output must retain condition units, perturbation seed, test trajectory
IDs, checkpoint hashes, config hash, and dataset hash.
