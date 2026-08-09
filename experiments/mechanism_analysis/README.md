# Mechanism Ablation and Physical Consistency

Paper-level study for formal-v3 phase4 mechanism controls and physical
consistency evaluation.

## Canonical implementation

- `physics_consistency.py` is the canonical physical-consistency evaluator.
- The phase4 ablation engine and formal records remain under
  `experiments/exp2_ablation/`.
- The old `experiments/exp4_physics_consistency/physics_consistency.py` path is
  retained as a compatibility shim for legacy CLI, tests, and summaries.

## Formal route

- CLI: `python run.py formal mechanism`
- Required phase: `phase4_final_mechanism_controls`
- Input: the eligible formal Main bundle, including reused baseline/full
  identities
- Outputs: matched ablation records, ablation bundle, and evaluator-derived
  physical-consistency evidence
- Paper mapping: Mechanism / Physics

Historical phase1--3 outputs are compatibility diagnostics and cannot satisfy
the formal phase4 gate.
