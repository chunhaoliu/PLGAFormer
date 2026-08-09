# Mechanism Ablation and Physical Consistency

Paper-level study for formal-v3 phase4 mechanism controls and physical
consistency evaluation. The phase4 runner and records remain under
`experiments/exp2_ablation/`; this directory is a logical registry entry only.

- CLI: `python run.py formal mechanism`
- Required phase: `phase4_final_mechanism_controls`
- Input: the eligible formal Main bundle, including reused baseline/full
  identities
- Outputs: matched ablation records, ablation bundle, and evaluator-derived
  physical-consistency evidence
- Paper mapping: Mechanism / Physics

Historical phase1--3 outputs are compatibility diagnostics and cannot satisfy
the formal phase4 gate.
