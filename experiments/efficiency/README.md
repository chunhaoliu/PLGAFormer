# Efficiency and Computational Cost

Paper-level non-training benchmark for parameters, FLOPs, batch-1 latency,
throughput, and peak memory under the formal task shape. The canonical
implementation is `experiments/efficiency/efficiency_experiment.py`.

- CLI: `python run.py formal efficiency`
- Legacy CLI/module: `experiments/exp6_efficiency/`
- Input: an eligible formal Main bundle for task shape and performance
  provenance
- Outputs: JSON/CSV benchmark records and optional figures under the preserved
  `experiments/exp6_efficiency/results/` root
- Formal artifact boundary:
  `experiments/exp6_efficiency/results/formal_v3/hgv_multiregime_state_v2_1/`
- Paper mapping: Efficiency

Hardware, software, warm-up, repeat, synchronization, dtype, and batch size
are recorded in the output metadata. This study never trains a model.

Historical non-bundle inputs require the explicit
`--allow-legacy-diagnostic` flag and are never paper-facing.
