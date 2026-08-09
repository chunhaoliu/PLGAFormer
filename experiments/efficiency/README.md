# Efficiency and Computational Cost

Paper-level non-training benchmark for parameters, FLOPs, batch-1 latency,
throughput, and peak memory under the formal task shape. The implementation
remains at `experiments/exp6_efficiency/efficiency_experiment.py`.

- CLI: `python run.py formal efficiency`
- Input: an eligible formal Main bundle for task shape and performance
  provenance
- Outputs: JSON/CSV benchmark records and optional figures under the registered
  formal-v3 efficiency root
- Paper mapping: Efficiency

Hardware, software, warm-up, repeat, synchronization, dtype, and batch size
are recorded in the output metadata. This study never trains a model.

Historical non-bundle inputs require the explicit
`--allow-legacy-diagnostic` flag and are never paper-facing.
