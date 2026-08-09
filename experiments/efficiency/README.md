# Efficiency

This directory contains the paper's single efficiency and deployment-cost study.

- Canonical benchmark: `experiments/efficiency/efficiency_experiment.py`
- Formal entry point: `python run.py formal efficiency`
- Measures: parameter count, FLOPs, batch-1 latency, throughput, and peak memory under the fixed task shape
- Input: an eligible Main evidence bundle for task-shape and performance provenance
- Existing output roots and benchmark metadata remain unchanged during this route cleanup.

The old `experiments/exp6_efficiency/` path is retained only as a compatibility boundary for legacy imports and commands. It is not a separate paper study. This benchmark does not train models.
