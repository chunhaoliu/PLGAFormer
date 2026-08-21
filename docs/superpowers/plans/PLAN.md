# Active HGV Mainline Plan

The approved implementation plan is:

`docs/superpowers/plans/2026-08-22-hgv-mainline-unification.md`

Execution has not started. The current Main and Ablation evidence remains
protected. Every implementation batch must run its targeted tests and formal
status check before the next batch begins.

## Execution Safety Override

Stage only the exact files changed by the current task. Do not use a broad Git
add command, and do not include unrelated user files. No protected dataset,
formal result, or checkpoint path may be moved, regenerated, or overwritten.

The next project after consolidation is a separate PLGAFormer performance
optimization cycle under the unchanged dataset and evaluation protocol.
