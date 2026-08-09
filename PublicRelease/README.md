# Public Release Boundary

This directory documents what belongs in a public release of `HGVTP-PLGAformer` and what should stay out unless explicitly approved.

## Goal

Make the repository reproducible enough for external readers to understand the code structure, rerun supported entrypoints, and see the intended experiment protocol without forcing publication of every large or local-only artifact.

## Intended Public Contents

These are generally safe and useful to keep in a public release:

- source code under active use
- top-level entrypoints such as `run.py`
- dependency declarations
- protocol and validation code
- experiment scripts
- lightweight JSON summaries intended as reproducibility artifacts
- paper tables and figures that are part of the documented results
- tests that validate protocol assumptions or artifact generation
- documentation that explains layout, commands, and release scope

## Usually Excluded Unless Explicitly Approved

These should normally stay out of the public release:

- large generated datasets
- raw caches and intermediate arrays
- local logs
- large checkpoints and ad hoc training snapshots
- private manuscript workspace clutter
- machine-specific temporary outputs

## Reproducibility Boundary

External users should be able to:

- inspect the repository structure
- install dependencies
- understand the experiment protocol
- regenerate supported data if the generator is included and permitted
- rerun the documented experiment entrypoints within the limits of their hardware
- run CLI planning checks such as `python run.py --task enhancements --quick --dry-run` without requiring full training dependencies

External users should not assume that a public snapshot includes:

- every internal cache used during development
- every locally generated result file
- every checkpoint from exploratory runs

## Supported Public Entrypoints

- Core experiments: `python run.py --task exp1`, `exp2`, `exp3`, and `exp4`.
- Enhancement experiments: `python run.py --task exp5`, `exp6`, `exp7`, or `python run.py --task enhancements`.
- Full chain: `python run.py --task all`; add `--include-enhancements` to include Exp5-Exp7.
- Fast planning/sanity mode: add `--quick` for Exp5-Exp7 and `--dry-run` when only checking routing.

## Results Boundary

- Exp1-Exp4 formal result summaries are part of the current manuscript artifact set.
- Current historical Exp1 formal summaries cover 32/64/128-step reporting under the older protocol. The formal release protocol is now `1000` complete trajectories, `1000` points per trajectory, `1 Hz`, `seq_len=256`, `pred_len=256`, and `window_stride=5`; a 256-step release requires regenerating the dataset, validating it, and rerunning formal multi-seed Exp1 before using those results as paper evidence.
- Exp5 and Exp7 scripts are public and runnable, but full-scale formal outputs should be regenerated in the final compute environment before using them as definitive paper evidence.
- Exp6 combines fresh lightweight efficiency measurements with formal Exp1 performance references; readers should treat it as an efficiency/profiling artifact, not an independent accuracy experiment.

## Recommended Release Notes

When creating a public snapshot, explicitly state:

- which dataset artifacts are included
- which artifacts must be regenerated locally
- which reported results come from precomputed summaries already committed
- any known hardware or runtime expectations for full reproduction
