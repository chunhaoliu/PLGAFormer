# Formal-v3 command boundary

Use the root coordinator for current formal evidence:

```bash
python run.py formal status --json
python run.py formal validate-data
python run.py formal main [runner options]
python run.py formal mechanism [runner options]
python run.py formal robustness [--main-bundle PATH] [runner options]
python run.py formal efficiency [--main-bundle PATH] [runner options]
python run.py formal ablation --phase phase4_final_mechanism_controls [runner options]  # explicit alias
python run.py formal aggregate --dry-run
python run.py formal aggregate
python run.py formal audit --json
python run.py formal paper --dry-run
python run.py formal paper --robustness PATH.json --efficiency PATH.json
```

`run_formal_sota_unit.py` and `run_formal_ablation_unit.py` remain resumable
training implementations. They write only to the formal-v3 protocol roots and
record the formal configuration hash in newly created run records. The
paper-facing `aggregate` and `paper` commands read bundle manifests through
`utils/formal_evidence.py`; they do not select a latest timestamp or fall back to
`partial_runs.json`.
`formal paper` writes to a temporary sibling directory and publishes only after
all required tables and the unified manifest pass generation.

`main`, `mechanism`, `robustness`, and `efficiency` are the four registered
paper-level studies. The latter two require an eligible Main bundle and expose
read-only `--dry-run` checks. Numbered wrappers below are explicit compatibility
or legacy diagnostics and do not define a paper-evidence path.


The old `experiments/generate_paper_artifacts_v2.py` route and formal-v2 result
folders are retained for historical diagnostics only. Do not use them as the
source of current paper numbers.

# Numbered Experiment Wrappers

These existing wrappers remain available while the numbered experiment
structure is reviewed:

- `python scripts/run_exp1_sota.py`
- `python scripts/run_exp2_ablation.py`
- `python scripts/run_exp3_robustness.py`
- `python scripts/run_exp4_physics_consistency.py`
- `python scripts/run_all_experiments.py`
- `python scripts/run_all_enhancements.py`
- `python scripts/run_regression_chain_smoke.py`

They do not implicitly define the current paper evidence chain. Whether they
should later be retained, moved, or consolidated is intentionally left for the
next project-structure discussion.
