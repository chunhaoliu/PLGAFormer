# Script entry points

The paper-facing route is coordinated by the four formal commands:

```text
python run.py formal main
python run.py formal mechanism
python run.py formal robustness
python run.py formal efficiency
```

The corresponding business logic lives under `experiments/overall_prediction/`,
`experiments/mechanism_analysis/`, `experiments/generalization_robustness/`,
and `experiments/efficiency/`. Formal scripts in this directory are runners,
validators, bundle tools, or thin compatibility wrappers.

Numbered `run_exp*.py`, `run_all*.py`, and regression scripts remain available
for legacy diagnostics. They are not a source of paper evidence and should not
be used to create a second set of study results.
