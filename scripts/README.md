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

The confirmatory route is explicit and separate from model development:

```text
python scripts/generate_confirmatory_holdout.py --help
python scripts/evaluate_confirmatory_holdout.py --preflight-only
python scripts/evaluate_confirmatory_holdout.py --audit-only
```

`plot_information_fusion_figures.py` redraws the submission figures from
explicit frozen source files and writes a SHA-256 manifest. The supporting
`plot_taes_*` scripts generate or validate deterministic source arrays and
diagnostic panels; they do not train models or promote new paper results.

Inactive numbered launchers and candidate-search scripts have been removed
from the active mainline. Archived historical code is not a source of paper
evidence and must not be used to create a second result set.
