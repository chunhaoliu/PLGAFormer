# Confirmatory Holdout Audit

## Decision

**Pass.** The preregistered holdout contains 360 newly simulated complete
trajectories, and all 15 cells (five learning methods by three existing
checkpoint seeds) completed without training, scaler fitting, checkpoint
replacement, trajectory removal, or inference-policy adjustment.

## Evidence identity

- Dataset SHA-256: `691371f35940b252d72261071db907b4e81a3ad9050a305cb7e7b2ff7278dff3`
- Protocol-document SHA-256: `b950b81756be3f10340c9a1b182117b29712ff30dabe7fad99ccd8dda03e1af8`
- Main Results bundle: `82208600652c5cebb249c9563a9b34e28e2a50d66673a71d6ffe3933d1924427`
- Confirmatory bundle: `27089cc27d28d1d705721915aaacba4a4898673b758ebe4b5ac84ecef8a76f50`
- Raw trajectories: `[360, 1000, 6]`
- Evaluation inputs/targets: `[35280, 256, 6]` / `[35280, 256, 3]`
- Six joint strata: exactly 60 complete trajectories each
- Trajectory-ID overlap with the original dataset: zero

The original v2.1 dataset retained SHA-256
`526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`.
Formal Main and Ablation evidence remained paper-eligible with no blockers.

## Overall confirmatory result

Mean +/- population standard deviation across checkpoint seeds 42/123/456,
in kilometres. Lower is better.

| Horizon | PLGAFormer ADE/FDE/RMSE | Second-ranked method | Reduction vs. second (ADE/FDE/RMSE) |
|---:|---:|---|---:|
| 32 s | **0.0356 / 0.0896 / 0.0450** | DLinear | 93.98% / 93.98% / 90.29% |
| 64 s | **0.1447 / 0.4628 / 0.2108** | iTransformer | 92.94% / 82.86% / 85.41% |
| 128 s | **0.7885 / 2.6538 / 0.9464** | iTransformer | 73.78% / 52.17% / 58.82% |
| 256 s | **4.0991 / 13.2387 / 4.3827** | iTransformer | 42.16% / 27.37% / 32.80% |

At 256 s, PLGAFormer standard deviations are 0.1435 km ADE, 0.7616 km
FDE, and 0.0966 km RMSE. PLGAFormer ranks first in all 12 horizon/metric
cells. It also ranks first for both ADE and FDE in longitudinal, turning, and
weaving trajectories at every registered horizon.

## Manuscript decision boundary

The confirmatory values should become the paper's primary learning-method
comparison because this dataset was frozen before inference. The original
360-trajectory test values are development evidence and must not be pooled with
the confirmatory holdout.

The supported claim is narrow: under the preregistered same-simulator-family
protocol, PLGAFormer ranks first among the five evaluated learning methods at
all registered horizons and metrics. This is not evidence for universal SOTA,
real-flight performance, cross-simulator transfer, or methods not evaluated.

## Validation

- Fail-closed confirmatory audit: pass; 15 records; zero provenance blockers.
- Existing formal status/audit: pass; `paper_eligible=true`.
- Repository suite: 305 passed, 3 skipped, 31 subtests passed.
