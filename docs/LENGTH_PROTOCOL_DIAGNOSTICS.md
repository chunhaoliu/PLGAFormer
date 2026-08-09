# Observation and Forecast Length Protocol

## 1. Paper-facing main task

The formal paper task is one direct Cartesian forecast:

- observation length: 256 samples = 256 s;
- forecast length: 256 samples = 256 s;
- reporting checkpoints: 32, 64, 128, and 256 s.

The four checkpoints are prefixes of the **same directly generated 256-step
forecast**. They are not four independently trained prediction tasks. Tables
and captions must therefore use wording such as “selected lead-time
checkpoints within a direct 256-s forecast.”

This design tests whether one model can retain accuracy over the complete long
forecast rather than giving every reporting horizon a separately optimized
model.

## 2. Input-length ablation

The controlled input-length ablation uses:

| Observation | Direct forecast | Reported checkpoints |
|---:|---:|---|
| 64 s | 256 s | 32, 64, 128, 256 s |
| 128 s | 256 s | 32, 64, 128, 256 s |
| 256 s | 256 s | 32, 64, 128, 256 s |

Only the supervised window view changes. The simulator artifact, complete
trajectory IDs, train/validation/test membership, maneuver labels, sampling
interval, and physical trajectories remain frozen.

## 3. Horizon-specific diagnostic

To determine whether the main-task advantage comes only from training one
long output, run a bounded diagnostic:

| Observation | Separately trained forecast | Reported endpoint |
|---:|---:|---:|
| 256 s | 64 s | 64 s |
| 256 s | 128 s | 128 s |
| 256 s | 256 s | 256 s |

Compare each endpoint with the corresponding 64/128/256-s prefix from the main
256-s model. This diagnostic is initially seed 42 for PLGAFormer, Transformer,
and the strongest learned baseline identified by the main table. Promote it to
three seeds only if model ranking or the paper conclusion changes.

## 4. Deriving window views

The command below rebuilds windows from the complete trajectories in the
frozen `hgv_multiregime_state_v2_1` artifact. It does not re-simulate data:

```powershell
python scripts/derive_multiregime_window_view.py `
  --seq-len 64 `
  --pred-len 256
```

Create the remaining views by changing the two lengths:

- input ablation: `(64,256)`, `(128,256)`, `(256,256)`;
- output diagnostic: `(256,64)`, `(256,128)`, `(256,256)`.

The original `(256,256)` artifact remains the canonical main-task file and
does not need to be duplicated. Every derived artifact records:

- exact parent dataset SHA-256;
- parent dataset protocol;
- derived window-view ID;
- complete-trajectory split counts;
- supervised-window counts;
- post-write trajectory-disjointness validation.

### Generated views (2026-08-04)

All four views inherit parent SHA-256
`526d50d05b14ed17f9093249cbb5a382f526468e36723457c08c790d449112e7`
and preserve the 1,260/180/360 complete-trajectory split.

| View | Artifact SHA-256 | Train/validation/test windows |
|---|---|---:|
| `64 -> 256` | `8e2b344601fa8dbce96e101d8ef84b8c872600c23d94815bbd53548703e3f563` | 20,160 / 24,660 / 49,320 |
| `128 -> 256` | `af22600b8ace00de73d647b54f5d3dab5b08e4cf961b8afdc6b39589b77fa84d` | 20,160 / 22,320 / 44,640 |
| `256 -> 64` | `b0553c4cf9f6d27fb5b3ebef44ad42543db21f8d9d95726492ac29c6f50bad7b` | 20,160 / 24,660 / 49,320 |
| `256 -> 128` | `4cc1335d1d72e206925b3bf14c45974ce29062a26fe193ade263a41636156466` | 20,160 / 22,320 / 44,640 |

## 5. Running a diagnostic

Example input-length validation-selection run:

```powershell
python scripts/run_length_protocol_diagnostic.py `
  --dataset-path data_generation/data/processed/derived_window_views/hgv_multiregime_state_v2_1_obs64_pred256.npz `
  --task-kind input-ablation `
  --models plgaformer,transformer `
  --seeds 42 `
  --selection-only `
  --epochs 50 `
  --batch-size 128 `
  --no-amp
```

Example separately trained 128-s output diagnostic:

```powershell
python scripts/run_length_protocol_diagnostic.py `
  --dataset-path data_generation/data/processed/derived_window_views/hgv_multiregime_state_v2_1_obs256_pred128.npz `
  --task-kind output-diagnostic `
  --models plgaformer,transformer `
  --seeds 42 `
  --selection-only `
  --epochs 50 `
  --batch-size 128 `
  --no-amp
```

The wrapper reads `seq_len` and `pred_len` from the NPZ, cross-checks them
against stored array shapes, and pins the model configuration before importing
the formal runner. A mismatched main, input-ablation, or output-diagnostic task
fails before training.

## 6. Evidence boundary

- The main table remains three seeds on the canonical `(256,256)` task.
- Input-length ablation is paper-facing only after matched seeds and fixed
  hyperparameters.
- Horizon-specific runs are diagnostic until their conclusion is stable.
- Prefix metrics and horizon-specific metrics must never be merged as if they
  came from the same training protocol.
- All uncertainty and significance calculations use complete trajectories as
  independent statistical units; windows are within-trajectory repeated
  observations.
