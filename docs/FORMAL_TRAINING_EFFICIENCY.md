# Formal Training Efficiency

This note records implementation-level optimizations for the active
`hgv_multiregime_state_v2_1` protocol. They do not change the dataset, complete
trajectory split, 256/256 input-output interface, training targets, loss,
validation selection rule, reporting horizons, or three formal seeds.

## Applied optimizations

1. The deterministic rotating-Earth physics prior is precomputed once for each
   dataset split and held in CPU memory. Training batches transfer the cached
   prior together with the source and target tensors.
2. Identical prior caches are reused across models and seeds while the same
   dataset object remains alive. This is especially important for the ablation
   runner.
3. The rotating-Earth RK4 implementation uses vectorized substep interpolation,
   persistent atmosphere-table buffers, and one aggregate finite-value check.
4. Gradient finiteness and clipping use one aggregate operation instead of a
   Python loop that synchronizes once per parameter tensor.
5. The formal in-memory `TensorDataset` uses zero DataLoader workers by default.
   On Windows, extra workers added process-startup cost without improving
   steady-state transfer throughput.
6. CUDA autocast with `bfloat16` is available explicitly through `--amp`. It is
   recorded in every protocol identity and result row and is not silently mixed
   with FP32 runs.

The cached and inline prior paths are tested for exact equality
(`rtol=0`, `atol=0`). A cache is keyed by source dataset identity, forecast
length, prior class/configuration, and scaler-buffer hash.

## Measured diagnostic evidence

Environment: NVIDIA GeForce RTX 4090, PyTorch `tslib` environment, Windows,
full v2.1 train and validation tensors, batch size 128. These are engineering
diagnostics, not paper performance results.

| Diagnostic | Result |
|---|---:|
| Formal dataset load and scaling | 4.19 s |
| Train prior cache, 20,160 windows | 59.06 MiB |
| Validation prior cache, 17,640 windows | 51.68 MiB |
| First construction of both caches | 24.29--28.14 s |
| Same-process cache reuse | 0.00 s |
| PLGAFormer forward, inline prior | 342.25 ms/batch |
| PLGAFormer forward, cached prior | 25.51 ms/batch |
| Forward speedup | 13.42x |
| Maximum output difference | 0.0 |

An initial one-epoch diagnostic suggested that BF16 might be slightly faster,
but it did not use the complete formal runner identity and was not used for the
precision decision. The matched validation-only pilots below used the same
formal runner, full train/validation data, seed 42, five-epoch warmup schedule,
batch size 128, and cached prior. Only autocast differed.

| Matched pilot | Epoch total | Mean epoch | Best validation loss |
|---|---:|---:|---:|
| FP32, 5 epochs | 185.66 s | 37.13 s | 0.004426 |
| BF16, 5 epochs | 288.79 s | 57.76 s | 0.004204 |

BF16 epoch time increased from 35.38 s in epoch 1 to 57.25--66.42 s in
epochs 2--5. A separate two-epoch BF16 repeat reproduced the increase
(45.25 s then 64.52 s). Both BF16 runs remained finite, but BF16 was slower
than FP32 in this environment. The single-seed, short-run validation-loss
difference is diagnostic only and is not model-performance evidence.

For the RAM-resident cached training dataset, measured DataLoader transfer
times were:

| Workers | First epoch | Steady epoch |
|---:|---:|---:|
| 0 | 0.246 s | 0.199 s |
| 2 | 2.055 s | 0.198 s |
| 4 | 2.528 s | 0.210 s |
| 8 | 3.394 s | 0.204 s |

Therefore `workers=0` is the formal default on this machine. A different host
may override it explicitly after measuring its own loader.

## Formal launch policy

Use cached FP32 for the active v2.1 formal Main Results and ablations. It is the
faster measured path, has stable epoch time, and preserves continuity with the
existing numeric protocol:

```powershell
D:\ProgramData\anaconda3\envs\tslib\python.exe scripts/run_formal_sota_unit.py `
  --selection-only --skip-existing `
  --models transformer,plgaformer,pit,kinematic,rotating_3dof,dlinear,patchtst,itransformer,af_ciln `
  --seeds 42,123,456 --epochs 50 --batch-size 128 `
  --workers 0 --cache-physics-prior --physics-prior-cache-batch-size 512
```

BF16 remains available for future environment-specific diagnostics through
`--amp --amp-dtype bfloat16`, but it is not the selected formal mode on the
audited RTX 4090/Windows environment. BF16 and FP32 records have different
protocol identities and cannot be reused as one another.

Run ablations in one process per phase so the cache is reused across variants
and seeds:

```powershell
D:\ProgramData\anaconda3\envs\tslib\python.exe scripts/run_formal_ablation_unit.py `
  --selection-only --phase phase1_structural --models all `
  --seeds 42,123,456 --epochs 50 --batch-size 128 `
  --workers 0 --cache-physics-prior --physics-prior-cache-batch-size 512
```

Do not split every ablation model into a separate process unless necessary:
doing so discards the process-local prior cache and repeats its construction.

