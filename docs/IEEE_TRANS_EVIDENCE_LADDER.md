# IEEE Transactions Evidence Ladder (v2.1)

Goal: publishable TAES/Transactions-quality evidence without using the
formal full matrix as a hyperparameter loop.

Hard rules:

- Active dataset remains `hgv_multiregime_state_v2_1` only.
- Do not rewrite manuscript performance claims from L0--L2.
- Paper-facing numbers come only from L3 `final/` (+ L4 matched ablations).
- Do not change loss, split, horizons, or architecture during L0--L2 unless a
  dedicated design note records the change and L2 is re-run.

## Ladder

| Level | Purpose | Models | Seeds | Data | Train flags | Output tier | May enter paper? |
|---|---|---|---|---|---|---|---|
| L0 | Smoke / runner health | PLGAFormer | 42 | `--subset-ratio 0.1` | `--selection-only --epochs 10 --patience 5 --warmup-epochs 2` | `convergence_pilot` | No |
| L1 | Fast relative screen | Transformer + PLGAFormer | 42 | `--subset-ratio 0.25` | `--selection-only --epochs 30 --patience 8` | `convergence_pilot` | No |
| L2 | Full-data freeze check | Transformer + PLGAFormer | 42 | `1.0` | `--selection-only --epochs 50 --patience 15` | `convergence_pilot` | No (decision only) |
| L3 | Formal Main Results | Full comparator set | 42,123,456 | `1.0` | full train + frozen test | `final` | Yes |
| L4 | Ablations + tables/figures + manuscript sync | Matched ablation phases | 42,123,456 | `1.0` | matched to L3 identity | `formal_v3` ablation + generated artifacts | Yes |

## Promotion gates

- L0 → L1: finishes without crash; finite losses; epoch time sane.
- L1 → L2: PLGAFormer validation objective not worse than Transformer by a
  large margin under the same subset identity; no NaNs.
- L2 → L3: full-data seed-42 selection history looks stable; config frozen.
- L3 → L4: complete three-seed Main Results JSON/CSV under `final/`.
- L4 → submit: tables/figures regenerated only from L3/L4 manifests; manuscript
  compile + visual QA pass.

## Non-goals at L0--L2

- Do not tune on PIT / AF-CILN.
- Do not report subset ADE/FDE as formal performance.
- Do not mix legacy `trajectory_level_v1` or `formal_v2` numbers with v2.1.

## Execution log (local)

- L0 passed 2026-08-08: PLGAFormer seed42, subset 0.1, 10 epochs,
  `formal_sota_seed42_full_20260808_000403`, wall ~79 s, finite losses.
- L1 passed 2026-08-08: seed42 subset 0.25 selection-only;
  Transformer best val `0.02382` @30 vs PLGAFormer `0.00232` @28;
  runs `..._baseline_20260808_000544` and `..._full_20260808_001256`.
- L2 passed 2026-08-08: full-data selection-only seed42;
  Transformer best val `0.04497` @20 (35 epochs) vs PLGAFormer `0.00223` @28
  (43 epochs); runs `..._baseline_20260808_002038` and `..._full_20260808_005104`.
  Ready for L3 only after explicit go-ahead (writes `final/` with test).
- L3 started 2026-08-08: formal Main Results to `final/` with frozen FP32
  protocol. Reuse existing seed42 Transformer/PLGAFormer `final/` JSONs;
  run remaining comparators first, then Transformer/PLGAFormer seeds 123/456,
  then PIT last (known expensive physics-loss path).
- L3 Phase A interrupted 2026-08-08 on missing `HGV_AF_CILN_ROOT`.
  Seed42 completed before failure: kinematic, rotating_3dof, dlinear,
  patchtst, itransformer (plus prior Transformer/PLGAFormer). Resuming with
  audited `--af-ciln-root tmp/AF-CILN`.


## Formal-v3 coordinator audit (2026-08-08)

`python run.py formal status --json` successfully read the active dataset and
23 formal-v3 Main Results records. The dataset hash matched the frozen contract.
The Main Results gate remains blocked by `full:456` and `pit:42/123/456`; the
phase4 spherical-prior and schedule-only ablation units are absent. Until those
records and their reused Main controls are complete, `formal paper` must refuse
numerical artifacts. This status is a tool-produced audit, not a performance
claim.
