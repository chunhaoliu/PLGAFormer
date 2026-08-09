# Legacy experiment index

The numbered experiment directories remain in place as compatibility
implementations and historical evidence roots. They are not a second formal
paper route and are not automatically imported by the four registered studies.

| Legacy path | Logical role | Formal status |
|---|---|---|
| `exp1_sota/` | Main comparison implementation | retained implementation; formal Main bundle only |
| `exp2_ablation/` | historical and phase4 ablation implementation | phase4 only for formal mechanism |
| `exp3_robustness/` | old noise/missing/history stress runner | legacy diagnostic; formal definitions live in `formal_v3.json` |
| `exp4_physics_consistency/` | physical evaluator implementation | evaluator input to mechanism study |
| `exp5_missing_data/` | missing-data experiment | legacy, excluded from the formal mainline |
| `exp5_ood_dynamics/` | dynamics-shift output root | consumed by the robustness study |
| `exp6_efficiency/` | efficiency implementation | consumed by the efficiency study |
| `exp7_longterm/` | extended ablation/development search | legacy, excluded from the formal mainline |

No historical code or result is deleted by this index. Any future move to an
outer archive requires a zero-reference audit and separate approval.
