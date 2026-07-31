# n<d ridge-selector audit — #1310 / #1639 corrections table

Layer 19, 5-fold group-held-out, seed 0, ambient d = 3584. Lambda grid 0.01-10000 (13 points). Materiality threshold 0.05 R^2 or a sign change.

**Store gap.** The run-2 SCRIPT-format activation store backing the published uncapped per-persona cells (eval_results/issue_1310/cells_*.json, no gcv_dof_cap field) was lost with its instance; those cells cannot be recomputed at 0 GPU-h.

`published` = the committed `r2_per_layer_obs[19]` (capped GCV, `gcv_dof_cap: 0.9`); the audit reproduces it in the `ref` arm before any other read. `ambient` reproduces the SELECTOR the lost run-2 script cells used. `forced lambda` arms are selection-bearing diagnostics, never headlines.

| cell | family | n | n_train | published | reproduced ref | ambient pure-GCV | inner-group-CV | reduced PCA basis | ambient verdict | published verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `onpolicy_base_Wren` | per_turn_prefill | 1535 | 1228 | +0.1140 | +0.1140 | -4.4101 | +0.3595 | -0.0369 | artifact-deflated | published-deflated |
| `onpolicy_base_HELIOS` | per_turn_prefill | 1402 | 1122 | +0.0582 | +0.0582 | -2.7094 | +0.3350 | +0.0161 | artifact-deflated | published-deflated |
| `onpolicy_base_Dana` | per_turn_prefill | 1577 | 1262 | +0.1505 | +0.1505 | -1.5633 | +0.3772 | +0.0240 | artifact-deflated | published-deflated |
| `onpolicy_base_Vex` | per_turn_prefill | 1701 | 1361 | -0.0210 | -0.0210 | -5.0447 | +0.3088 | -0.2033 | artifact-deflated | published-deflated |
| `agg_base_Wren` | scene_aggregated | 300 | 240 | +0.1965 | +0.1965 | -0.3067 | +0.1965 | +0.1379 | artifact-deflated | robust |
| `agg_base_HELIOS` | scene_aggregated | 300 | 240 | +0.1975 | +0.1975 | -0.2858 | +0.1975 | +0.1333 | artifact-deflated | robust |
| `agg_base_Dana` | scene_aggregated | 300 | 240 | +0.2178 | +0.2178 | -0.2084 | +0.2178 | +0.1629 | artifact-deflated | robust |
| `agg_base_Vex` | scene_aggregated | 300 | 240 | +0.1327 | +0.1327 | -0.4406 | +0.1327 | +0.0918 | artifact-deflated | robust |
| `onpolicy_instruct_Wren` | per_turn_prefill | 1798 | 1438 | -0.1783 | -0.1783 | -5.4321 | +0.2760 | -0.5574 | artifact-deflated | published-deflated |
| `onpolicy_instruct_HELIOS` | per_turn_prefill | 1800 | 1440 | -0.0996 | -0.0996 | -4.8889 | +0.3134 | -0.4338 | artifact-deflated | published-deflated |
| `onpolicy_instruct_Dana` | per_turn_prefill | 1738 | 1390 | -0.1888 | -0.1888 | -5.4417 | +0.2512 | -0.5240 | artifact-deflated | published-deflated |
| `onpolicy_instruct_Vex` | per_turn_prefill | 1798 | 1438 | -0.1762 | -0.1762 | -5.5084 | +0.2366 | -0.2098 | artifact-deflated | published-deflated |
| `agg_instruct_Wren` | scene_aggregated | 300 | 240 | +0.3548 | +0.3548 | -0.0626 | +0.3548 | +0.2758 | artifact-deflated | robust |
| `agg_instruct_HELIOS` | scene_aggregated | 300 | 240 | +0.4011 | +0.4011 | +0.0830 | +0.4011 | +0.3295 | artifact-deflated | robust |
| `agg_instruct_Dana` | scene_aggregated | 300 | 240 | +0.3286 | +0.3286 | -0.0368 | +0.3261 | +0.2562 | artifact-deflated | robust |
| `agg_instruct_Vex` | scene_aggregated | 300 | 240 | +0.2329 | +0.2329 | -0.2548 | +0.2329 | +0.1539 | artifact-deflated | robust |

## Selected lambda per arm (grid-edge proximity)

| cell | ambient: median lambda | folds at grid floor (0.01) | capped: median lambda | inner-CV: median lambda |
|---|---|---|---|---|
| `onpolicy_base_Wren` | 0.01 | 5/5 | 100 | 3162.28 |
| `onpolicy_base_HELIOS` | 0.01 | 5/5 | 100 | 3162.28 |
| `onpolicy_base_Dana` | 100 | 2/5 | 100 | 3162.28 |
| `onpolicy_base_Vex` | 0.01 | 5/5 | 100 | 3162.28 |
| `agg_base_Wren` | 0.01 | 5/5 | 10000 | 10000 |
| `agg_base_HELIOS` | 0.01 | 5/5 | 10000 | 10000 |
| `agg_base_Dana` | 0.01 | 5/5 | 10000 | 10000 |
| `agg_base_Vex` | 0.01 | 5/5 | 10000 | 10000 |
| `onpolicy_instruct_Wren` | 0.01 | 5/5 | 100 | 3162.28 |
| `onpolicy_instruct_HELIOS` | 0.01 | 5/5 | 100 | 3162.28 |
| `onpolicy_instruct_Dana` | 0.01 | 5/5 | 100 | 3162.28 |
| `onpolicy_instruct_Vex` | 0.01 | 5/5 | 100 | 10000 |
| `agg_instruct_Wren` | 0.01 | 5/5 | 3162.28 | 3162.28 |
| `agg_instruct_HELIOS` | 0.01 | 5/5 | 3162.28 | 3162.28 |
| `agg_instruct_Dana` | 0.01 | 5/5 | 3162.28 | 3162.28 |
| `agg_instruct_Vex` | 0.01 | 5/5 | 10000 | 10000 |

## Forced-lambda diagnostic reads

| cell | lambda 1e2 | lambda 1e3 | lambda 1e4 |
|---|---|---|---|
| `onpolicy_base_Wren` | +0.1140 | +0.3475 | +0.3418 |
| `onpolicy_base_HELIOS` | +0.0582 | +0.3166 | +0.3226 |
| `onpolicy_base_Dana` | +0.1505 | +0.3690 | +0.3550 |
| `onpolicy_base_Vex` | -0.0210 | +0.2855 | +0.2967 |
| `agg_base_Wren` | +0.0244 | +0.1636 | +0.1965 |
| `agg_base_HELIOS` | +0.0252 | +0.1573 | +0.1975 |
| `agg_base_Dana` | +0.0745 | +0.1934 | +0.2178 |
| `agg_base_Vex` | -0.0677 | +0.0841 | +0.1327 |
| `onpolicy_instruct_Wren` | -0.1783 | +0.2393 | +0.2716 |
| `onpolicy_instruct_HELIOS` | -0.0996 | +0.2841 | +0.3022 |
| `onpolicy_instruct_Dana` | -0.1888 | +0.2173 | +0.2434 |
| `onpolicy_instruct_Vex` | -0.3095 | +0.1819 | +0.2366 |
| `agg_instruct_Wren` | +0.1982 | +0.3316 | +0.3498 |
| `agg_instruct_HELIOS` | +0.2846 | +0.3848 | +0.3907 |
| `agg_instruct_Dana` | +0.1970 | +0.3091 | +0.3241 |
| `agg_instruct_Vex` | +0.0488 | +0.1927 | +0.2329 |

## Mapping baselines (ambient space; standing dual-read rule)

| cell | identity+learned-bias $R^2$ | kNN acc@1 (capped GCV) | kNN acc@1 (ambient) | chance acc@1 |
|---|---|---|---|---|
| `onpolicy_base_Wren` | -1.3997 | 0.202 | 0.097 | 0.0007 |
| `onpolicy_base_HELIOS` | -1.7176 | 0.157 | 0.107 | 0.0007 |
| `onpolicy_base_Dana` | -1.7022 | 0.211 | 0.165 | 0.0006 |
| `onpolicy_base_Vex` | -1.5874 | 0.133 | 0.054 | 0.0006 |
| `agg_base_Wren` | -0.0819 | 0.083 | 0.060 | 0.0033 |
| `agg_base_HELIOS` | -0.0343 | 0.060 | 0.060 | 0.0033 |
| `agg_base_Dana` | -0.0990 | 0.087 | 0.080 | 0.0033 |
| `agg_base_Vex` | -0.1408 | 0.050 | 0.037 | 0.0033 |
| `onpolicy_instruct_Wren` | -0.1934 | 0.093 | 0.037 | 0.0006 |
| `onpolicy_instruct_HELIOS` | -0.1704 | 0.098 | 0.033 | 0.0006 |
| `onpolicy_instruct_Dana` | -0.1917 | 0.102 | 0.048 | 0.0006 |
| `onpolicy_instruct_Vex` | -0.3458 | 0.058 | 0.021 | 0.0006 |
| `agg_instruct_Wren` | +0.0931 | 0.170 | 0.083 | 0.0033 |
| `agg_instruct_HELIOS` | +0.0967 | 0.200 | 0.087 | 0.0033 |
| `agg_instruct_Dana` | +0.0717 | 0.110 | 0.070 | 0.0033 |
| `agg_instruct_Vex` | +0.0336 | 0.107 | 0.033 | 0.0033 |
