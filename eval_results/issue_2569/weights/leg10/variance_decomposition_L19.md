# Four-way variance decomposition of the L19 answer state (task #2569, leg 10)

Setup: answers are Qwen2.5-7B-Instruct's own sampled generations (1 sampled rollout per context, temperature 1.0, top_p 0.95, max 1024 tokens, seed 42); v_A is the mean over the full response span incl. the 2 template-end tokens (v_x); v_C is the last prompt-token residual state (cx_last), layer 19. Sample: 100,000 single-draw rows ({'lmsys': 61000, 'wildchat': 39000}). Sampling-noise banks: LMSYS 10-draw (same decode recipe), WildChat 5-draw, and the #2617 10-draw probe bank as an off-distribution companion. Definitions: L is the variance fraction carried by the banked linear map from v_C. S is the mean within-prompt variance across rollouts over total variance. W is context information beyond the last-prompt-token state, read as the nearest-neighbor pair intercept minus S. N is the remainder, nonlinearly readable from v_C. The identity L + N + W + S = 1 holds by construction and N + W = 1 - L - S is pinned once S is measured.

## Pooled (variance-weighted over all 3,584 dims)

| piece | fraction |
|---|---|
| L (linear from v_C, leg-2 ceiling) | 0.726 |
| S (sampling noise) | 0.092 |
| W (context beyond v_C) | -0.003 |
| N (nonlinear remainder) | 0.185 |

Banked-map pooled R^2 recomputed on the sample rows: 0.7277.
NN intercept (raw v_C, finest 50% of pairs): 123.0 absolute against a total variance of 1376.8.

## kNN and MLP lower bounds on N

| read | value |
|---|---|
| kNN R^2 (k=5) | 0.6304 (N >= -0.096) |
| kNN R^2 (k=10) | 0.6426 (N >= -0.083) |
| kNN R^2 (k=20) | 0.6384 (N >= -0.088) |
| kNN R^2 (k=50) | 0.6188 (N >= -0.107) |
| #1901 MLP w8192 minus ridge (963k, same rows) | 0.0562 |
| #1901 MLP w32768 minus ridge (963k, same rows) | 0.0592 |

## Per direction

| direction | L | S | W | N |
|---|---|---|---|---|
| refusal_axis_2617 | 0.880 | 0.061 | -0.008 | 0.067 |
| r_B_evil | 0.877 | 0.050 | -0.000 | 0.073 |
| r_B_sycophancy | 0.916 | 0.033 | -0.009 | 0.060 |
| r_B_hallucination | 0.858 | 0.067 | -0.018 | 0.092 |
| answer_PC1 | 0.905 | 0.039 | -0.009 | 0.066 |
| answer_PC2 | 0.907 | 0.042 | -0.009 | 0.060 |
| answer_PC3 | 0.885 | 0.057 | -0.007 | 0.065 |
| answer_PC4 | 0.930 | 0.025 | -0.004 | 0.049 |
| answer_PC5 | 0.915 | 0.030 | -0.003 | 0.058 |
| answer_PC_bottom1 | 0.059 | 0.384 | -0.021 | 0.578 |
| answer_PC_bottom2 | 0.046 | 0.419 | -0.074 | 0.609 |
| answer_PC_bottom3 | 0.053 | 0.407 | -0.037 | 0.577 |
| answer_PC_bottom4 | 0.050 | 0.425 | -0.064 | 0.589 |
| answer_PC_bottom5 | 0.049 | 0.417 | -0.068 | 0.602 |
| worst_R2_dir1 | -0.881 | 0.432 | -0.008 | 1.457 |
| worst_R2_dir2 | -0.465 | 0.434 | -0.047 | 1.077 |
| worst_R2_dir3 | -0.397 | 0.450 | -0.054 | 1.002 |
| worst_R2_dir4 | -0.331 | 0.385 | -0.018 | 0.965 |
| worst_R2_dir5 | -0.303 | 0.437 | -0.075 | 0.940 |
| worst_R2_dir6 | -0.239 | 0.433 | -0.042 | 0.848 |
| worst_R2_dir7 | -0.218 | 0.411 | -0.040 | 0.847 |
| worst_R2_dir8 | -0.192 | 0.440 | -0.045 | 0.797 |
| worst_R2_dir9 | -0.187 | 0.435 | -0.035 | 0.787 |
| worst_R2_dir10 | -0.174 | 0.443 | -0.063 | 0.794 |

## Caveats

- #2091-family rollouts carry a 19.5% token-cap hit rate and 3-9% off-language drift; both inflate S
- the NN intercept is an extrapolation in 3,584 dimensions; k-th neighbor curves shown for k=1,2,5,10
- pooled L is the leg-2 population ceiling on the full 963k pool; the sample rows are map-training rows (optimism bounded by the 0.726 vs 0.719 ceiling-vs-heldout gap)
- S transfers from separate context banks (LMSYS stoch10 + WildChat rung), mixed by the sample corpus mix
