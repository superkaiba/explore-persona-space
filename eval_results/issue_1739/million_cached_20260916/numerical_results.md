# Cached million-map numerical results

Frozen layer 19 map trained on 963,444 generic pairs. Five natural datasets; WildChat retains4 examples per behavior and is excluded from inferential conclusions. All intervals are paired 2,000-draw group bootstraps, pointwise and conditional on the fitted maps/directions/readouts. No multiplicity correction or equivalence test.

## Matched behavior regressions

| Dataset | n | Raw context | Context covariance | True map | Shuffled maps (mean rho) |
|---|---:|---:|---:|---:|---:|
| HH | 1847 | 0.0822 | 0.0866 | 0.0700 | 0.0713 |
| ToxicChat | 370 | 0.3888 | 0.3291 | 0.3913 | 0.3616 |
| AITA | 1304 | 0.7598 | 0.7300 | 0.7457 | 0.7470 |
| NQ | 3164 | 0.4260 | 0.4259 | 0.4311 | 0.4189 |
| SimpleQA | 4021 | 0.3580 | 0.3925 | 0.3330 | 0.3575 |

| Dataset | Map minus raw (95% CI) | Map minus covariance (95% CI) | Map minus shuffled mean (95% CI) |
|---|---|---|---|
| HH | -0.0123 [-0.0447, +0.0212] | -0.0167 [-0.0580, +0.0289] | -0.0013 [-0.0358, +0.0344] |
| ToxicChat | +0.0025 [-0.0335, +0.0353] | +0.0623 [-0.0230, +0.1536] | +0.0297 [-0.0152, +0.0751] |
| AITA | -0.0141 [-0.0237, -0.0047] | +0.0158 [+0.0029, +0.0279] | -0.0013 [-0.0115, +0.0090] |
| NQ | +0.0051 [-0.0025, +0.0135] | +0.0052 [-0.0077, +0.0190] | +0.0122 [+0.0027, +0.0221] |
| SimpleQA | -0.0250 [-0.0344, -0.0154] | -0.0594 [-0.0752, -0.0446] | -0.0245 [-0.0348, -0.0142] |

Covariance uses context-only sufficient statistics from exactly the 963,444 map-training rows; no answer vectors or behavior labels enter it. Shrinkage .01 was selected by 400 generic validation contexts from [.01,.05,.1,.3], at the lower grid boundary. Each readout uses the same historical selected training IDs after normalized-content overlap removal, the same pool-standardized targets and GCV lambda grid .01–1e6. All selected readout lambdas are interior. Shuffled transformations can preserve context information, so these are conditioning controls, not chance-level predictors.

## Fixed directions: no behavior-specific regression

| Dataset | Answer direction on real answers | Answer direction on predicted answers | Context-native | Answer direction on raw contexts | Regularized preimage |
|---|---:|---:|---:|---:|---:|
| HH | 0.0723 | 0.0578 | 0.0009 | 0.0421 | 0.0510 |
| ToxicChat | 0.4399 | 0.4260 | 0.4032 | 0.4489 | 0.4100 |
| AITA | 0.2153 | 0.2549 | 0.3187 | 0.1865 | 0.3218 |
| NQ | -0.0090 | -0.0846 | -0.1269 | -0.1328 | -0.1713 |
| SimpleQA | 0.6050 | 0.4492 | 0.5065 | 0.4641 | 0.4346 |

## Inverse sensitivity (diagnostic; no selection on these labels)

| Dataset | Rank 189 | Rank 378 (selected) | Rank 756 | Full rank 3584 |
|---|---:|---:|---:|---:|
| HH | 0.0393 | 0.0510 | 0.0414 | -0.0274 |
| ToxicChat | 0.4057 | 0.4100 | 0.4076 | 0.0365 |
| AITA | 0.3127 | 0.3218 | 0.2967 | -0.2212 |
| NQ | -0.1062 | -0.1713 | -0.1620 | -0.0478 |
| SimpleQA | 0.4714 | 0.4346 | 0.4329 | 0.5407 |

Rank 378 was chosen from all 0–3584 ranks by reconstruction of standardized generic validation contexts from actual answer vectors. Test reconstruction R²=0.01197; training-mean baseline R²=−0.02271. Behavior-direction residual fractions after forward-mapping the inverse direction are .366 evil, .363 sycophancy, .292 hallucination. Full inversion improves SimpleQA point correlation but collapses ToxicChat and reverses AITA; selecting that variant on SimpleQA would be post-hoc behavior-specific tuning.

## Held-out preimage cosine tails

Top/bottom deciles are ranked by cosine in centered standardized context coordinates, with stable context-ID tie breaking. Evil/sycophancy values are saved 0–100 judge scores; hallucination is the saved fraction among decided responses. These are descriptive means, not causal effects or unconditional rollout incidence.

| Dataset | Tail n | Bottom mean | Top mean | Bottom kept/observed rollouts | Top kept/observed rollouts |
|---|---:|---:|---:|---:|---:|
| HH | 184 | 0.0290 | 0.0519 | 920/920 | 627/920 |
| ToxicChat | 37 | 0.0000 | 19.7390 | 183/185 | 168/185 |
| AITA | 130 | 13.3071 | 26.3769 | 649/650 | 650/650 |
| NQ | 316 | 0.4753 | 0.3880 | 1580/1580 | 1580/1580 |
| SimpleQA | 402 | 0.3219 | 0.8806 | 2010/2010 | 2004/2010 |

## Scope and verification

The direction contrasts are unfiltered positive/negative instruction contrasts, fixed at layer 19. There is no behavior-label layer, sign, inverse-rank or generic-covariance selection. The frozen map averages completion plus closing-template tokens; evaluation answer activations average completion only. This accepted pooling mismatch remains. Only4WildChat contexts per behavior survive prior content-overlap filtering; no meaningful WildChat or refusal result is claimed.

Independent numerical review checked all 3 behaviors against SciPy correlations, paired bootstrap draws/intervals and prior fixed evaluation IDs/DVs. Scores reproduce prior fixed-direction results exactly. No remaining retained training/evaluation normalized-content overlap was found. Four focused algebra/undefined-cell tests passed. Repository-wide mapped tests: 29 passed, 1 unrelated pre-existing thread-cap failure; no-flags workflow lint completed with 30 pre-existing errors, none attributed to this round’s new scripts.


## Reproducibility

Source: `2cf01b584b922c0df3db7405feffc6666f2ce60d`. [Immutable archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/449c48d4047efffec50dbe8492d78c905b4bb588/issue1739_million_cached_20260916): all 99 files (527,619,423 bytes) verified by remote hashes. It includes per-context predictions, bootstrap draws, transforms, all 30top/bottom prompts per method/behavior/metric, held-out saved completion panels, source and input provenance. Computation used CPU only; source-frozen supervised run completed successfully in about 17 minutes including upload. The independent watchdog acknowledged the completion alert, with no recovery needed. Completed-run services were disabled after verification.
