# Character-shift rank and query dependence

D has one row per literal matched query and columns for representation coordinates; D=target-source. Raw and mean-centered spectra use squared singular values. All six unordered character pairs, two models, context and K5-mean answers. Full-cohort descriptive spectra; train-only subspaces projected onto observed held-out differences under five global conversation folds.

For each character pair, D[q,:] is the difference for the same full query. If every query receives exactly one fixed vector b, raw D has rank one and centered D has rank zero. Rank one alone allows a query-dependent signed amplitude. We therefore measure both spectral dimensionality and deviation from the training-mean vector.

## Query constancy

The Fixed vector and Same line percentages use total squared raw shift as denominator. Entries are unweighted means across the six character pairs. Fixed b is estimated on four global conversation folds and evaluated on the fifth. The variable-strength column is an oracle projection of the observed held-out difference onto the line spanned by b. Its signed coefficients use the target, so it measures geometric coverage rather than prediction. Perpendicular residual instead uses total fixed-vector residual energy as denominator: it is the fraction of fixed-vector error that remains even after allowing signed strength to vary.

| Model | Representation | Fixed vector | Same line, varying signed strength | Perpendicular share of fixed-vector residual | Mean of pair-median cosines to b |
|---|---|---:|---:|---:|---:|
| qwen2.5-7b | context | 17.74% | 20.41% | 96.72% | 0.441 |
| qwen2.5-7b | answer | 10.17% | 14.24% | 95.46% | 0.323 |
| qwen2.5-7b-instruct | context | 18.62% | 21.41% | 96.55% | 0.452 |
| qwen2.5-7b-instruct | answer | 11.46% | 16.11% | 94.72% | 0.353 |

## Spectral rank

Spectral energy is sigma_i squared. r90 is the smallest number of directions capturing 90% of the observed squared energy, not algebraic or noise-corrected population rank. Stable rank = sum(energy)/max(energy); participation ratio = sum(energy)^2/sum(energy^2). Full spectra are descriptive on all retained queries. Held-out projection curves learn their bases on the other four folds and use the observed held-out difference to measure coverage; no unseen-persona or prompt-family claim. Ranges below are the min/max across six pairs.

| Model | Representation | Spectrum | Top direction (mean) | Top 10 (mean) | r50 range | r90 range | r95 range | Participation ratio range | Held-out top 10 (mean) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b | context | raw | 21.24% | 46.71% | 9–22 | 322–387 | 541–607 | 10–31 | 45.47% |
| qwen2.5-7b | context | centered | 11.75% | 35.52% | 25–31 | 391–431 | 604–657 | 36–57 | 34.02% |
| qwen2.5-7b | answer | raw | 15.03% | 40.82% | 17–26 | 356–405 | 569–623 | 21–46 | 39.28% |
| qwen2.5-7b | answer | centered | 7.86% | 34.43% | 27–30 | 394–419 | 604–645 | 53–56 | 32.76% |
| qwen2.5-7b-instruct | context | raw | 22.17% | 45.42% | 10–25 | 359–423 | 580–645 | 9–31 | 44.16% |
| qwen2.5-7b-instruct | context | centered | 11.99% | 33.20% | 29–37 | 427–474 | 642–701 | 36–66 | 31.65% |
| qwen2.5-7b-instruct | answer | raw | 17.25% | 43.63% | 14–21 | 327–375 | 538–591 | 18–36 | 42.04% |
| qwen2.5-7b-instruct | answer | centered | 10.02% | 36.59% | 22–25 | 365–394 | 572–614 | 41–51 | 34.77% |

## Rank of mean shifts between characters

Four character means are estimated on exactly the same all-character query intersection. Their centered contrast matrix has rank at most three by construction. The six pairwise mean differences have the same normalized spectrum; this identity is numerically verified. A small mean-contrast rank does not establish that each query has the same shift.

| Model | Representation | Common queries | First direction | First two directions | r95 |
|---|---|---:|---:|---:|---:|
| qwen2.5-7b | context | 495 | 65.00% | 90.18% | 3 |
| qwen2.5-7b | answer | 495 | 50.57% | 90.86% | 3 |
| qwen2.5-7b-instruct | context | 496 | 61.84% | 90.01% | 3 |
| qwen2.5-7b-instruct | answer | 496 | 47.33% | 90.32% | 3 |

## Per-pair values

| Model | Representation | Pair | Queries | Fixed vector (%) | Varying strength (%) | Raw r90 | Centered r90 | Centered held-out top 10 (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-7b | context | HELIOS ↔ Wren | 1431 | 22.64 | 25.84 | 335 | 421 | 34.81 |
| qwen2.5-7b | answer | HELIOS ↔ Wren | 1431 | 9.15 | 13.00 | 385 | 417 | 33.46 |
| qwen2.5-7b | context | HELIOS ↔ Dana | 1390 | 28.38 | 30.89 | 322 | 431 | 31.87 |
| qwen2.5-7b | answer | HELIOS ↔ Dana | 1390 | 13.64 | 18.51 | 371 | 419 | 32.97 |
| qwen2.5-7b | context | HELIOS ↔ Vex | 1313 | 22.05 | 24.79 | 324 | 402 | 34.00 |
| qwen2.5-7b | answer | HELIOS ↔ Vex | 1313 | 13.76 | 17.86 | 356 | 401 | 32.57 |
| qwen2.5-7b | context | Wren ↔ Dana | 1337 | 7.89 | 11.02 | 387 | 413 | 33.63 |
| qwen2.5-7b | answer | Wren ↔ Dana | 1337 | 4.33 | 7.98 | 405 | 419 | 32.22 |
| qwen2.5-7b | context | Wren ↔ Vex | 1282 | 10.42 | 12.64 | 357 | 391 | 35.88 |
| qwen2.5-7b | answer | Wren ↔ Vex | 1282 | 8.94 | 12.90 | 366 | 394 | 33.19 |
| qwen2.5-7b | context | Dana ↔ Vex | 1304 | 15.06 | 17.30 | 355 | 406 | 33.90 |
| qwen2.5-7b | answer | Dana ↔ Vex | 1304 | 11.23 | 15.19 | 368 | 405 | 32.15 |
| qwen2.5-7b-instruct | context | HELIOS ↔ Wren | 1432 | 23.11 | 26.33 | 373 | 463 | 32.36 |
| qwen2.5-7b-instruct | answer | HELIOS ↔ Wren | 1432 | 10.13 | 15.00 | 355 | 389 | 34.60 |
| qwen2.5-7b-instruct | context | HELIOS ↔ Dana | 1391 | 28.92 | 31.49 | 359 | 474 | 28.85 |
| qwen2.5-7b-instruct | answer | HELIOS ↔ Dana | 1391 | 14.79 | 20.43 | 343 | 394 | 34.43 |
| qwen2.5-7b-instruct | context | HELIOS ↔ Vex | 1315 | 22.39 | 25.11 | 359 | 441 | 32.08 |
| qwen2.5-7b-instruct | answer | HELIOS ↔ Vex | 1315 | 15.65 | 20.41 | 327 | 379 | 34.13 |
| qwen2.5-7b-instruct | context | Wren ↔ Dana | 1338 | 8.44 | 11.82 | 423 | 452 | 31.22 |
| qwen2.5-7b-instruct | answer | Wren ↔ Dana | 1338 | 4.85 | 8.42 | 375 | 390 | 34.69 |
| qwen2.5-7b-instruct | context | Wren ↔ Vex | 1282 | 12.14 | 14.65 | 386 | 427 | 33.90 |
| qwen2.5-7b-instruct | answer | Wren ↔ Vex | 1282 | 10.74 | 15.33 | 331 | 365 | 36.48 |
| qwen2.5-7b-instruct | context | Dana ↔ Vex | 1306 | 16.71 | 19.07 | 385 | 444 | 31.51 |
| qwen2.5-7b-instruct | answer | Dana ↔ Vex | 1306 | 12.61 | 17.08 | 343 | 384 | 34.26 |

## Scope, limitations and reproduction

- Held-out projection uses the observed target difference: subspace coverage, not prediction or zero-shot transfer.
- Pair cohorts differ; common-cohort mean-shift analysis separately uses the intersection across all four characters.
- Four character means have contrast rank at most three by construction.
- Story scaffolds differ across characters; finite-five-rollout answer noise is not corrected.
- Ranks are measured in the observed sample and bounded by sample count, not estimates of a noise-free population rank.

All 24 planned pair/representation/model panels, 120 fold evaluations and four mean-shift panels completed. The constant-vector per-query errors and aggregate fraction match the prior strict matched-query analysis. All input banks, parent cohort arrays, output arrays and source code are fingerprinted. Exact-constant synthetic controls return raw rank one and centered rank zero; direct-SVD tests validate the Gram calculation for both n<d and n>d. Tiny eigenvalues below the recorded Gram roundoff tolerance are excluded from projection bases; this is numerical resolution, not denoising.

Reproduce: `issue2054_k5_matched_run.py --rank --out eval_results/issue_2054/k5_matched_rank --inputs eval_results/issue_2054/k5_matched_offsets_strict/inputs.json`, then `issue2054_k5_matched_rank_plot.py --out eval_results/issue_2054/k5_matched_rank --figures figures/issue_2054/k5_matched_rank`. Use a fresh output directory. Eight BLAS threads per worker, two checkpoint workers; no new generation or downloads. Parent result SHA-256: 3e4d958221bee06f23fa6f77a79ce8e7101671d662a16d7c6bf34baf574437a7.
