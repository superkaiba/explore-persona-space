# Base-only Assistant-story transfer review

PASS for the completed base transfer grid. Snapshot: 2026-09-16T01:36:03.811902+00:00; reviewed 2026-09-16T01:42:35.459276+00:00.

Source Git SHA: `b3d843420034183473f92ab64062aab263f73f41`. Input SHA256: `e5f5d1635f7cc75db1f690c3417d9b67c11b594fd4d5b3e9c91c8c57685927a4`.

Coverage is exact: 60/60 primary directed transfer folds, 5/5 pooled-six diagnostic folds, and 5/5 Assistant-story own-map folds. Every direction has folds 0–4 exactly once. All 70 metadata records reproduce their receipt SHA256 and byte size. This is not an overall-job completion verdict.

All triples below are **R² / Euclidean top-1 / cosine top-1**, with arithmetic means across five folds. Character averages weight the four targets equally.

## Frozen transfer and question-match sensitivity

| Source → target | Full test cohort | Query-matched test cohort | Full pool | Matched pool |
|---|---:|---:|---:|---:|
| Assistant-story → HELIOS | 0.509 / 70.9% / 77.1% | 0.512 / 76.4% / 83.5% | 1,543–1,659 | 571–648 |
| Assistant-story → WREN | 0.481 / 67.8% / 73.4% | 0.489 / 75.6% / 80.4% | 1,569–1,655 | 511–564 |
| Assistant-story → DANA | 0.481 / 67.5% / 74.1% | 0.488 / 75.1% / 80.6% | 1,561–1,645 | 510–541 |
| Assistant-story → VEX | 0.392 / 65.6% / 70.5% | 0.401 / 75.2% / 79.3% | 1,556–1,637 | 482–554 |
| Assistant-story → Chat | -0.468 / 3.9% / 4.9% | -0.468 / 3.9% / 4.9% | 1,560–1,656 | 1,559–1,656 |
| Assistant-story → Plain | -0.268 / 6.1% / 8.6% | -0.268 / 6.1% / 8.7% | 1,560–1,656 | 1,559–1,656 |
| HELIOS → Assistant-story | 0.500 / 65.2% / 73.6% | 0.508 / 73.0% / 80.8% | 1,559–1,656 | 571–648 |
| WREN → Assistant-story | 0.461 / 58.8% / 69.6% | 0.468 / 67.0% / 76.7% | 1,559–1,656 | 511–564 |
| DANA → Assistant-story | 0.441 / 58.3% / 68.5% | 0.457 / 69.1% / 78.4% | 1,559–1,656 | 510–541 |
| VEX → Assistant-story | 0.428 / 48.5% / 59.4% | 0.439 / 61.0% / 71.0% | 1,559–1,656 | 482–554 |
| Chat → Assistant-story | -0.344 / 0.8% / 1.4% | -0.344 / 0.8% / 1.4% | 1,559–1,656 | 1,559–1,656 |
| Plain → Assistant-story | -0.380 / 1.0% / 2.3% | -0.380 / 1.0% / 2.3% | 1,559–1,656 | 1,559–1,656 |
| Pooled-six → Assistant-story | 0.527 / 70.7% / 78.8% | 0.552 / 89.3% / 91.4% | 1,559–1,656 | 91–110 |

Assistant-story own map: 0.524 / 71.9% / 78.4%; own identity+bias: -0.901 / 63.9% / 68.4%.
assistant story to characters: 0.466 / 67.9% / 73.8% frozen, equal weight across characters.
characters to assistant story: 0.458 / 57.7% / 67.8% frozen, equal weight across characters.

Full-cohort pools span 1,543–1,659 targets (chance top-1 0.060–0.065%). Query-matched character pools span 482–648 (chance 0.154–0.207%). Matched Chat/Plain pools span 1,559–1,656 (chance 0.060–0.064%). The pooled-six matched diagnostic uses 91–110 targets (chance 0.909–1.099%); its retrieval is not comparable to the full-pool retrieval without this qualification.

## Target-trained calibration

| Source → target | Bias | Bias + scalar | Target own map |
|---|---:|---:|---:|
| Assistant-story → HELIOS | 0.512 / 71.2% / 77.0% | 0.512 / 70.0% / 76.1% | 0.530 / 72.9% / 78.4% |
| Assistant-story → WREN | 0.487 / 68.2% / 74.4% | 0.488 / 65.6% / 72.3% | 0.529 / 74.1% / 79.9% |
| Assistant-story → DANA | 0.491 / 69.0% / 75.9% | 0.491 / 68.1% / 75.3% | 0.544 / 78.4% / 83.4% |
| Assistant-story → VEX | 0.410 / 66.2% / 71.2% | 0.420 / 54.9% / 61.8% | 0.482 / 72.3% / 76.5% |
| Assistant-story → Chat | 0.073 / 6.5% / 8.0% | 0.147 / 1.3% / 1.6% | 0.410 / 16.3% / 19.2% |
| Assistant-story → Plain | 0.156 / 9.0% / 11.5% | 0.199 / 3.1% / 4.0% | 0.469 / 30.4% / 35.8% |
| HELIOS → Assistant-story | 0.505 / 65.8% / 73.8% | 0.506 / 68.5% / 75.9% | 0.524 / 71.9% / 78.4% |
| WREN → Assistant-story | 0.478 / 60.7% / 69.1% | 0.479 / 63.3% / 71.4% | 0.524 / 71.9% / 78.4% |
| DANA → Assistant-story | 0.465 / 60.6% / 68.9% | 0.465 / 62.2% / 70.2% | 0.524 / 71.9% / 78.4% |
| VEX → Assistant-story | 0.448 / 49.0% / 57.9% | 0.451 / 56.7% / 65.1% | 0.524 / 71.9% / 78.4% |
| Chat → Assistant-story | 0.176 / 2.2% / 3.1% | 0.177 / 2.6% / 3.8% | 0.524 / 71.9% / 78.4% |
| Plain → Assistant-story | 0.185 / 2.9% / 4.6% | 0.185 / 3.2% / 4.9% | 0.524 / 71.9% / 78.4% |
| Pooled-six → Assistant-story | 0.530 / 70.6% / 78.1% | 0.531 / 75.0% / 81.4% | 0.524 / 71.9% / 78.4% |

These are target-training adaptations, not frozen transfer. The most pronounced bias→bias+scalar retrieval drops include Assistant-story→Chat (6.5%→1.3%), →Plain (9.0%→3.1%), and →VEX (66.2%→54.9%), despite improved R².

## Identity-plus-bias controls

| Source → target | Bias fitted on source | Bias fitted on target |
|---|---:|---:|
| Assistant-story → HELIOS | -1.133 / 64.0% / 66.7% | -0.998 / 66.1% / 70.4% |
| Assistant-story → WREN | -1.494 / 69.8% / 74.5% | -1.044 / 71.6% / 77.4% |
| Assistant-story → DANA | -1.266 / 75.5% / 77.2% | -0.780 / 77.4% / 80.9% |
| Assistant-story → VEX | -2.049 / 78.4% / 82.2% | -1.375 / 78.6% / 83.4% |
| Assistant-story → Chat | -10.224 / 2.0% / 2.1% | -2.515 / 9.7% / 11.9% |
| Assistant-story → Plain | -8.032 / 13.4% / 14.1% | -1.813 / 27.4% / 31.8% |
| HELIOS → Assistant-story | -1.032 / 63.3% / 69.5% | -0.901 / 63.9% / 68.4% |
| WREN → Assistant-story | -1.312 / 56.9% / 60.7% | -0.901 / 63.9% / 68.4% |
| DANA → Assistant-story | -1.413 / 54.2% / 58.2% | -0.901 / 63.9% / 68.4% |
| VEX → Assistant-story | -1.407 / 54.8% / 58.5% | -0.901 / 63.9% / 68.4% |
| Chat → Assistant-story | -5.653 / 44.2% / 45.0% | -0.901 / 63.9% / 68.4% |
| Plain → Assistant-story | -5.503 / 35.0% / 35.6% | -0.901 / 63.9% / 68.4% |
| Pooled-six → Assistant-story | -1.454 / 55.9% / 58.3% | -0.901 / 63.9% / 68.4% |

## Supported conclusions and limits

- BASE ONLY: A source-only Assistant-story map predicts unseen named-character answer representations with positive frozen R² in all 20 target/fold cells and strong held-out retrieval. Reverse character-to-Assistant-story directions are also positive in every fold.
- Frozen Chat→Assistant-story, Plain→Assistant-story, Assistant-story→Chat and Assistant-story→Plain all have negative R² in every fold. Their retrieval is weak but above chance; describe poor transfer, not zero information or total failure.
- Query-matched sensitivity preserves the same positive/negative R² pattern. Higher matched retrieval is not an isolated improvement in the mapping: the retrieval pool is smaller and the retained question distribution differs.
- Bias and bias+scalar use target-training labels and are adaptation results. They make cross-framing R² positive but do not approach the corresponding own-map scores.
- R² and retrieval dissociate: in Assistant-story→Chat/Plain and →VEX, adding scalar calibration to bias improves R² while sharply lowering Euclidean and cosine top-1.
- Identity+bias controls have negative R² but sometimes higher retrieval than the learned map. For Assistant-story→DANA/WREN/VEX, the source-trained identity+bias baseline exceeds frozen-map Euclidean top-1; do not claim learned-map dominance on both metrics.
- The frozen six-setting pooled map, whose fit excludes Assistant-story, reaches approximately the new own-map R². This is additional unseen-setting transfer evidence, not proof of a unique internal mechanism.

- Only the base checkpoint transfer grid is complete in this reviewed snapshot. The overall job, answer-geometry comparison, qualitative audit and instruction-tuned results are outside this verdict.
- Framing changes narrative content, answer boundaries and speaker presentation together; matching the question does not remove narrative-added information. These results do not isolate a persona-name effect or semantic answer similarity.
- Metrics are equal means over five held-out conversation folds, with fixed named settings. Fold ranges are descriptive, not confidence intervals or evidence about a population of all possible characters.
- Full target pools differ slightly across settings due to complete-five eligibility. Matched-character comparisons retain 2,546–3,042 conversations per pair; the all-six-source matched pooled diagnostic retains only 496.
- Receipt verification in this review reconstructs and hashes all 70 metadata records against the provided receipts. Prediction NPZ contents and live backend state were not independently reread here; the coordinating agent provided the receipt-verified remote snapshot.
