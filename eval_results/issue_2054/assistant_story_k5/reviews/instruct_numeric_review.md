# Instruct Assistant-story transfer review

PASS for the completed Instruct transfer grid. Snapshot: 2026-09-16T02:34:29.561982+00:00; source Git SHA `b3d843420034183473f92ab64062aab263f73f41`.
Input SHA256: `a666c6263171f0431ca86172bacd0b9e969a91436394037b6c2995a0e8aae42f`.

Exact coverage: 60/60 primary transfer folds, 5/5 pooled diagnostics and 5/5 own-map folds. Each direction contains folds 0–4 exactly once. All 70 metadata records match receipt hashes and sizes. This is not a whole-job completion verdict.

**Key asymmetry:** Assistant-story→Chat transfers positively, whereas Chat→Assistant-story remains negative. A blanket bidirectional story/chat failure claim would be false for Instruct.

All triples below are **R² / Euclidean top-1 / cosine top-1**, averaged equally across folds.

## Frozen transfer and query-matched sensitivity

| Source → target | Full cohort | Query-matched cohort | Full R² fold range | Full pool | Matched pool |
|---|---:|---:|---:|---:|---:|
| Assistant-story → HELIOS | 0.528 / 68.5% / 75.8% | 0.529 / 75.2% / 81.9% | [0.526, 0.530] | 1,543–1,659 | 573–648 |
| Assistant-story → WREN | 0.500 / 66.7% / 73.1% | 0.505 / 76.6% / 80.8% | [0.494, 0.504] | 1,570–1,655 | 512–566 |
| Assistant-story → DANA | 0.485 / 64.7% / 72.8% | 0.489 / 72.7% / 80.5% | [0.482, 0.490] | 1,562–1,647 | 512–542 |
| Assistant-story → VEX | 0.402 / 62.4% / 67.4% | 0.406 / 72.2% / 76.7% | [0.396, 0.413] | 1,556–1,637 | 482–554 |
| Assistant-story → Chat | 0.270 / 46.6% / 59.4% | 0.270 / 46.6% / 59.4% | [0.258, 0.289] | 1,560–1,656 | 1,560–1,656 |
| Assistant-story → Plain | -0.815 / 14.0% / 15.0% | -0.815 / 14.0% / 15.0% | [-0.891, -0.709] | 1,560–1,656 | 1,560–1,656 |
| HELIOS → Assistant-story | 0.524 / 70.9% / 77.1% | 0.533 / 79.3% / 84.2% | [0.515, 0.530] | 1,560–1,656 | 573–648 |
| WREN → Assistant-story | 0.480 / 62.5% / 69.1% | 0.488 / 71.8% / 78.0% | [0.473, 0.485] | 1,560–1,656 | 512–566 |
| DANA → Assistant-story | 0.452 / 60.6% / 67.4% | 0.465 / 71.9% / 76.8% | [0.443, 0.458] | 1,560–1,656 | 512–542 |
| VEX → Assistant-story | 0.433 / 52.0% / 61.0% | 0.446 / 62.4% / 70.5% | [0.425, 0.437] | 1,560–1,656 | 482–554 |
| Chat → Assistant-story | -0.243 / 10.6% / 16.0% | -0.243 / 10.6% / 16.0% | [-0.280, -0.218] | 1,560–1,656 | 1,560–1,656 |
| Plain → Assistant-story | -0.773 / 0.2% / 0.4% | -0.773 / 0.2% / 0.4% | [-0.954, -0.655] | 1,560–1,656 | 1,560–1,656 |
| Pooled-six → Assistant-story | 0.546 / 73.7% / 80.2% | 0.572 / 92.6% / 95.2% | [0.537, 0.551] | 1,560–1,656 | 91–110 |

Assistant-story own map: 0.545 / 73.0% / 78.8%; identity+bias: -0.904 / 51.7% / 58.2%.
assistant story to characters: 0.478 / 65.6% / 72.3%, equal character weighting.
characters to assistant story: 0.472 / 61.5% / 68.7%, equal character weighting.

Full pools: 1,543–1,659 (chance top-1 0.060–0.065%). Matched character pools: 482–648 (0.154–0.207%). Matched Chat/Plain pools: 1,560–1,656 (0.060–0.064%). Pooled-six matched pools: 91–110 (0.909–1.099%).

## Target-training calibration

| Source → target | Bias | Bias + scalar | Target own map |
|---|---:|---:|---:|
| Assistant-story → HELIOS | 0.530 / 68.9% / 75.4% | 0.530 / 70.0% / 76.1% | 0.547 / 73.0% / 78.9% |
| Assistant-story → WREN | 0.505 / 66.9% / 73.2% | 0.505 / 65.7% / 72.1% | 0.554 / 74.7% / 80.3% |
| Assistant-story → DANA | 0.496 / 65.8% / 73.2% | 0.496 / 63.6% / 71.4% | 0.561 / 77.7% / 83.3% |
| Assistant-story → VEX | 0.421 / 63.5% / 68.7% | 0.429 / 53.1% / 60.1% | 0.512 / 72.2% / 76.9% |
| Assistant-story → Chat | 0.435 / 53.1% / 59.6% | 0.446 / 42.3% / 50.7% | 0.675 / 84.4% / 86.6% |
| Assistant-story → Plain | 0.038 / 23.0% / 25.8% | 0.212 / 6.0% / 7.5% | 0.454 / 33.3% / 36.5% |
| HELIOS → Assistant-story | 0.527 / 70.4% / 76.7% | 0.527 / 71.0% / 77.1% | 0.545 / 73.0% / 78.8% |
| WREN → Assistant-story | 0.492 / 63.5% / 70.2% | 0.492 / 64.6% / 71.4% | 0.545 / 73.0% / 78.8% |
| DANA → Assistant-story | 0.471 / 60.8% / 68.8% | 0.471 / 60.0% / 68.2% | 0.545 / 73.0% / 78.8% |
| VEX → Assistant-story | 0.454 / 51.4% / 60.6% | 0.454 / 53.8% / 62.4% | 0.545 / 73.0% / 78.8% |
| Chat → Assistant-story | 0.212 / 21.1% / 28.4% | 0.239 / 8.3% / 13.7% | 0.545 / 73.0% / 78.8% |
| Plain → Assistant-story | 0.193 / 2.2% / 3.5% | 0.201 / 4.8% / 7.2% | 0.545 / 73.0% / 78.8% |
| Pooled-six → Assistant-story | 0.549 / 73.5% / 80.0% | 0.549 / 76.2% / 81.7% | 0.545 / 73.0% / 78.8% |

Bias→bias+scalar Euclidean top-1 falls 21.1%→8.3% for Chat→Assistant-story, 53.1%→42.3% for Assistant-story→Chat, 23.0%→6.0% for →Plain, and 63.5%→53.1% for →VEX, while R² rises in each case. These adaptations use target-training labels.

## Identity-plus-bias controls

| Source → target | Source-trained bias | Target-trained bias |
|---|---:|---:|
| Assistant-story → HELIOS | -1.004 / 49.2% / 55.2% | -0.893 / 51.0% / 58.3% |
| Assistant-story → WREN | -1.377 / 59.6% / 66.6% | -0.975 / 59.6% / 67.4% |
| Assistant-story → DANA | -1.200 / 68.4% / 71.7% | -0.741 / 68.7% / 74.3% |
| Assistant-story → VEX | -1.908 / 67.5% / 73.7% | -1.271 / 67.5% / 74.0% |
| Assistant-story → Chat | -3.748 / 30.0% / 30.2% | -0.955 / 46.5% / 53.3% |
| Assistant-story → Plain | -9.684 / 12.2% / 13.0% | -2.888 / 30.6% / 34.2% |
| HELIOS → Assistant-story | -1.015 / 51.6% / 59.3% | -0.904 / 51.7% / 58.2% |
| WREN → Assistant-story | -1.264 / 45.1% / 48.9% | -0.904 / 51.7% / 58.2% |
| DANA → Assistant-story | -1.361 / 42.6% / 47.6% | -0.904 / 51.7% / 58.2% |
| VEX → Assistant-story | -1.382 / 43.6% / 47.7% | -0.904 / 51.7% / 58.2% |
| Chat → Assistant-story | -3.828 / 23.4% / 27.7% | -0.904 / 51.7% / 58.2% |
| Plain → Assistant-story | -4.402 / 22.8% / 24.9% | -0.904 / 51.7% / 58.2% |
| Pooled-six → Assistant-story | -1.349 / 41.0% / 46.5% | -0.904 / 51.7% / 58.2% |

## Cross-checkpoint framing directions

| Source → target | Base frozen | Instruct frozen |
|---|---:|---:|
| Assistant-story → Chat | -0.468 / 3.9% / 4.9% | 0.270 / 46.6% / 59.4% |
| Chat → Assistant-story | -0.344 / 0.8% / 1.4% | -0.243 / 10.6% / 16.0% |
| Assistant-story → Plain | -0.268 / 6.1% / 8.6% | -0.815 / 14.0% / 15.0% |
| Plain → Assistant-story | -0.380 / 1.0% / 2.3% | -0.773 / 0.2% / 0.4% |

## Supported conclusions and limits

- Assistant-story→named-character frozen transfer has positive R² in all 20 character/fold cells, with strong Euclidean and cosine retrieval. Every reverse named-character→Assistant-story direction also has positive R² in every fold.
- Assistant-story→Chat is a directional exception to a blanket framing-barrier narrative: frozen R² is positive in all folds (0.258–0.289), averaging 0.270; Euclidean/cosine top-1 are 46.6%/59.4%. Chat→Assistant-story remains negative in all folds, averaging −0.243, with 10.6%/16.0% retrieval.
- Plain→Assistant-story and Assistant-story→Plain remain negative in every frozen R² fold (means −0.773 and −0.815). Retrieval remains above chance and asymmetric: 0.24%/0.36% versus 14.0%/15.0% Euclidean/cosine top-1.
- Query-matched sensitivity preserves every direction’s frozen R² sign. Character retrieval pools are smaller; increased matched retrieval must not be interpreted as a controlled mapping improvement.
- Bias and bias+scalar require target-training examples. Scalar calibration increases R² but sharply lowers retrieval relative to bias-only for Chat→Assistant-story and Assistant-story→Chat/Plain/VEX.
- Identity+bias has negative R² but sometimes higher retrieval. Source-trained identity+bias exceeds frozen-map Euclidean top-1 for Assistant-story→DANA/VEX and Chat/Plain→Assistant-story. Learned-map dominance is not uniform across metrics.
- The frozen six-setting pooled map excludes Assistant-story during fitting and reaches R² 0.546, versus own-map 0.545, with Euclidean/cosine top-1 73.7%/80.2%. This is comparable performance, not statistically established superiority.
- Within-story transfer is strong in both checkpoints. Base Assistant-story→Chat has negative frozen R² (−0.468), whereas Instruct has positive R² (0.270). Preserve checkpoint and directional distinctions.

- This verdict covers the completed Instruct transfer grid. The answer comparison and overall job are not declared complete from this snapshot. Transfer metrics do not establish semantic answer similarity.
- Framing bundles narrative content, reply boundaries and speaker presentation. Question matching does not remove narrative-added information or isolate a persona-name effect.
- Entries are equal-fold means; fold ranges are descriptive, not confidence intervals. Four fixed named characters do not represent a random sample of all personas.
- Query-matched character cohorts contain 2,547–3,047 conversations per pair. The pooled-six common-question diagnostic contains only 496, with 91–110 candidates per fold.
- All 70 metadata bodies were independently reconstructed and checked against receipt hashes and sizes. This audit does not reopen all prediction arrays or re-fit maps.
