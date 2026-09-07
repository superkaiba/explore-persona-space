# Direct answer-side residual SAE analysis

This follow-up compares the observed answer change, the map-predicted answer change, and their residual using frozen layer-19 vectors. No model inference was rerun.

## Numerical summary

| Pair family | n | Median cos(observed, predicted) | Median predicted/observed norm | Median residual/observed norm |
|---|---:|---:|---:|---:|
| question_topic_66 | 66 | 0.725 | 0.809 | 0.705 |
| oneword_topic_24 | 24 | 0.547 | 0.668 | 0.850 |
| minimal_refusal_flips_60 | 60 | 0.799 | 0.756 | 0.606 |

The SAE rankings are descriptive. Decoder directions are correlated and non-orthogonal, so their scores overlap and cannot be treated as additive variance contributions. The available Codex descriptions cover only a previously selected subset of answer features.

## Top-100 description coverage and substantive-topic counts

Primary score: RMS cosine between each unit-normalized pair direction and each unit SAE decoder direction.

| Pair family | Component | Labeled in top 100 | Interpretable | Substantive topic/domain |
|---|---|---:|---:|---:|
| question_topic_66 | observed | 26 | 26 | 11 |
| question_topic_66 | predicted | 52 | 52 | 24 |
| question_topic_66 | residual | 38 | 37 | 10 |
| oneword_topic_24 | observed | 35 | 35 | 13 |
| oneword_topic_24 | predicted | 62 | 61 | 25 |
| oneword_topic_24 | residual | 47 | 46 | 11 |
| minimal_refusal_flips_60 | observed | 88 | 84 | 18 |
| minimal_refusal_flips_60 | predicted | 98 | 94 | 24 |
| minimal_refusal_flips_60 | residual | 76 | 74 | 17 |

For the 66 question-topic pairs, 11/11 labeled substantive-topic features in the observed top 100 also occur in the predicted top 100; the residual-to-predicted overlap is 9/10. For the 24 one-word pairs, the corresponding overlaps are 13/13 and 10/11. Thus the labeled feature view does not place topic information uniquely in the unmapped residual.

These are counts among labeled top-100 features, not prevalence estimates. The comparison is qualitative because label coverage is incomplete and selected. Use the full rankings JSON for feature-level descriptions and the NPZ for all scores.
