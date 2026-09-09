# Qwen3-8B chat-data mapping/rank reproduction

The end-of-thought map has lower operational rank (122 → 65; thinking minus no-thinking -57). Full-map held-out R² changes from 0.6886 to 0.6691 (-0.0194).

[Rank-curve figure](https://github.com/superkaiba/explore-persona-space/blob/b35d23960e6ea86d9c5dab59c799c7cde117e3d6/figures/issue_2588/qwen3_chat_rank.png)

| Metric | No thinking: prompt last | Thinking: end of thought |
|---|---:|---:|
| Selected layer (zero-based) | 24 | 22 |
| Full-map test R² | 0.6886 | 0.6691 |
| Identity + learned bias test R² | -1.0966 | -0.2624 |
| Operational rank | 122 | 65 |
| Rank / 4096 | 0.0298 | 0.0159 |
| Test R² at selected rank | 0.6580 | 0.6369 |
| TRAIN-input participation ratio | 26.32 | 20.14 |
| Map cosine top-1 retrieval | 73.25% | 72.56% |
| Map Euclidean top-1 retrieval | 72.85% | 69.95% |
| Identity + bias cosine top-1 | 52.30% | 0.80% |
| Identity + bias Euclidean top-1 | 49.10% | 1.21% |
| Test retrieval pool | 998 | 995 |
| Chance top-1 | 0.100% | 0.101% |
| Repeat-answer aligned pairs | 993 | 992 |
| Repeat-answer weighted Pearson | 0.9253 | 0.8800 |
| Repeat-answer cosine top-1 | 86.71% | 81.75% |
| Repeat-answer chance top-1 | 0.101% | 0.101% |

## Coverage

| Split | Planned per arm | No thinking retained | Thinking retained |
|---|---:|---:|---:|
| Train | 10000 | 9937 | 9968 |
| Validation | 400 | 399 | 399 |
| Test | 1000 | 998 | 995 |

Retained counts are condition-specific; missing responses are not imputed. Repeat-answer diagnostics use the intersection of valid seed-43 and seed-44 answers, rather than treating the two draws as independent models. The CSV also records each selected ridge penalty and the original producer's R².

## Method and limitations

This reproduces the earlier long-cap chat-data panel for one model. Each condition uses its own generated answers. Layers are selected by validation cosine top-1 retrieval; ridge penalties by validation R². Training-output PCA defines the nested maps. Operational rank is the smallest rank with validation SSE no more than 10% above that condition's own full map; test performance does not select the rank. The full rank-0–4096 validation/test curves remain in the source JSONs. Reconstruction is checked against the producer within 3e-4 R² and full-rank recovery within 1e-4.

Different targets and separately selected layers make this a descriptive comparison, not an isolated causal effect of reasoning. Rank thresholds are relative to each map's own performance, not a common absolute R² target. Neither map rank nor input participation ratio measures the total information in reasoning. Repeat-answer correlations are reliability diagnostics, not a validated bound on CoT-conditioned prediction. No population p-value or confidence interval is reported for this single model.

The frozen LMSYS-Chat-1M split contains 13 exact prompt strings in both validation and test (24 validation and 60 test rows before exclusions); training has no exact-text overlap with either. These are not wholly independent validation/test prompts. Capture uses the inherited teacher-forced replay of persisted on-policy text, not token-identical replay of sampled completion token IDs. Thinking has a 32768-token cap; the model context window prevents the inherited meaningful cap increase. No split changes, extra models, GPQA, judges, or inferential sweep were added.

## Provenance

Immutable capture/fit revision: `5f60a146e91248e5e1c84cf879ff5b4f1357a087`. Exact model, manifest, scientific-source, rank-consumer and input-file hashes are in `rank_a.json` and `rank_b.json`; report input hashes are in `report.meta.json`.
