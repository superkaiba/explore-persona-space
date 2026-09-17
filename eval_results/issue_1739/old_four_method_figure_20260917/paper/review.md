# Behavior section rewrite: evidence and review

## Outline and paragraph roles

- Opening: define the three behaviors, evaluation regimes, and original augmented-map protocol.
- Method: transfer a fixed answer-derived readout without fitting a behavior-specific regression.
- Evidence: compare predicted-answer and observed-answer readouts using the nine displayed cells.
- Comparison: quantify gains against both fixed context projections and state hallucination exceptions.
- Application and limitation: use the separate million-pair map's approximate inverse to retrieve a verified SimpleQA failure, with NQ counterevidence and baseline ranks.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| The map makes answer-derived readouts available before generation | Fixed contrastive direction applied to predicted answer activations; no additional behavior regression | Supported; original layer choice used labeled training analyses |
| Performance approaches observed-answer readouts on average | Equal nine-cell means: 0.224329 versus 0.265372 | Supported descriptively; no equivalence test or uniform-performance claim |
| Mapping improves on both context projections in most settings | Context-native mean 0.064932; answer-on-context mean 0.030040; mapped wins 7/9 and 8/9 | Supported for this augmented-map/whitened/method-specific-layer protocol |
| Pre-images can retrieve ordinary contexts with observed failures | SimpleQA rank-1 Dalí question; five original completions, five decided, five fabricated; saved gold and raw source verified | Supported as a separate million-pair-map example; context-native also ranks it highly |
| High pre-image cosine identifies higher-risk SimpleQA contexts | Top/bottom decile context-mean fabrication fractions 0.880597 / 0.321891 | Supported on SimpleQA; NQ reverses, 0.387975 / 0.475316 |

`claim_metrics.json` records the exact arithmetic and original cells.
`preimage_case_audit.json` records raw examples, all method ranks, tail denominators,
source paths and verified hashes. The regularized pre-image is the rank-378
truncated inverse; it is not the transpose pullback of the answer score.

## Self-review

- Contribution: the section now concerns reuse of answer-derived instruments and context retrieval, matching the user's requested claims.
- Clarity and flow: one claim per paragraph, readout source/target explained, technical details in appendix.
- Experimental strength: all 36 source values preserved; averages/wins reproduced independently; hallucination exceptions retained.
- Evaluation completeness: original protocol and generic-map controls kept distinct; ID map overlap, conditional judgments and original OOD SEMs disclosed.
- Method soundness: no label-free layer-selection, universal equivalence, unique pre-image advantage, or causality claim; observed-answer performance is not called an upper bound.

Independent evidence review by `four_method_review` found no material blockers.
The full manuscript compiled with resolved references. The parent inspected the
behavior pages and revised plot at manuscript scale; plot labels contain no sample counts.
