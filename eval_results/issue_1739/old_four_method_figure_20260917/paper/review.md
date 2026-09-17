# Behavior section rewrite: evidence and review

## Outline and paragraph roles

- Opening: define behaviors, evaluation regimes, and the original augmented-map protocol.
- Combined claim: transfer fixed answer-derived readouts before generation, approach observed-answer performance on average, and exceed both context baselines in most settings; retain the hallucination exceptions.
- Quantitative retrieval claim: compare pre-images against both context controls on identical held-out pools, report SimpleQA and NQ top/bottom gaps, and keep the verified Dalí example with all three ranks.

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

## Quantitative retrieval revision

All 15 cosine-decile rows (three methods, five datasets) were independently
reconstructed from cached predictions and existing labels by `four_method_review`.
SimpleQA top/bottom percentages: pre-image 88.06/32.19; context-derived 90.03/21.39;
answer direction on context 87.78/21.94. Pre-images are never the best of these
three methods on top-decile mean or top-minus-bottom separation across the five
cached datasets. The claim explicitly avoids a general advantage or significance
assertion. NQ reverses the pre-image ranking and remains in the main text.

The main paragraph retains the distinct million-pair map. Appendix H gives the
15-row table, common standardized coordinates, baseline directions D v_C and
D v_A, floor(n/10) tail sizes, direction-specific selections, rubric-score versus
percentage units, and judgment-coverage limitations. No new fitting, judging,
bootstrap, or generation was performed. Independent final prose review passed;
the full manuscript compiled and the behavior page and appendix table were
visually inspected. The unchanged figure still has no sample-count labels.
