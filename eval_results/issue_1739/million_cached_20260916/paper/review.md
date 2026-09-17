# Manuscript presentation review, 2026-09-17

## Outline and paragraph roles

1. Opening: fixed-direction transfer without behavior-specific regression, frozen map and evaluation scope.
2. Evidence A: answer directions predict behavior on some datasets, including NQ failure.
3. Evidence B: preimage scores against context-native directions, without a consistent advantage.
4. Interpretation: highest-ranked prompts reveal recurring templates, without establishing induced behavior.
5. Evidence C: matched regression and context-only covariance, without a consistent prediction advantage.

## Claim-evidence map

- Fixed-direction prediction is dataset-dependent | fixed.json, mapped_answer and real_answer arms, all five natural datasets | supported.
- Preimages lack consistent context-direction superiority | fixed.json, preimage/context_native and paired differences | supported.
- Retrieval groups recurring prompts | hashed generic examples and archived top-30 rankings, counts17/22/30 | descriptive only.
- Regression lacks consistent gain | regression.json, mapped_answer minus raw/context_covariance/shuffled means | supported with pointwise conditional uncertainty, no equivalence claim.

## Independent review and fixes

The independent Codex review checked all six input hashes, plotted estimates and intervals,
map diagnostics, retained counts, bootstrap group counts, top-30 pattern/overlap counts,
and exact prompt excerpts. It checked the rendered figures and caption/header agreement.
Applied corrections: distinguish inverse answer displacement from affine prediction,
qualify the harmful-compliance rubric as including malicious persona/style, specify
normalized reference-answer bootstrap keys for NQ/SimpleQA, and restore the historical
judge and alias/classification protocol. Hallucination's denominator includes decided
correct, abstained, and fabricated responses. No automated Claude calls were made.

## Five-dimension self-review

- Contribution: fixed-direction transfer and interpretation are separated from supervised probe performance. No general predictive gain is claimed.
- Clarity: 303-word main section before figure, matching neighboring bold claim headers and panel pointers. Caption headers match verbatim.
- Experimental strength: weak HH and reversed NQ results remain visible. No cherry-picked inverse rank or shuffled seed.
- Evaluation completeness: exact-pool covariance and shuffled-pair controls are shown. Four-example WildChat is excluded explicitly. Refusal-specific evaluation remains absent, without an absence-as-zero plot.
- Method soundness: SVD inverse and transpose scoring are distinct. Rank and covariance are selected without behavior labels. Pooling mismatch, judge omissions, covariance grid boundary, and conditional bootstrap scope are explicit.

## Validation

- Renderer reads six completion-hash-verified JSONs and reproduces their values directly.
- Ruff lint and format checks passed.
- Both vector PDFs, PNGs and grayscale PNGs use the shared c2a-v2 style and fixed publication scale.
- Paper compiles with latexmk through bibtex and stable references, 52 pages, zero unresolved references.
- Main text and appendix page images were inspected, with no new clipping or overlaps.
- Writing-tells hard-ban scan passed. New flags concern meaningful distinctions: conditional versus unconditional scores, conditioning versus chance controls, and full-prompt versus template deduplication.
- Served PNG bytes match local figure hashes. URLs appear in publication.json.
- Overleaf main verified at ffe7d6923292916a099bcfb2c8c70429afb20f4f after a fresh fetch and fast-forward push.

## User correction: contrastive projections, 2026-09-17

The main result now compares exactly three fixed projections: the contrastive answer direction on predicted answers and on observed answers, and the contrastive context direction on contexts. Regression/covariance controls remain supplementary in Appendix H. Panel A shows these three arms together; panel B retains the preimage comparison. Abstract and discussion now summarize contrastive directions rather than regression performance. A bounded independent review found no required fixes and rechecked all newly quoted transferred-minus-context direction differences and CIs. Its optional clarification that correlations use mean retained behavior scores per context was applied. All original data estimates remain unchanged, including the improved but still negative NQ score. The corrected plot has distinct markers, readable legends, no clipping, and a verified browser image hash.

## Real-data regime extension (2026-09-17)

Thomas requested the previous behavior-by-regime layout for the three fixed projection methods, without synthetic evaluation. The CPU-only extension scores the historical in-distribution rung with the same frozen directions and 963,444-pair map; prior OOD/WildChat predictions are preserved exactly. 38,455/40,000 in-distribution contexts survive: 1,532 harmful-compliance contexts have no valid score, and 13 TriviaQA contexts overlap map fitting/validation. The eight natural datasets total 49,161 contexts. Generic chat remains 4 per behavior and is explicitly insufficient. No missing value is drawn as zero.

Independent review verified scoring, bootstrap batching, paired methods, independent dataset streams, and all equal-weight OOD means/intervals. SciPy independently reproduced every finite dataset rho; previous scores/IDs/DVs/groups are bit-for-bit unchanged. Review requested stronger continuity pins; all map, membership, prompt-index, supplementary-rollout, and slice-member pins passed independently, and future reproduction runs enforce these checks before scoring. Original scoring completion manifests remain immutable at source ec2c2e903285932d67cec00ed5361a464ea7d4bb.

The manuscript now names the added in-distribution datasets and their grouping, includes TriviaQA in the judging protocol, updates the settings table to 49,161, and states the negative TriviaQA result. The main figure averages dataset-specific OOD correlations, not pooled contexts; the appendix preserves all five OOD datasets individually and the preimage comparison. No headline claims a consistent transfer advantage.

Five-dimension review: contribution remains conditional fixed-direction transfer; writing separates ID/OOD and observed-answer validation; experiment uses identical frozen scores and exact retained rollouts; coverage shows all requested non-synthetic regimes with generic-chat insufficiency visible; design retains the pooling and conditional-judging limitations.

Final checks: all requested review fixes applied, 53-page LaTeX compilation succeeded with zero unresolved references, main and appendix figure pages inspected at manuscript scale, grayscale series distinguishable, browser images hash-verified, and Overleaf main verified at 9d5cb6dc66b2f22e07cdaf97ac268c94a9d9dcce.
