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
