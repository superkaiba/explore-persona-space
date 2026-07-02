---
title: 'Audit #667-line last-layer (L27) activation reads for mixed pre-/post-final-norm
  convention contamination (bare base vs PeftModel trained)'
kind: analysis
tags: []
created_at: '2026-07-02T20:44:28Z'
has_clean_result: false
---
## Overview / Motivation

Filed from task #868's Phase-2 critique (alternatives lens, reconciler-upheld). The shared helper `extract_layer_activations` routes a bare model through the HOOK path (pre-final-norm last layer) but a `peft.PeftModel` through the FALLBACK path (post-final-norm last layer) — see #886 for the code fix. The #667-line extraction drivers compare bare-base vs PeftModel-trained activations, so their LAST-layer (L27) reads mix conventions.

## Goal

Quantify whether the #667-line base-vs-trained last-layer (block 27) activation reads are contaminated by the mixed pre-/post-final-RMSNorm convention, and whether any #667-line conclusion that used L27 cells changes after correction.

## Detail

- Affected call sites: `scripts/issue667_extract.py:591-592` (paired base/trained reads), `:1051-1052` (`_context_vector_all_layers(base)` vs `(trained)`, all 28 blocks incl. L27), `scripts/issue667_pertoken_extract.py:194-195`, `scripts/issue667_pertoken_context_extract.py:195-196` — all load via `load_base_and_trained` (`issue667_extract.py:466-482`: bare base + `PeftModel` trained).
- Expected contamination scale: the pre-vs-post-norm divergence is large — #493 measured cosine diff ~1.62e-1 at L27 vs ~1e-4-2e-3 layer-graded noise at other layers (`scripts/issue493_extraction_metric_bakeoff.py:192-199`).
- The trained−base shift at L27 conflates the LoRA-induced change with the final-RMSNorm application. Non-last layers are unaffected.
- Audit shape (0 GPU-h candidate first): grep which #667-line RESULTS actually consumed L27 cells (per-layer predictor sweeps, layer-selection argmax, heatmaps); if L27 never carried a headline (e.g. best layers were mid-stack), the audit closes with a scope note on the #667-line clean-results; if L27 DID carry weight, re-extract the affected cells with a convention-consistent read (needs GPU) and re-run the affected fits.
- Deliverable: a record-correction note (or all-clear) on the affected #667-line task bodies.

## Constraints

- Depends on understanding, not blocking, #886 (the code fix). Audit reads existing artifacts first.

## Provenance

- Origin: #868 Phase-2 alternatives-lens finding (reconciled REVISE), 2026-07-02.
