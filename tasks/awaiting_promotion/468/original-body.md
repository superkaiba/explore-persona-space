---
title: 'Why does the #463 cosine→EM signal appear at last-prompt-token but not the
  canonical response-mean extraction?'
kind: experiment
tags: []
created_at: '2026-06-02T18:26:27Z'
has_clean_result: false
parent_id: 463
goal: 'Explain why the #463 cosine→EM predictor appears at the last-prompt-token extraction
  but not the canonical response-mean extraction, and decide which extraction is principled
  for predicting emergent misalignment.'
relates_to:
- beh-b-to-bprime
- app5
---
# Why does the #463 cosine→EM signal appear at last-prompt-token but not the canonical response-mean extraction?

## Goal

Explain why the #463 cosine→EM predictor appears at the last-prompt-token extraction but not the canonical response-mean extraction, and decide which extraction is principled for predicting emergent misalignment.

## Why
The persona-vectors paper (arXiv 2507.21509) treats the **response-mean** (mean residual over the model's own generated tokens) as the canonical persona-direction recipe; #463's headline rides on the **last-prompt-token** recipe instead. That the canonical recipe fails to replicate is one of the main reasons #463 is LOW confidence: if the signal only lives at the final prompt token, it may be an artifact of the chat-template's last token rather than a property of the persona representation. We need to know which it is before trusting (or discarding) the predictor.

## Design (mostly re-analysis of existing #463 data + a few new extraction variants; predictor-only, NO training)
Existing #463 per-cell JSONs already store BOTH extractions for all 28 layers (`predictor_cossim_training/<cell>_lit.json`, `cos_by_extraction.{last_prompt_token,response_mean}`), so start there:

1. **Characterize the divergence.** Per cell, deep layers (L18–27), lit, training probes: compare last-prompt-token vs response-mean cosines. Is response-mean noisier, saturated/compressed toward 1.0, or rank-reordering the cells? Correlate each extraction's per-cell cosine with EM and with response length.
2. **Length / content confound on response-mean.** response-mean pools over the model's OWN generations under S_narrow vs S_broad, which differ in length and topic. Check whether response-mean cosine is dominated by generation-length or generation-content variance rather than persona direction.
3. **New extraction variants** (the only part needing fresh forward passes): mean-over-response EXCLUDING the first k tokens; last-response-token; max-pool over response; and last-prompt-token at the true final content token vs the chat-template's trailing tokens (to test the "it's just the template's last token" artifact hypothesis). Run at L18–27, lit, training probes, 18 cells.

## Success / kill
- A clear mechanistic account of the divergence + a principled recommendation (e.g. "last-prompt-token at the final CONTENT token is the right recipe because response-mean is length-confounded") → resolves the #463 caveat in the persona-geometry direction.
- If the last-prompt-token signal is shown to be an artifact of the chat-template's trailing token (disappears at the final content token) → further deflates #463; the predictor is not measuring persona representation.

## Compute
Predictor-only. Re-analysis of existing JSONs is ~free; the new extraction variants are ~1–3 GPU-hr on 1×H100 (base-model forward passes only). New code: extraction-variant flags for `issue463_predictor_cossim.py` + an analysis script over the existing cossim_training JSONs.

## References
Parent #463 (the divergence: last-prompt-token ρ=0.71 vs response-mean ρ=0.41 n.s.). Persona Vectors (arXiv 2507.21509, canonical response-mean recipe). CLAUDE.md "Persona-distance metrics" canonical defs (cosine extraction points (a) last prompt token, (b) response-mean).
