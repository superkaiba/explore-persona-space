---
title: Assemble qualitative-examples panels for the paper (discrimination + patching)
kind: analysis
tags: []
created_at: '2026-08-22T20:09:08Z'
has_clean_result: false
parent_id: 2094
origin_prompt: 'Paper outline 2026-08-22: ''QUALITATIVE EXAMPLES'' + claims.md C2
  qualitative panel gap'
workflow: v1
---
## Goal

Assemble the paper's qualitative-examples panels from EXISTING artifacts (0 GPU): (1) discrimination — concrete context/answer pairs the map distinguishes vs fails to distinguish (from #2202 failure attribution, #2215 minimal pairs, #1482 per-language errors); (2) context-vector patching — before/after generation pairs for clean context-end patches (from #2094/#2162 per-pair companions and judged dashboards). Output: one figure/table per panel under figures/paper/ (c3_qualitative_*.pdf, appendix_patching_examples.pdf) plus the verbatim examples committed as text, each with provenance (issue #, artifact path, any display substitution disclosed inline).

## Provenance

Paper outline (Thomas, 2026-08-22): "QUALITATIVE EXAMPLES" under Results I limits; claims.md C2 row "Qualitative examples: patching only the context vector — no assembled qualitative panel yet". Global paper convention: R² + acc@1 + qualitative examples for every claim.

## Design notes

- Read-only over eval_results/ + HF raw completions; no new generation, no new judging.
- Follow the display-substitution disclosure rule and plain-English condition names (no bare codes).
