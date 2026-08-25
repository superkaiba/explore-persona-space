---
title: 'Cross-model transfer: does base-geometry prediction of post-inoculation re-elicitation
  replicate on Llama-3.1-8B-Instruct?'
kind: experiment
tags: []
created_at: '2026-08-23T19:24:38Z'
has_clean_result: false
parent_id: 2474
origin_prompt: 'park-time follow-up proposal 3 on #2474 (epm:follow-ups v1, 2026-08-23T19:15Z);
  screened not-redundant by both critics'
workflow: v1
goal: 'Replicate the inoculated-FT -> re-elicitation -> base-geometry-predictor pipeline
  on Llama-3.1-8B-Instruct: does the misalignment positive (Qwen pooled rho +0.708)
  and the capitalization sign flip (-0.747) transfer beyond the Qwen family, or are
  they family-specific representation quirks?'
---
## Goal

Replicate the inoculated-FT -> re-elicitation -> base-geometry-predictor pipeline on Llama-3.1-8B-Instruct: does the misalignment positive (Qwen pooled rho +0.708) and the capitalization sign flip (-0.747) transfer beyond the Qwen family, or are they family-specific representation quirks?

## Context

Filed from #2474's park-time follow-up proposals (proposal 3, `question_relation: substantially-different` — a fresh Goal, so it cannot fold into the parent). Screened not-redundant by both follow-up critics (2026-08-23): no second-family replication of this pipeline exists in the corpus (ruled out: #544, #1426, #1336, #2502, #697, #2389).

Estimated 32 GPU-h. `auto_run: no` — model pick, layer selection, and map re-derivation for the new family need human scoping before planning (the parent's layer pins 16/27 and the #779 LMSYS map are Qwen-specific and must be re-derived, not reused).

Full proposal spec: the `epm:follow-ups v1` marker on #2474 (2026-08-23T19:15Z), proposal 3.
