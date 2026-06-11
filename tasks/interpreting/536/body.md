---
title: Standardize persona-distance cosine across the leakage line (mean-centered
  vs raw, ~6x geometry diff) + reanalyze affected predictor calls
kind: analysis
tags:
- needs-thomas
created_at: '2026-06-09T18:35:37Z'
has_clean_result: false
---
RESEARCH-INTEGRITY FLAG, surfaced by the /daily research-integrity scan but never surfaced to Thomas (it was parked in logs/daily/*.md with visible:false; the SessionStart greenlight hook went vestigial on 2026-06-08 so it had no surfacing path — that surfacing gap is being fixed separately).

WHAT: persona-distance cosine is computed two different ways across the leakage line — mean-centered vs raw activations — which compresses the geometry by ~6x. Because the leakage predictors (#404/#458 line and downstream) lean on persona-distance as a load-bearing metric, several MODERATE/LOW predictor CALLS become reanalysis candidates (e.g. tasks touching cosine-predicts-leakage: #61, #66, #91, #96, #99, #142, #227, #245).

WHY IT MATTERS: if 'cosine predicts which personas catch a behavior' conclusions were drawn under inconsistent normalization, their effect sizes/directions may not hold. This is a correctness audit, not a new experiment.

ASK (needs Thomas's judgment): (1) decide the canonical persona-distance definition (mean-centered vs raw) and pin it in persona-distance-metrics.md; (2) grep the leakage line for both code paths; (3) re-run the affected predictor analyses under the standardized metric and re-grade the confidence calls that change.

Source: /daily 'proposal 5' (biggest research-integrity flag), logs/daily/2026-06-02.md + retro. Filed from my-goat as a proposed task to rescue it from the surfacing gap.
