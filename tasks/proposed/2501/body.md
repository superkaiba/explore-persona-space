---
title: 'verify_task_body: band-clearance-count caption check (check 45 extension)'
kind: infra
tags: []
created_at: '2026-08-23T17:12:25Z'
has_clean_result: false
workflow: v1
---
# verify_task_body: mechanize band-clearance-count caption checks (check 45 extension)

## Goal
Extend verify_task_body.py check 45 (or add a sibling check) to cross-verify counting claims in figure captions against the pinned stats artifact for permutation-band figures: when a caption asserts "N of M arms clear their null bands", recompute the count from the referenced stats JSON (observed max vs null p95/p975 per family) and FAIL/WARN on mismatch.

## Problem (driving incident)
Task #2474 clean-result round 1: the prefit_perm_band_caps.png caption claimed "Only the two training-reference transforms clear their bands" while the pinned prefit_stats.json shows THREE clear (ceiling_trainref observed max 0.665 > null p97.5 0.583). Caught only by the Claude clean-result-critic's Lens-3 recompute (marker epm:clean-result-critique v1, 2026-08-23T16:57:59Z), which sketched the mechanization: "extend verify_task_body.py check 45 to band-clearance counts".

## Sketch
Parse captions adjacent to perm-band figure embeds for count phrases ("N of M ... clear", "only the N ..."); resolve the figure's meta.json sidecar -> stats path + family list; recompute clearance (observed_pooled_max_over_layers vs null_max_p95/p975 per the sidecar's band definition); WARN on mismatch (caption semantics fuzzy -> WARN not FAIL), with a standalone N/A escape for bodies without band-count claims. Fixture: the #2474 shape (claimed 2, actual 3) firing; the corrected caption silent.

## Provenance
Surfaced as a mechanizable prose follow-up in the #2474 clean-result-critic round-1 verdict. Filed by the #2474 orchestrator per the workflow-fix-on-bug surfaced-prose rule. Distinct fingerprint from #2490 (new-script CLI choice sets) and #2491 (null-localization prose).
