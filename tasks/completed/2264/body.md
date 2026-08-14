---
title: 'verify_task_body: WARN checks for fold-staled fixtures (per-unit-exemption
  token; acknowledgment result-count)'
kind: infra
tags: []
created_at: '2026-08-13T04:27:41Z'
has_clean_result: false
parent_id: 2224
origin_prompt: 'workflow-fix prose follow-up from the #2224 round-3 clean-result-critic
  (fold-staled Methodology fixtures)'
workflow: v1
---
## Goal
Add two WARN-grade checks to scripts/verify_task_body.py catching the two Methodology fixtures that recurrently go STALE when follow-up rounds are folded into a v4 clean-result body (surfaced by the #2224 round-3 clean-result-critic, 2026-08-13):

1. **Per-unit-exemption check (Lens 11 mechanization):** WARN when a `### <result>` block's only figure caption reports a correlation/AUC-family statistic (r / rho / R2 / AUC) and the block contains neither a second embedded figure nor a literal `Per-unit exemption` token. (#2224: two folded aggregate results shipped without the per-unit view or the exemption line the body itself used elsewhere.)

2. **Acknowledgment result-count check:** when the body's conciseness-cap acknowledgment paragraph states "across N results", compare N to the actual `###` count under `## Results`; WARN on mismatch (and WARN on a "single-round" claim when the body carries folded-round deviation bullets). (#2224: the acknowledgment said "single-round ... nine results" while the folded body had 13.)

Both WARN-only (forward-only discipline: v3/v2 bodies exempt via the v4 sentinel gate). Add fixture-backed tests beside the existing verify_task_body tests.

## Provenance
Surfaced as a workflow-surface prose follow-up in the #2224 round-3 clean-result-critic verdict (post-fold delta review). Fold rounds recurrently stale-ify these two fixtures; mechanical WARNs catch them at the next fold instead of costing a critic round.
