---
title: 'verify_task_body.py check-20: WARN when conciseness WARNs fire with NO acknowledgment
  paragraph at all'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-10T08:26:38Z'
has_clean_result: false
origin_prompt: 'Surfaced by clean-result-critic during #2203 final gate: check-20
  acknowledgment sub-check (#1523) only fires on partial acknowledgment; an all-missing
  acknowledgment with fired WARN classes passes silently.'
workflow: v1
---
## Goal

Close a silent-pass gap in `scripts/verify_task_body.py` check-20 (clean-result conciseness caps): the WARN-acknowledgment sub-check (added in #1523) only fires when a body carries a *partial* acknowledgment paragraph (some fired check-20 WARN classes named, others not). A body that fires one or more check-20 WARN classes and carries **no acknowledgment paragraph at all** passes the sub-check silently — the reader never learns the conciseness overage was deliberate vs. an oversight.

## Context

Surfaced by the clean-result-critic during the #2203 final gate (round 1). The #2203 body fired three check-20 WARN classes (Takeaways bullets > 30 words; all per-result blocks in the 120-179 band; total prose 1345 vs the 800 budget) with zero acknowledgment sentences (`grep -ci acknowledg` = 0), yet `verify_task_body.py --issue 2203` returned OVERALL PASS with the WARNs. SPEC.md § Conciseness caps (v4) requires "a body that ships check-20 WARNs carries a WARN-acknowledgment sentence that names EACH fired class" — so the all-missing case should at minimum WARN (ideally with the same "names each fired class" bar as the partial case). The clean-result-critic caught it as a Lens-12 blocker this round, but the mechanical gate should catch the all-missing shape too, not only the partial one.

## Deliverable

In `scripts/verify_task_body.py` (the check-20 acknowledgment logic, approximately lines 1498-1516), add a "no acknowledgment paragraph found" WARN that fires when: (a) ≥1 check-20 WARN class is fired, AND (b) no acknowledgment paragraph is present at all. Keep the existing partial-acknowledgment sub-check unchanged. Add/extend a fixture in `tests/test_verify_task_body.py` reproducing the all-missing shape (fired classes + zero acknowledgment) and asserting the new WARN fires; confirm the partial-acknowledgment and fully-acknowledged shapes still behave as before. Update SPEC.md only if the wording needs to disambiguate "partial" vs "absent" acknowledgment.

This is a WARN-tier gate change (not a hard FAIL) — it should surface the gap to the analyzer/critic, not block promotion mechanically.
