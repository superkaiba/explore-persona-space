---
title: 'verify_task_body check 17: match Context footer against verbatim ## Provenance,
  not just non-empty'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T13:15:26Z'
has_clean_result: false
origin_prompt: 'clean-result-critic Lens 5 on #2254: Context footer quoted the refined
  goal; mechanical check 17 passed on a non-empty blockquote'
workflow: v1
---
## Goal
Strengthen `scripts/verify_task_body.py` check 17 (Context-footer originating-prompt provenance) so it detects when the `**Context:**` footer blockquotes the REFINED goal instead of the VERBATIM originating prompt recorded in `original-body.md ## Provenance`. On #2254 the footer quoted the refined goal; the mechanical check passed (a non-empty blockquote was present) and only the clean-result-critic Lens 5 caught the mismatch.

## Scope
- Read check 17's current logic. If `original-body.md` has a `## Provenance` verbatim block, verify the footer blockquote is a substring/fuzzy-match of THAT text (not just non-empty, not the refined goal).
- Fail-soft when no `## Provenance` block exists (grandfathered bodies) — WARN, don't hard-FAIL.
- Regression fixture: a body whose footer quotes the refined goal while original-body.md has a distinct verbatim Provenance → must FAIL after the fix.

Provenance: surfaced by clean-result-critic Lens 5 on #2254 (workflow-fix-candidate, prose follow-up).
