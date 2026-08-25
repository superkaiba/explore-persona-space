---
title: verify_task_body.py Lens 14 ack source spans to EOF, so footer-only concern
  ids mechanically PASS
kind: infra
tags:
- concerns-ledger
- verify-task-body
created_at: '2026-08-24T14:18:23Z'
has_clean_result: false
origin_prompt: 'Reconciler workflow-fix-candidate v1 during #2254 clean-result reconciliation:
  footer-only concern ids mechanically PASS Lens 14 because the Results H2 span runs
  to EOF.'
workflow: v1
---
## Goal

`check_concerns_audit`'s v4 acknowledgment source uses `section_text(body, "Results")`, whose H2 span runs to end-of-document. The `**Repro:**` / `**Context:**` footer is therefore silently inside the acknowledgment source, so concern ids mentioned ONLY in the footer pass Lens 14 even though the lens text and SPEC (1183-84) accept only result-H3 prose, Takeaways-bullet prose, or a deferral marker.

## Evidence

Demonstrated live on task #2254, clean-result critique round 3 (2026-08-24). Five open binding concern ids appeared only at body line 297, inside the footer. `verify_task_body.py --issue 2254` returned OVERALL PASS with zero FAILs, and the Claude clean-result-critic cited that mechanical PASS as part of its PASS verdict on Lens 14.

The Codex twin raised it (`binding-concerns-footer-only`, BLOCKER). The binding reconciler upheld it as a CONCERN — downgrading the severity because the lens's own guidance for substantively-acknowledged-but-misplaced concerns is a CONCERNS bullet rather than a standalone FAIL — and confirmed by execution that the verifier's Results span reaches EOF, so the mechanical PASS was a checker artifact rather than evidence of conformance.

Why this matters beyond one body: a reviewer reasonably treats a verifier PASS as evidence. Here the checker silently widened its own acceptance surface, so the mechanical gate and the written lens disagree, and the disagreement is invisible at the point of use.

## Fix sketch (from the reconciler's workflow-fix-candidate, confidence high)

Truncate the v4 Results acknowledgment source at the `\n---\n` + `**Repro:**` footer boundary — or reuse the existing footer-extraction helper — before the substring scan. Add a fixture with a footer-only concern id expecting FAIL.

Also worth checking while there: whether any OTHER check reuses `section_text(body, "Results")` and inherits the same EOF-spanning behavior.

## Provenance

Surfaced by the binding reconciler during #2254 Step 9a-bis reconciliation, 2026-08-24. Sibling ledger defects filed this session: #2530 (list-concerns --open-only undercounts a severity-downgrade re-raise) and #2534 (codex twin emits MAJOR/MINOR severities the ledger rejects all-or-nothing). All three distort what a reviewer sees about the concern ledger.
