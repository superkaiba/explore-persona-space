---
title: 'workflow-fix: letter-arrow transition token class for figure opaque-code check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:95f283ce5e14
created_at: '2026-08-02T02:42:55Z'
has_clean_result: false
origin_prompt: 'clean-result-critic round-1 prose follow-up on #1902: extend verify_task_body.py
  sidecar opaque-config-code token classes with a letter-arrow-transition class ([A-Z]\s*(?:->|→)\s*[A-Z](?:_[a-z]+)?)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1902 (emitting agent: clean-result-critic, round 1).

## Goal

Extend verify_task_body.py's figure-sidecar opaque-config-code token classes with a letter-arrow-transition class so panel titles like "B->S_single" are flagged mechanically.

## Workflow gap

- **Bug observed:** #1902's clusters_delta_qc_scatter.png panel titles carried letter-coded transitions ("B->S_single", "S->D_multi", "D->R_single/multi") in a promoted-body figure; the existing opaque-token classes (snake slugs >=3 segments, digit-bearing, @L-pins, H-codes) did not match, and only the LM figure lens (clean-result-critic Lens 3) caught it.
- **Why it is a workflow gap:** the no-opaque-condition-codes rule is enforced mechanically for slug/digit shapes but a bare letter-arrow transition code escapes every class; each escape costs a critic round.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -nE "opaque|token.class|slug" scripts/verify_task_body.py` → opaque-token check block present near L520-536 (check 26 family); `grep -cE '\[A-Z\].*->.*\[A-Z\]|letter.arrow' scripts/verify_task_body.py` → 0 hits in-target (absence-of-guard claim: no letter-arrow class exists — this IS the evidence) (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # letter-arrow transition codes ("B->S_single", "S→D") in figure titles/sidecars
+ r"\b[A-Z]\s*(?:->|→)\s*[A-Z](?:_[a-z]+)?\b",
```
Scoped to the same figure-title/sidecar surfaces as the existing opaque-token classes; keep existing exemptions.

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for sibling checks before editing; watch false-positive surface (legitimate A->B prose in Methodology is NOT a figure title — keep the check scoped to figure titles + sidecar-echoed strings).

## Constraints / invariants

- Workflow-surface only. Ruff + existing tests (tests/test_verify_task_body.py) pass; add a regression test for the new class.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 95f283ce5e14
