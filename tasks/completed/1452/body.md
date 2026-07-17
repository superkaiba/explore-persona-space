---
title: 'daily-fix: flag hardcoded cross-issue upload prefixes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ff4664993e7e
- daily-auto-filed
created_at: '2026-07-17T06:56:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): reused parent upload code
  carried a hardcoded issue928_ data-repo prefix, so #1005''s decomp/preds tensors
  overwrote the PARENT''s HF artifacts — caught only at the upload-verification gate,
  then re-homed + parent restored; artifact-reuse rule (i) names ''data-repo Hub calls
  prefix-scoped'' but implementer + code-review both missed it'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1005 upload-verifier FAIL ~11:07Z: tensors uploaded to hardcoded #928 prefixes, overwriting the parent's HF artifacts).

## Goal

Catch inherited hardcoded issue-prefixes in upload call sites mechanically before they clobber a parent's artifacts.

## Workflow gap

- **Bug observed:** reused parent upload code carried a hardcoded issue928_ data-repo prefix, so #1005's decomp/preds tensors overwrote the PARENT's HF artifacts — caught only at the upload-verification gate, then re-homed + parent restored; artifact-reuse rule (i) names 'data-repo Hub calls prefix-scoped' but implementer + code-review both missed it
- **Why it is a workflow gap:** Artifact reuse is the project default; a reused uploader whose prefix is not re-threaded DESTROYS parent data — the worst failure class short of data loss without backup.
- **Confidence (emitter):** medium-high (destructive incident today; recovery required parent restore)
- verified-at-filing: `grep -in 'prefix.scoped\|issue.*prefix' scripts/workflow_lint.py` -> 0 hits (no such check — absence claim); incident evidence on #1005 events.jsonl (upload-verification FAIL + re-home, 2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

add a lint (or code-reviewer checklist item) that greps experiment upload call sites for a hardcoded issue<M>_ data-repo prefix where M != the current issue, WARN/FAIL on mismatch

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ff4664993e7e

