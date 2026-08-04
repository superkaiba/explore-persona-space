---
title: 'daily-fix: widen body-audit condition-label regex H1c forms'
kind: infra
tags:
- wf-fix
- wf-fix-fp:abc36e453f80
- daily-auto-filed
created_at: '2026-07-31T06:52:29Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): the condition_labels regex
  pattern class [CcHhP][1-9] misses H<digit><lowercase> hypothesis-tag forms (H1c/H4b)
  in body prose, the sibling of the check-28 class #1826 fixed for figure sidecars
  only.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 Step C (parked workflow-fix-candidate routing) from a candidate parked on task #1826 (emitting agent: #1826's code-reviewer round-1 bug-class sweep, synthesized to a formal block by the recursion-guarded session; parked 2026-07-30T12:39:52Z). #1826 itself fixed the sibling instance (check 28's H-code class in `scripts/verify_task_body.py`, merged 934114a8eff8) — this file was explicitly out of that plan's must-ask fence.

## Goal

Widen the `condition_labels` token pattern in `scripts/audit_clean_results_body_discipline.py` so `H<digit><lowercase-letter>` hypothesis-tag forms (H1c, H4b) in body prose are caught, matching what #1826 fixed for figure sidecars.

## Workflow gap

- **Bug observed:** the condition_labels regex (pattern class `[CcHhP][1-9]`) misses H<digit><lowercase-letter> hypothesis-tag forms (H1c/H4b) in body prose the same way check 28's old `\bH\d\b` did in figure sidecars — the sibling instance found by the code-reviewer's bug-class sweep on #1826's diff.
- **Why it is a workflow gap:** the prose-discipline audit's token pattern is narrower than the hypothesis tags plans actually emit, so the class it audits passes silently on the H1c form; same failure shape as #1826 (fixed there for figure sidecars only — the plan's must-ask fence barred touching this separate surface).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'CcHhP' scripts/audit_clean_results_body_discipline.py` → 1 hit, line 385: `r"\b[CcHhP][1-9](?:'|′)?"` (no `[a-z]?` suffix; the candidate's cited :360-365 has drifted to :385 — same pattern, same file) (2026-07-31 filing time). Landed-fix check: #1826's merged diff touched `scripts/verify_task_body.py` check 28, not this file.

## Proposed change (candidate diff sketch — refine in planning)

Widen the condition_labels token pattern `[CcHhP][1-9]` → `[CcHhP][1-9][a-z]?` with its own calibration pass against that check's known-goods, + a test pin in tests/test_audit_clean_results_body_discipline.py.

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the `[CcHhP][1-9]` pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. Calibrate against known-good bodies so the widened pattern does not newly FAIL grandfathered clean-results (the check is WARN/FAIL discipline-sensitive).
- ruff on touched files passes; the audit script's existing tests stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: abc36e453f80 (tag-authoritative; supersedes body-carried fingerprint: 92e1ec47d599)
- origin: parked candidate-block on #1826 events.jsonl, ts 2026-07-30T12:39:52Z (routed by /daily 2026-07-30 Step C)
