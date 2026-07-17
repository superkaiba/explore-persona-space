---
title: 'daily-fix: plan-glob vs uploader allow-pattern parity check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1b3b10255ffd
- daily-auto-filed
created_at: '2026-07-17T06:56:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): no review surface diffs
  plan-declared artifact globs against the uploader''s allow-patterns: #825''s upload
  helper allowed only **/*.npy + **/*.json so all 404 row_index*.jsonl files (48.9
  MB) declared in plan v24 section 6.5 were never eligible for upload — caught only
  by the upload-verifier with the instance still alive'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#825 round, ~21:19Z upload-verifier FAIL).

## Goal

Catch plan-declared-but-uploader-ineligible artifact classes at implementation review instead of at the upload-verification gate.

## Workflow gap

- **Bug observed:** no review surface diffs plan-declared artifact globs against the uploader's allow-patterns: #825's upload helper allowed only **/*.npy + **/*.json so all 404 row_index*.jsonl files (48.9 MB) declared in plan v24 section 6.5 were never eligible for upload — caught only by the upload-verifier with the instance still alive
- **Why it is a workflow gap:** The upload-verifier is the LAST line; a declared artifact class silently excluded by an allow-pattern should die in code review, not at the gate (and would be lost entirely on a crashed instance).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -icn 'allow.pattern\|glob' .claude/rules/upload-policy.md` -> 0 (no parity rule — absence claim); incident evidence on #825 events.jsonl (upload-verification FAIL, 2026-07-16 ~21:19Z)

## Proposed change (candidate diff sketch — refine in planning)

add an upload-policy.md rule + a code-reviewer checklist item: when a plan declares artifact globs, the implementation review diffs them against the uploader's allow-patterns/globs and FAILs on a declared-but-ineligible class

## Scope / surfaces

- Primary target: `.claude/rules/upload-policy.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1b3b10255ffd

