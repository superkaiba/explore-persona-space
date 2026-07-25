---
title: 'daily-fix: inline worker briefs carry lint-cert command'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1758d08f359c
- daily-auto-filed
created_at: '2026-07-25T06:49:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The 1092 pooled-probe-runner
  teammate finished its analysis but stalled on the final commit three times - the
  inline payload commit guard blocked its uncertified repo-root py commit and it idled
  instead of running the inline lint gate certification or reporting the blocker on
  the teammate channel'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session f4b1d707, task #1092).

## Goal

Inline-round worker/teammate briefs must arm the worker with the payload-certification recipe so a guard-blocked commit becomes a certification run (or an immediate report), not a silent stall.

## Workflow gap

- **Bug observed:** the #1092 pooled-probe-runner finished the pooled-probe-transfer analysis but left 8 files staged and never committed — "the runner has stalled on the last mile three times now" (orchestrator, 00:01:52Z); its post-stand-down note shows the commit guard blocked its uncertified repo-root `.py` commit ("needs the inline_lint_gate.py certification (guard blocked my manual-check commit)") and it idled rather than running the certification or reporting on the teammate channel. The orchestrator had to stand it down, run `inline_lint_gate.py` (PASS, cert 40b0a97ca552), and commit itself. Related friction the same night: the no-flags workflow_lint run exceeded the 120s foreground Bash timeout and was harness-moved to background mid-gate (1 firing).
- **Why it is a workflow gap:** the 9a-ter inline payload lint gate documents the ORCHESTRATOR duty, but worker briefs composed for inline rounds don't carry the certification command or the report-a-blocked-commit-immediately contract; the teammate-coordination rule (report SendMessage ends the turn) doesn't cover mid-work guard blocks.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'inline_lint_gate' .claude/skills/issue/SKILL.md` → gate recipe present in 9a-ter; no worker-brief clause naming the certification command or a guard-blocked-report duty (absence bind, read in context 2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

In SKILL.md § 9a-ter (inline payload lint gate / worker-brief recipe): (a) briefs that stage repo-root `.py` payloads MUST inline the `inline_lint_gate.py` certification command; (b) a guard-blocked commit is reported on the teammate channel immediately (never idled on); (c) run the no-flags lint as background Bash from the start (~2-4 min).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` § 9a-ter

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- sha-verify (filing-time, #1467): `40b0a97ca552` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 1758d08f359c

- workflow_fix_target: .claude/skills/issue/SKILL.md
