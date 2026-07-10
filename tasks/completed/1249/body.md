---
title: Step 10d bare merge/push forms + fail-soft Step-0 probes
kind: infra
tags:
- wf-fix
- wf-fix-fp:18cf43fe357e
- daily-auto-filed
created_at: '2026-07-10T06:55:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): 14 sessions/day hook-blocked
  on piped git merge/push in Step 10d recovery; 5+ sessions had informational ls probes
  exit 2 and cancel parallel sibling calls'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from the nightly transcript problem sweep (miners 00/01/02/03 — the two most frequent frictions of the day, each costing one wasted turn per occurrence across ~19 session-incidents).

## Goal

Update the /issue SKILL.md snippets so (a) the Step 10d merge/push recovery recipes show the BARE (unpiped) command forms — or a redirect-to-file form — since `guard_piped_git_push.sh` blocks any piped `git merge`/`git push`, and (b) the Step 0 informational `ls` probes (`ls .claude/cache/manual-issue-<N>.json`, `ls ~/.eps-autonomous/issue-<N>*.json`) are fail-soft so a normal missing-file exit 2 cannot cancel a parallel sibling tool call.

## Workflow gap

- **Bug observed:** (a) 14 sessions today composed `git -C "$WT" merge origin/main 2>&1 | tail` (or `git push | tail`) during Step 10d conflict recovery; the PreToolUse guard blocked each, costing a turn before the bare re-run (sessions incl. #1208/#1178/#1198/#1155/#1171/#1213/#1170/#1167/#1199/#1154/#1149 + 5 in group 00). (b) 5+ sessions' Step-0 probe `ls` on an absent optional file exits 2 and the harness cancels the parallel sibling Bash call (#1159/#1173/#1170/#1213/#1209/#1188).
- **Why it is a workflow gap:** The SKILL.md recovery/probe snippets are what sessions copy; the guard (correct) and the snippets (piped / fail-hard) disagree, so every session re-learns the same two lessons at one wasted turn each.

## Proposed change (refine in planning)

(a) In every Step 10d merge/push recovery snippet, use bare forms; where output capture is needed, use `> /tmp/merge-out.txt 2>&1` redirect instead of a pipe, and add one sentence: "never pipe a git merge/push — the guard blocks it; redirect to a file instead". (b) Convert the Step 0 optional-file probes to fail-soft (`ls ... 2>/dev/null || true`, or `[ -f ... ] && cat ...`). Coordinate with (do not duplicate) the sibling filings `skill-edit-c-cert-arm-shape` and `guard1-foreign-strip-resurrect`, which touch adjacent Step 10d lines.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; SKILL-content-pinning test suites (`tests/test_step10d_guard3.py`, `tests/test_step_completed_resume.py`, `tests/test_issue_skill_exit_breadcrumb.py`) stay green.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 18cf43fe357e

- workflow_fix_target: .claude/skills/issue/SKILL.md
