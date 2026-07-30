---
title: 'daily-fix: recursion guard — briefs must not ban candidates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6b888bb5c204
- daily-auto-filed
created_at: '2026-07-28T07:02:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): two wf-fix sessions (#1732''s
  briefs; #1730''s discovering session) misread the recursion guard as ''do not emit
  candidate blocks'' — subagent briefs instructed ''do NOT emit workflow-fix-candidate
  blocks; prose only'', suppressing the escape valve the nightly sweep enumerates'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Sessions d71a9e1e (#1732 briefs, 17:10/17:53Z) + a85d5c0d (#1730, misread as don't-emit), 2026-07-27 (miners B P4 + G P1).

## Goal

Stop recursion-guarded sessions from suppressing the candidate-emission escape valve in their subagent briefs.

## Workflow gap

- **Bug observed:** the #1732 implementer and code-reviewer briefs said 'Do NOT emit any workflow-fix-candidate blocks; log surfaced concerns in your report prose only' — but prose is NOT enumerable by `scripts/sweep_parked_wf_candidates.py`, so a real bug surfaced this way is a lost record. The guard's actual semantics: candidates ARE emitted, the orchestrator PARKS them.
- **Why it is a workflow gap:** the rule's escape-valve paragraph addresses the parking orchestrator, not the brief-composer; nothing tells a session composing subagent briefs that banning emission defeats the sweep (`grep -c 'do not emit' .claude/rules/workflow-fix-on-bug.md` -> 0 — no guidance either way).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -ci 'do not emit' .claude/rules/workflow-fix-on-bug.md` -> 0, compose time; the #1732 brief text is quoted verbatim in the session transcript (miner-probed: the phrase is NOT from SKILL.md — `grep -n 'Do NOT emit' .claude/skills/issue/SKILL.md` -> 0).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/rules/workflow-fix-on-bug.md` § Recursion guard, after the escape-valve paragraph: 'Brief-composers in a workflow-fix session: instruct subagents to emit candidate blocks NORMALLY — the orchestrator parks them (epm:workflow-fix-candidate with a parked note; enumerable by the sweep). Never instruct prose-only concern reporting; prose is not enumerable and the record is lost.'

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md` (§ Recursion guard)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 6b888bb5c204

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
