---
title: 'workflow-fix: daily sweep counts warn firings, not command-text echoes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:74a753edfa0f
created_at: '2026-07-17T22:34:00Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from /issue 1473 investigation: /daily 2026-07-16
  counted 13 choom failures where 12 were the warn string embedded in tool_use command
  text (gate recipe echoes) and only 1 was a real tool_result firing (benign pgrep-empty
  shape); the sweep needs a count-firings-not-echoes discipline so quantified failure
  claims are firings-only.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #1473 (emitting agent: /issue 1473 orchestrator session, investigation of the "fix sudo choom" daily-held item).

## Goal

The /daily problem sweep counts error/warn evidence by actual tool_result firings, never raw string occurrences (command text echoes recipe warn strings).

## Workflow gap

- **Bug observed:** /daily 2026-07-16 reported "13 failed vs 8 ok" choom failures in one session's transcript; re-parsing that transcript (6d38a307) shows 12 of the 13 "failures" were the literal `[warn] choom failed` string embedded in tool_use COMMAND text (the Step-10d/9c gate recipes carry the warn message verbatim in their preambles), and only 1 was a real tool_result firing — and that one was the documented-benign pgrep-empty shape, not a sudo denial. The miscount produced false needs-human task #1473; investigation found sudoers already `NOPASSWD: ALL`, 0/50 serial+concurrent stress failures, and zero sudo denials in auth.log.
- **Why it is a workflow gap:** the daily sweep's transcript mining has no count-firings-not-echoes discipline; any workflow recipe that embeds its own error/warn string in command text (the choom preambles, gate FATAL echoes, warn-and-continue messages) inflates occurrence counts and can spawn false fix tasks.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "tool_result" .claude/skills/daily/SKILL.md` → 0 hits in-target (absence claim: no firings-vs-echo discipline exists there); repo-wide relocation grep `grep -rln "count tool_result firings\|command-text echo\|string-occurrence" .claude/ CLAUDE.md scripts/` → 0 files; landed-fix history check `git log --oneline --since='7 days ago' -- .claude/skills/daily/SKILL.md` → 6 commits, all verified-at-filing/SHA machinery, none landing this discipline (2026-07-17)

## Proposed change (candidate diff sketch — refine in planning)

```
+ In the "Problem sweep" transcript-mining instructions (near the "Go
+ through today's transcripts in detail" paragraph), add an
+ evidence-counting discipline bullet: when counting occurrences of an
+ error/warn string in a transcript, count only matches inside
+ tool_result CONTENT (actual firings); matches inside tool_use
+ `command` text are recipe echoes (the workflow's own gate preambles
+ embed their warn strings verbatim) and MUST be excluded. Any
+ quantified claim ("N failed vs M ok") in a filed body must be
+ firings-only, with the counting method stated.
```

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'transcripts in detail' .claude/`) and update every hit;
  list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md § Recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: 74a753edfa0f

Surfaced prose (verbatim, from the /issue 1473 investigation report): "the false '13 failed vs 8 ok' figure came from counting `[warn] choom failed` string occurrences in the transcript, 12 of which were the gate recipe's own warn text embedded in command strings — the daily sweep should count tool_result firings, not raw string matches."
