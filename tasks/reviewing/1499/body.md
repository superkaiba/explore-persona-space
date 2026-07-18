---
title: 'daily-fix: alias step 9b to 9b-same in post_step_completed'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6d8cd14d2f87
- daily-auto-filed
created_at: '2026-07-18T06:46:40Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-17 problem sweep (route 2): the #1335 follow-up-loop
  orchestrator posted post_step_completed.py --step 9b where the canonical workflow.yaml
  id is 9b-same; the helper refused (exit 2), the step-completed record was dropped,
  and the error was ignored.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-17 (route 2) from a transcript-mined problem: the #1335 same-issue follow-up session (transcript 40c10095, 2026-07-17 ~11:45Z) invoked `post_step_completed.py --step 9b` for the follow-up-loop entry; the helper refused (exit 2 — step id not in workflow.yaml), the step-completed record was silently dropped, and the orchestrator ignored the error and continued.

## Goal

Make the Step 9b-same step-completed record robust: either alias legacy `9b` → `9b-same` in `scripts/post_step_completed.py`, or make the `/issue` SKILL.md same-issue follow-up loop name the canonical `--step 9b-same` id explicitly at the call site — and ensure an exit-2 refusal is not silently swallowed.

## Workflow gap

- **Bug observed:** an orchestrator posted `--step 9b` where the canonical workflow.yaml id is `9b-same`; `post_step_completed.py` correctly refused (it validates step ids against `workflow.yaml § steps`), but the record was lost and the session continued without it.
- **Why it is a workflow gap:** the step-completed trail is the resume/triage record; a dropped record on the follow-up-loop entry degrades crash recovery and the tick triage read for `followups_running` holds. The naming trap (prose says "Step 9b", yaml id is `9b-same`) invites recurrence.
- **Confidence:** medium
- verified-at-filing: `grep -n '"9b-same"' .claude/workflow.yaml` → 1 hit at line 2937 (canonical id present); `grep -n 'post_step_completed.py.*--step 9b' .claude/skills/issue/SKILL.md` → 0 hits (no literal recipe bug — the orchestrator improvised the id from the prose step name, so the fix is an alias in the helper and/or an explicit id at the SKILL call-site prose); `grep -n "9b" scripts/post_step_completed.py` → 0 hits (no alias handling today; the helper refuses unknown ids by design). All run 2026-07-18 UTC.

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/post_step_completed.py
+ _STEP_ALIASES = {"9b": "9b-same"}   # legacy prose name → canonical yaml id
+ step = _STEP_ALIASES.get(step, step)
```
And/or a SKILL.md same-issue-follow-up-loop sentence naming `--step 9b-same` verbatim.

## Scope / surfaces

- Primary target: `scripts/post_step_completed.py`, `.claude/skills/issue/SKILL.md`
- Grep before editing: `grep -rn "9b-same" .claude/ scripts/` and reconcile every step-name reference.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes; helper tests green.
- This session runs under the recursion guard — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 6d8cd14d2f87

- workflow_fix_target: scripts/post_step_completed.py

source: /daily 2026-07-17 transcript sweep (chunk-1 miner), session 40c10095 (#1335 follow-up loop), ~2026-07-17T11:45Z — `--step 9b` exit 2, record dropped, error ignored.
