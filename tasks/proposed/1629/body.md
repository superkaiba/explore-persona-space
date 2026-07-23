---
title: 'daily-fix: issue-tick no-op in interactive sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4623bdb758c6
- daily-auto-filed
created_at: '2026-07-23T07:02:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): an /issue-tick cron prompt
  fired into a live interactive Q&A and the user had to interrupt it (991161bd, 06:27Z);
  the tick has no human-activity check'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). Mid-Q&A on the #1613 doc, an `/issue-tick` cron prompt fired INTO the interactive conversation and Thomas had to interrupt it 2 s later ("[Request interrupted by user]") to keep his thread. A tick cron armed in (or delivered to) a session a human is actively using derails the interactive thread.

## Goal

`/issue-tick` first-action triage no-ops (or tears down its cron) when the session shows recent HUMAN (non-cron) user messages — an interactive session is not a stalled autonomous session, and the human's thread wins.

## Workflow gap

- **Bug observed:** 991161bd, 2026-07-23T06:27:15–17Z: `/issue-tick` command-message arrived mid-interactive-Q&A; user interrupt 2 s later.
- **Why it is a workflow gap:** the tick's design target is autonomous sessions (Step 0 arm) + interactive pod-launched runs (Step 6d.2 re-arm); nothing in `tick_triage.py` / the tick SKILL distinguishes "human actively typing here" from "stalled session needing a re-drive".
- **Confidence:** medium-high.
- verified-at-filing: `grep -c 'interactive\|human' .claude/skills/issue-tick/SKILL.md` → 1 hit, which is the Step 6d.2 re-arm context line ("interactive pod-launched runs"), not a human-activity check (context read; absence claim for the no-op check), 2026-07-23 UTC.

## Proposed change (refine in planning)

In `scripts/tick_triage.py` (or the SKILL's first-action contract): detect recent human user messages in the session transcript (or a recency signal the harness exposes) and return a NO-OP verdict (optionally CRON-TEARDOWN when the session has shifted to interactive doc work with the task at a gate/terminal state).

## Scope / surfaces

- Primary targets: `.claude/skills/issue-tick/SKILL.md`, `scripts/tick_triage.py`.

## Constraints / invariants

- The stalled-session re-drive coverage for genuinely autonomous sessions is unchanged; fail toward ticking (a missed no-op is annoying, a missed re-drive strands a task). Recursion guard applies.

## Provenance

- fingerprint: 4623bdb758c6

- workflow_fix_target: .claude/skills/issue-tick/SKILL.md
