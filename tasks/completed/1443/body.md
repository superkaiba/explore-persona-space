---
title: 'daily-fix: anchor repo-root guard cd-scope latch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c4bd5606f0d2
- daily-auto-filed
created_at: '2026-07-17T06:51:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the driver loop''s cd-scope
  latch (now L1178-1179) matches unanchored cd /tmp/ / .claude/worktrees/ text inside
  a quoted ssh payload fragment and arms scoped across && into a local gated clause
  — a pre-existing allow on the internal-&& payload variant, independent of the #1413
  mask'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked FORMAL candidate on task #1413 (Phase-2 Alternatives critic; fp fa8b026c75b7). #1413's own merge (a87f5898f6) added the conservative ssh single-quote mask; this candidate is the driver-side root fix, explicitly noted as independent.

## Goal

Stop remote-payload text from mutating the repo-root guard's local cd-scope latch state.

## Workflow gap

- **Bug observed:** the driver loop's cd-scope latch (now L1178-1179) matches unanchored cd /tmp/ / .claude/worktrees/ text inside a quoted ssh payload fragment and arms scoped across && into a local gated clause — a pre-existing allow on the internal-&& payload variant, independent of the #1413 mask
- **Why it is a workflow gap:** The latch regexes are quote-blind and run before the ssh waiver, so remote-payload text mutates local scoping state — a guard-correctness hole.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "cd +" scripts/guard_repo_root_branch.sh` -> latch at L1178-1179 still unanchored (`grep -qE 'cd +[^;&|]*\.claude/worktrees/'` / `'cd +/tmp/'` — no ^ anchor) on the post-#1413-merge tree; `git log --oneline --since='7 days ago' -- scripts/guard_repo_root_branch.sh` -> a87f5898f6 (#1413 mask — different fix)

## Proposed change (candidate diff sketch — refine in planning)

Anchor both latch greps to clause-initial cd (`^cd +...`), or refuse latch-arming when the clause head is ssh/quoted; add a regression test beside the existing guard tests.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: c4bd5606f0d2


Verbatim formal candidate block lives on task #1413 events.jsonl (epm:workflow-fix-candidate, 2026-07-16T22:24:51Z, fp fa8b026c75b7).
