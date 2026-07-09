---
title: 'workflow-fix: Fix ls-tree fail-open in Step 10d stale-folder'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4d0fdfd59daf
- daily-auto-filed
created_at: '2026-07-09T06:59:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The Step 10d post-merge
  stale-task-folder guard''s `mapfile -t DUPES < <(git ls-tree ... | grep ... || true)`
  collapses a FAILED git ls-tree into ''no duplicate folders'', silently skipping
  cleanup on a non-certification path.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1047.

## Goal

Make the Step 10d post-merge stale-task-folder guard fail loud when git ls-tree itself fails, instead of reading the failure as 'no duplicates'.

## Workflow gap

- **Bug observed:** The Step 10d post-merge stale-task-folder guard's `mapfile -t DUPES < <(git ls-tree ... | grep ... || true)` collapses a FAILED git ls-tree into 'no duplicate folders', silently skipping cleanup on a non-certification path.
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  - mapfile -t DUPES < <(git ... ls-tree -d -r --name-only origin/main \
  -   | grep -E "^tasks/[^/]+/<N>$" | grep -v -F -x "$CANON" || true)
  + LSTREE_OUT=$(mktemp); if ! git ... ls-tree -d -r --name-only origin/main > "$LSTREE_OUT"; then
  +   echo "FATAL: ls-tree failed — cannot certify no stale folders" >&2; exit 1; fi
  + mapfile -t DUPES < <(grep -E "^tasks/[^/]+/<N>$" "$LSTREE_OUT" | grep -v -F -x "$CANON" || true)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: parked candidate on task #1047 at 2026-07-05T16:24:53Z

Verbatim parked note:

```
source: prose-followup (code-reviewer r4 standing rec, out of round scope, pre-existing). target_file: .claude/skills/issue/SKILL.md (post-merge stale-task-folder guard). proposed_change: the guard's 'mapfile -t DUPES < <(git ls-tree ... || true)' collapses a FAILED ls-tree into 'no duplicate folders' — a missed-cleanup fail-open on a non-certification path; apply the same materialize-then-check pattern the #1047 round-3/4 fixes gave the gate triggers (+ a guard3-style pin). routed: parked — running under workflow_fix_target recursion guard; log + notify, not auto-filed. related_task: #1047
```
