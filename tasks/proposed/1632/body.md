---
title: 'daily-fix: exclude /tmp scratch from ruff format hook'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f362848c4a14
- daily-auto-filed
created_at: '2026-07-23T07:03:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): the PostToolUse ruff hook
  reformats freshly-Written /tmp/*.py scratch, invalidating Edit anchors (String to
  replace not found, #1602 session)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). The PostToolUse ruff format hook rewrites freshly-Written `/tmp/*.py` scratch scripts, invalidating the harness's "file state is current in your context" assumption: in #1602 (abee1289, 17:41:29Z) an Edit on a just-Written /tmp edit script failed "String to replace not found" because the formatter had reshaped the file between the Write and the Edit — one extra Read+Edit round.

## Goal

The PostToolUse ruff-format hook in `.claude/settings.json` skips `/tmp/` paths (scratch scripts are ephemeral; formatting them buys nothing and breaks Edit anchors).

## Workflow gap

- **Bug observed:** abee1289 (#1602), 17:41:29Z Edit failure → 17:41:37Z "The formatter hook reshaped the file — reading the R11a region to get the current form."
- **Why it is a workflow gap:** the hook matcher formats every `.py` regardless of location; /tmp scratch is written to be executed once, and the reshape races the writer's in-context file state.
- **Confidence:** high.
- verified-at-filing: `.claude/settings.json` PostToolUse hook (line ~162) read at filing: `if [ -n "$file_path" ] && echo "$file_path" | grep -qE '\.py$'` — no path exclusion (presence claim, binds), 2026-07-23 UTC.

## Proposed change (refine in planning)

Add a `/tmp/` (and `/dev/shm/`) exclusion to the hook's path test, e.g. `grep -qE '\.py$'` → also `! echo "$file_path" | grep -qE '^/(tmp|dev/shm)/'`.

## Scope / surfaces

- Primary target: `.claude/settings.json` (PostToolUse ruff hook command).

## Constraints / invariants

- Repo-tree formatting behavior unchanged. Recursion guard applies.

## Provenance

- fingerprint: f362848c4a14

- workflow_fix_target: .claude/settings.json
