---
title: 'daily-fix: PreToolUse guard rejecting bare | python in Bash'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9c052fb4e320
- daily-auto-filed
created_at: '2026-06-30T06:44:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-29 auto-filed route-2: Bare `| python -c` / `| python3
  -c` on the consumer side of a pipe was used ~41x across 4+ sessions; at least one
  died with `python: command not found` (exit 127). The CLAUDE.md rule exists but
  is not mechanically enforced.'
---
## Overview / Motivation
Auto-filed by /daily 2026-06-29 problem sweep. The "use `uv run python`, never bare python" rule is documented but unenforced; agents piped bare `| python -c` ~41x across 4+ sessions today, one hitting exit 127.

## Goal
A bare `| python`/`| python3` consumer-side invocation is mechanically blocked or flagged before it runs.

## Workflow gap
- **Bug observed:** ` ... | python -c "..."` -> `/bin/bash: python: command not found` (exit 127); only survived elsewhere because `python3` happens to exist on the VM. Recurs because it is composed inline.
- **Why it is a workflow gap:** the rule lives in CLAUDE.md but nothing enforces it; the heredoc-dotenv case already has a lint check to mirror.
- **Confidence (emitter):** medium; high recurrence (~41x).

## Proposed change
- Preferred: a PreToolUse hook in `.claude/settings.json` that rejects a Bash command containing ` | python ` or ` | python3 ` (consumer-side), with a message pointing to `uv run python`.
- Alternative/additional: a `workflow_lint.py --check-pipe-python` mode for `scripts/*.sh`.
- The planner decides the least-false-positive mechanism (a hook can false-positive on a literal string; scope carefully).

## Scope / surfaces
- `scripts/workflow_lint.py` and/or `.claude/settings.json` (PreToolUse hook).
- This is a hook/behavior change -> full /issue review applies.

## Constraints / invariants
- Workflow surface only. `workflow_lint.py --check-references` + `--check-asks` stay green.
- Recursion guard: EPM_WORKFLOW_FIX_SESSION=1.

## Provenance
- workflow_fix_target: scripts/workflow_lint.py, .claude/settings.json
- fingerprint: 9c052fb4e320

Sessions: bc75d989, ac34fc1d, b710f40b, e8c185b0 (and others).
