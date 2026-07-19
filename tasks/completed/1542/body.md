---
title: 'workflow-fix: smoke must probe smoke-fenced branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9f885f6710e2
- daily-auto-filed
created_at: '2026-07-19T07:07:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): #1481''s impolite lane
  crashed twice on IsADirectoryError in a cfg.smoke-FENCED branch no smoke run could
  reach; gotchas line 244 covers only import/signature drift in smoke-skipped branches,
  not runtime logic (c3-P6).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from the 2026-07-18 /daily
transcript problem sweep (c3-P6). Route-2 filing (gotchas.md lesson content
is landed via the pipeline, never hand-edited by /daily).

## Goal

Require the smoke run to EXERCISE (execute) every non-smoke-fenced load path,
or give a smoke-FENCED branch a standalone 1-cell probe before production
dispatch — extending the existing smoke-skipped-branch entry beyond
import/signature verification to the branch's actual runtime logic.

## Workflow gap

- **Bug observed:** #1481's impolite lane died on an `IsADirectoryError` in a
  branch that was `cfg.smoke`-FENCED, so no smoke run could ever exercise it;
  it was the SECOND crash (after an r1 fix), because the fenced branch was
  still unreachable by any smoke.
- **Why it is a workflow gap:** `gotchas.md` line 244 covers smoke-SKIPPED
  branches only for IMPORT drift + call-signature binds (AST `--verify-imports`
  + `inspect.signature(...).bind`); it does NOT require the fenced branch's
  RUNTIME LOGIC to run. A path-handling bug (`IsADirectoryError`) in a fenced
  branch is invisible to import/signature checks and fires only in production.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n -i 'smoke-fenced\|fenced branch\|non-fenced' .claude/rules/gotchas.md` → 1 hit (line 244, the lazy-import / signature-bind entry — import & arity drift only; no requirement to EXECUTE the fenced branch's logic, no 1-cell-probe rule) (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# gotchas.md — extend the smoke-skipped-branch entry (line 244):
+ Import + signature checks do NOT catch a runtime-logic bug in a smoke-FENCED
+ branch (e.g. an IsADirectoryError from path handling). RULE: the smoke must
+ EXERCISE every non-smoke-fenced load path; a branch that is genuinely
+ smoke-fenced (cfg.smoke short-circuits it) gets a standalone 1-cell probe
+ that actually EXECUTES it before production dispatch. (#1481: an impolite
+ lane crashed twice on IsADirectoryError in a cfg.smoke-fenced branch.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Extend the existing smoke-gate entry (line 244); optionally cross-reference
  `crash-fix-rounds.md` for the second-crash-in-fenced-branch relaunch
  discipline.

## Constraints / invariants

- Workflow-surface only. The 1-cell probe must be cheap (single cell), not a
  full production run.
- `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: f6791449195d

Surfaced problem (c3-P6): #1481 impolite lane died twice on IsADirectoryError
inside a cfg.smoke-fenced branch that no smoke run could reach.
