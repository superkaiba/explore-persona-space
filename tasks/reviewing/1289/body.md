---
title: 'daily-fix: Step 9c/review diff base = fetched origin/main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:91db1c8224f0
- daily-auto-filed
created_at: '2026-07-13T06:44:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): With the shared root main
  temporarily behind origin/main (unresolved #1279 conflict), three sessions got foreign-file-inflated
  diffs/selections from the local-main default base (202KB foreign diff in #1280;
  41-file gate in #1281; hand --base origin/main deviation in #1282).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 problem sweep (transcript-mined; sessions d78028e0 (#1282), eb4dce6b (#1280), 87cd22dc (#1281)).

## Goal

Make Step 9c test selection and code-review diff scoping resilient to a shared repo-root `main` that is temporarily behind `origin/main` — default the diff base to fetched `origin/main` instead of local `main`.

## Workflow gap

- **Bug observed:** on 2026-07-12 ~08:47–09:20Z the shared root sat behind origin with an unresolved conflict on `tasks/reviewing/1279/events.jsonl`. Three concurrent sessions were hit: #1280's `git diff main...HEAD` returned a 202,578-byte diff full of foreign files (self-corrected to `origin/main...HEAD` → 11,637 bytes); #1281's Step 9c selector reported foreign touched files (`.claude/hooks/guard_lessons_edit_check.py`, `tests/conftest.py`), inflating the gate to 41 test files; #1282 had to hand-run the selector with `--base origin/main` and document the deviation in its test-verdict note.
- **Why it is a workflow gap:** `select_step9c_tests.py` defaults `--base main` (scripts/select_step9c_tests.py:454) and the SKILL.md recipes reference local `main`, so every session independently rediscovers the same workaround whenever the always-shared root lags. #1280 already moved Step 10d Guard 1 to `origin/main...HEAD`; the selector + reviewer scoping still key on local main.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "\-\-base" scripts/select_step9c_tests.py` → line 454 `default="main"` (2026-07-13); transcript quotes above from the three sessions.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- parser.add_argument("--base", default="main", help="diff base (default: main)")
+ parser.add_argument("--base", default="origin/main", help="diff base (default: fetched origin/main)")
```
plus a `git fetch origin main` precondition (or fall back to local main when offline), and align the SKILL.md Step 9c / code-reviewer diff-scoping prose with the same base (the diff-size-budget rule's three-dot recipes included).

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`, `.claude/skills/issue/SKILL.md` (Step 9c), `.claude/agents/code-reviewer.md` (Step 0 scoping), `.claude/rules/diff-size-budget.md`.
- Grep for `main...HEAD` across the workflow surface and align every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff passes.
- Offline/no-network fallback must fail toward the current behavior (local main), never block.
- Recursion guard applies to the spawned session.

## Provenance

- fingerprint: 91db1c8224f0

- workflow_fix_target: scripts/select_step9c_tests.py

Origin: /daily 2026-07-12 transcript sweep (sessions d78028e0/eb4dce6b/87cd22dc). Sibling shipped fix: #1280 (Guard-1 three-dot on origin/main).
