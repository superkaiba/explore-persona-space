---
title: 'workflow-fix: gotchas entry — script-mode scripts.* imports crash pod-side'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ae30be9d3fff
created_at: '2026-07-02T08:56:05Z'
has_clean_result: false
origin_prompt: 'failure-lesson from #823 (gotcha_candidate: yes): script-mode sys.path
  trap — deferred scripts.* imports fail pod-side; _ensure_repo_root_on_syspath +
  script-mode import checks'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson (`gotcha_candidate: yes`) raised on task #823 (emitting agent: experiment-implementer).

## Goal

Add a gotchas.md entry: deferred `scripts.*` imports in src-layout experiment drivers fail in script mode pod-side; guard with `_ensure_repo_root_on_syspath()` and import-check in script mode from a non-repo cwd.

## Workflow gap

- **Bug observed:** run_823.py's deferred `from scripts.issue779_collect import ...` crashed pod-side with ModuleNotFoundError because script mode puts the script dir, not repo root, on sys.path.
- **Why it is a workflow gap:** the trap is codebase-generic (any experiment driver under src/ deferred-importing a top-level `scripts/` module) and the review/smoke surface missed it twice — a `-c`-mode import check passes (cwd on sys.path) while script mode fails; GPU-phase smoke carve-outs never execute deferred imports. Cost: a full GCE launch died at Phase-3 entry (~30 min paid, task #823, 2026-07-02).
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix validated by script-mode repro)

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## Script-mode `scripts.*` imports crash pod-side (src-layout drivers)
+ In script mode (`python /abs/path/driver.py`) sys.path[0] is the SCRIPT's dir —
+ not cwd, not repo root — so `from scripts.X import ...` raises ModuleNotFoundError
+ on pods/GCE clones. `-c`-mode import checks hide it (cwd on sys.path). Guard every
+ deferred scripts.* import with _ensure_repo_root_on_syspath() (parents[N] + sentinel
+ file assert + sys.path.insert), and verify imports in SCRIPT MODE from a non-repo
+ cwd. Worked example: run_823.py `_ensure_repo_root_on_syspath` (#823, commit 14234c9112).
```

Also consider a one-line pointer in `.claude/agents/code-reviewer.md` Step 0.6 (deferred-imports check must run in script mode from a non-repo cwd) — planner's call.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'ensure_repo_root_on_syspath\|No module named .scripts' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: ae30be9d3fff

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: phase3_tf_extract
lesson: In script mode (python /abs/path/script.py), sys.path[0] is the script's own directory, not cwd or repo root; a top-level non-package scripts/ dir is unreachable. Fix by calling _ensure_repo_root_on_syspath() immediately before any deferred import of scripts.* modules.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
