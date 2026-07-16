---
title: 'workflow-fix: signature-bind fenced calls in the deferred-import sweep'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9017c3d9ada2
created_at: '2026-07-15T10:35:53Z'
has_clean_result: false
origin_prompt: 'code-reviewer #1332 r1 prose follow-up: extend the deferred-import
  sweep to signature-bind skip-flag-fenced calls (inspect.signature(fn).bind with
  call-site shape); targets .claude/agents/experiment-implementer.md + .claude/rules/gotchas.md'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a code-reviewer prose follow-up raised on task #1332 (emitting agent: code-reviewer, round 1).

## Goal

Extend the deferred-import sweep (the #606 `--verify-imports` pattern) to signature-bind fenced calls: for each call to an imported helper inside a skip-flag-fenced branch, dry-run `inspect.signature(fn).bind(...)` with the call-site's argument shape.

## Workflow gap

- **Bug observed:** the AST deferred-import sweep green-lit two production bind/target errors on calls inside `--skip-upload` / `not smoke` fenced branches (#1332 round 1: `verify_repo_paths_uploaded` called with a mismatched signature → TypeError at the terminal upload stage; plus a 404'd Hub prefix probed only in the fenced branch).
- **Why it is a workflow gap:** the sweep verifies that imported symbols EXECUTE (import-time), but not that fenced CALLS bind — so the exact class of deterministic post-GPU crash the sweep exists to prevent (#606) still ships whenever the crash site is a call-arity/target error inside a smoke-fenced branch.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rln "verify-imports\|deferred.import\|deferred import" .claude/agents/experiment-implementer.md .claude/rules/gotchas.md` → 6 hits in 2 files (experiment-implementer.md: 2; gotchas.md: 4) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch: in `.claude/agents/experiment-implementer.md` § After-implementation deferred-imports step, add: for each imported helper called inside a skip-flag-fenced branch, dry-run `inspect.signature(fn).bind(<call-site args shape>)`; in `.claude/rules/gotchas.md` "Lazy imports inside smoke-skipped branches" entry, extend the recipe with the bind check.

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md`, `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'verify-imports' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md, .claude/rules/gotchas.md
- fingerprint: 9017c3d9ada2

Verbatim surfaced prose (code-reviewer #1332 round 1):

> **Follow-ups (orchestrator should consider):** the implementer's AST deferred-import sweep (the #606 `--verify-imports` pattern referenced in `.claude/rules/gotchas.md` and `experiment-implementer.md`) verifies imported symbols EXECUTE but not that fenced CALLS bind — both Criticals here were bind/target errors on calls inside `--skip-upload`/`not smoke` branches that the import sweep green-lit. A concrete extension: for each call to an imported helper inside a skip-flag-fenced branch, dry-run `inspect.signature(fn).bind(...)` with the call-site's argument shape (target files: `.claude/agents/experiment-implementer.md` § After implementation deferred-imports step, and/or `.claude/rules/gotchas.md` "Lazy imports inside smoke-skipped branches" entry).
