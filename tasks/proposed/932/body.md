---
title: 'workflow-fix: verify_plan check — fail-loud acceptance claims need committed
  tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a326765ae05d
created_at: '2026-07-03T12:26:28Z'
has_clean_result: false
origin_prompt: 'codex-critic statistics S5 on #913: fail-loud acceptance claims should
  be backed by committed tests when the defect is silent swallowing (recurring workflow-surface
  verifier candidate)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #913 (emitting agent: codex-critic statistics twin, S5; surfaced by the reconciler's "Observed but not raised").

## Goal

Add a verify_plan.py check requiring plans whose acceptance criteria assert fail-loud / no-silent-swallow behavior to name a committed test backing the claim, not only run-book grep commands.

## Workflow gap

- **Bug observed:** Task #913's plan certified its fail-loud acceptance criterion only via run-book grep gates; the Codex statistics critic showed a differently-worded re-swallow would ship green past all committed tests
- **Why it is a workflow gap:** verify_plan.py (and the Statistics critic lens) has no check that a fail-loud acceptance claim is backed by a committed test — run-book grep gates verify the invariant once at review time, not durably.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

- verify_plan.py: new WARN-level check — when a plan's acceptance criteria / success criteria contain fail-loud vocabulary ("fail loud", "no silent", "not swallowed", "raises", "no except") coupled to a run-book-only verification (a grep gate with no named test), WARN asking for a committed test naming.
- Optionally mirror as a Statistics & Measurement lens bullet in .claude/rules/critic-lens-reference.md.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'fail.loud\|grep gate' .claude/ scripts/verify_plan.py`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; keep the check WARN-level (advisory), never a hard FAIL on grandfathered plans.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: a326765ae05d

Verbatim surfaced prose (codex-critic statistics twin, #913 round 1, S5): "the acceptance items certified only by run-book checks are 'no `except Exception` remains' and the broader 'no warning-and-continue path remains.' For this fix, that should be a committed AST/source test, not only a grep command. This is a recurring workflow-surface verifier candidate: fail-loud acceptance claims should be backed by committed tests when the defect is silent swallowing."
