---
title: 'workflow-fix: gotchas.md — production-n-calibrated gates bind at smoke n'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a4c2a7c6c975
created_at: '2026-07-15T16:52:09Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1345 r3: smoke leg killed by
  production-n parity gate + yield floor'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson (gotcha_candidate: yes) raised on task #1345 (emitting agent: experiment-implementer, crash-fix round 3).

## Goal

Add the smoke-scale-gate trap to .claude/rules/gotchas.md: production-n-calibrated kill gates deterministically kill smoke legs.

## Workflow gap

- **Bug observed:** #1345's smoke leg was deterministically killed by a ±0.02 parity gate comparing n=8 smoke fits against n≈5000 anchors (R²≈−1.5), and a smoke yield floor of 2 (kept=1) rc-21-halted the exact phase the smoke exists to exercise. Two GCE launch cycles burned before the fix round.
- **Why it is a workflow gap:** the PASS_UNIFIED smoke=sweep-with-tiny-n convention (issue SKILL Step 6d.0) interacts with plan kill-criteria calibrated at production n; no rule warns implementers to thread the smoke flag into gate-bearing phases. gotchas.md owns codebase/infra traps of this class.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "smoke" .claude/rules/gotchas.md | grep -ci "gate\|floor\|parity"` → 0 (no existing smoke-scale-gate bullet, 2026-07-15). Per-target: .claude/rules/gotchas.md 0 hits for the pattern (absence-of-guard claim — the 0-hit result IS the evidence).

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet "Production-n-calibrated gates bind at smoke n":
+ a ±tol anchor-reproduction check or absolute yield floor is structurally
+ unsatisfiable at smoke n (PASS_UNIFIED runs the identical chain at tiny n).
+ Thread --smoke into every gate-bearing phase: run the gate COMPUTATION
+ identically, demote production-n-calibrated verdicts to informational,
+ size smoke floors so any nonzero yield proceeds; production halt paths
+ stay byte-untouched + unit-pinned. (#1345 r3; agent-memory twin:
+ experiment-implementer/feedback_smoke_scale_gates.md)

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing (`grep -rln 'smoke' .claude/rules/ .claude/skills/issue/SKILL.md | head`) — consider whether the issue SKILL Step 6d.0 PASS_UNIFIED prose should cross-reference the trap; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. LESSONS.md index untouched (gotchas.md row exists).
- `scripts/workflow_lint.py` passes.
- Recursion guard: the spawned session carries workflow_fix_target and must not auto-route its own candidates.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: a4c2a7c6c975
