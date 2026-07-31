---
title: 'workflow-fix: verify_plan base-predictor-vs-change-DV companion check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d3c998cad003
created_at: '2026-07-31T00:48:25Z'
has_clean_result: false
origin_prompt: 'Statistics critic round-1 mechanizable Must-Fix on #1900: verify_plan.py
  check that a base-side predictor raced against a change DV registers a level companion
  + stated signed-vs-abs winner convention (#559/#605 pattern)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1900 (emitting agent: critic, Statistics & Measurement lens,
round-1 Must-Fix tagged `mechanizable: yes`).

## Goal

Add a `verify_plan.py` check that a plan racing a base-side predictor against a
trained−base CHANGE DV registers a level-DV companion column (and, where a
sibling panel uses a LEVEL DV, a change companion) plus a stated
signed-vs-|ρ| winner-selection convention.

## Workflow gap

- **Bug observed:** plan #1900 v4 raced base propensity (P7, a base-side log P / base judge score) against a marker trained−base CHANGE DV (mechanical ≈ −1 coupling — the #559/#605 pattern) and a content LEVEL DV (mechanical +shared-base coupling), with no stated winner sign convention — the champion verdict and its replication read could both be manufactured by per-panel DV identity. Caught only by the round-1 Statistics critic.
- **Why it is a workflow gap:** the base-side-predictor-vs-change-DV mechanical coupling is a documented recurring pattern in this project's leak-predictor line (#559/#605 measured the sign flip: base↔level ρ +0.28/+0.19 vs base↔delta ρ −0.43/−0.54), and the critic explicitly tagged the finding `mechanizable: yes` for `verify_plan.py` (grep for a registered level/change companion + a stated signed/|ρ| convention when both "trained − base"/change-DV vocabulary and a base-side predictor row co-occur).
- **Confidence (emitter):** low-medium (the check's trigger predicate needs care to avoid false positives on plans that merely mention change DVs; the spawned session's planner may legitimately deflect with a reasoned no-change report or narrow the trigger)
- verified-at-filing: `grep -cn 'level companion\|winner convention\|signed.*rho\|change-DV' scripts/verify_plan.py` → 0 hits (absence-of-guard claim — 0-hit in-target IS the evidence) (2026-07-30). Landed-fix history check: `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` shows 5 commits, none touching DV-identity / companion-column semantics.

## Proposed change (candidate diff sketch — refine in planning)

```
+ def check_change_dv_base_predictor_companion(plan, ...):  # cNN
+     # trigger: plan text carries BOTH (a) a change-DV signature
+     # ("trained − base", "trained-base", "delta log P" as a DV) AND
+     # (b) a base-side predictor raced against it ("base propensity",
+     # "base log P" in a candidate/predictor roster).
+     # requirement: a registered level (or change) companion column AND a
+     # stated winner sign convention ("signed" vs "|ρ|"/absolute).
+     # N/A escape: `N/A — no base-side predictor vs change DV`
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'change DV\|trained − base' .claude/rules/ scripts/verify_plan.py`);
  cross-reference `.claude/rules/critic-lens-reference.md` Statistics lens item 2
  (the #559/#605 pattern owner) so the mechanical check and the lens text stay
  consistent. Pin with a test in `tests/test_verify_plan.py`; register any new
  N/A escape phrase in `.claude/skills/adversarial-planner/SKILL.md` Phase 1.5.0.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: d3c998cad003

Surfaced prose (verbatim, from the Statistics critic's round-1 Must-Fix on #1900):
"mechanizable: yes — assert §6/§4-P3 register a P7-vs-level read for the marker
race, a graded-change companion for content, and a stated signed/|ρ| convention
(grep the registered column list)."
