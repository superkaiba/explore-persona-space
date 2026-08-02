---
title: 'workflow-fix: dropped-at-gate condition placement check (Takeaways + result)
  in verify_task_body'
kind: infra
tags:
- wf-fix
- wf-fix-fp:085eef8e26cb
created_at: '2026-08-02T08:57:33Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1947 r1 mechanizable finding: Methodology ''dropped
  at gate'' declaration must require the dropped condition named in Takeaways + a
  result (After-Every-Experiment 8a)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `mechanizable: yes` clean-result-critic Lens 13 finding on task #1947 (emitting agent: clean-result-critic, round 1).

## Goal

Add a verify_task_body check requiring a Methodology-declared dropped-at-gate condition to be named in Takeaways and a result section.

## Workflow gap

- **Bug observed:** the #1947 body declared the sycophancy yield-gate drop only in Methodology **Design:** and passed all 65 checks; clean-result-critic Lens 13 caught the missing Takeaways/result placement manually.
- **Why it is a workflow gap:** CLAUDE.md After-Every-Experiment item 8(a) mandates the missing condition be named in `## Takeaways` AND the relevant `### <result>` prose, but `scripts/verify_task_body.py` has no check enforcing the placement — check 11b covers denominator consistency, not the naming placement.
- **Confidence (emitter):** high (mechanizable: yes tagged in the critique marker)
- verified-at-filing: `grep -cE 'dropped at|yield.gate' scripts/verify_task_body.py` → 0 hits (2026-08-02); absence-of-check claim, 0-hit in-target is the evidence (structural check absence, not a text-matching-guard subclass); `git log --oneline --since='7 days ago' -- scripts/verify_task_body.py` shows 3 recent commits, none adding dropped-condition placement handling.

## Proposed change (candidate diff sketch — refine in planning)

```
+ # Check N: a Methodology section declaring a condition "dropped at <gate>"
+ # (yield gate / data gate / kill criterion) requires the dropped condition's
+ # name to appear in ## Takeaways AND in at least one ### result's prose
+ # (After-Every-Experiment 8(a)); FAIL names the declaration line + the
+ # missing placement(s). v4 bodies only; grandfathered shapes exempt.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Companion: `tests/test_verify_task_body.py` (fixture pair: declaring body with/without the placements); SPEC.md mechanical-checks list if it enumerates checks.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff clean; forward-only (never newly hard-FAIL v3/v2/legacy bodies).
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 085eef8e26cb

Surfaced prose (clean-result-critic round-1 verdict, task #1947, 2026-08-02T08:54:40Z): "fix 1's check (Methodology 'dropped at … gate' ⇒ dropped-condition name required in Takeaways/a result) … candidate verifier extensions for the orchestrator to route per the workflow-fix protocol."
