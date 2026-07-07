---
title: 'workflow-fix: verify_plan check for 7B capture booked on eval/debug intent'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f7436ef0e447
created_at: '2026-07-07T03:11:54Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #825 methodology critic r1 (base-separator-control):
  verify_plan.py lacks a mechanical check for a 7B activation-capture phase booked
  on --intent eval/debug (the #666/#744 OOM class); fingerprint f7436ef0e447'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #825 (emitting agent: critic, Methodology lens, round 1 of the
`base-separator-control` amendment review).

## Goal

Add a verify_plan.py check that flags a plan whose design/§9 text indicates hidden-state/activation extraction on a ≥7B model while the launch command books `--intent eval` or `--intent debug`, pointing at `capture-7b`.

## Workflow gap

- **Bug observed:** Plan v17 booked `--intent eval` (L4) for a 7B all-layer activation-capture phase — the exact #666/#744 OOM configuration — and sailed through verify_plan PASS (0 FAIL/0 WARN); only manual critic review caught it.
- **Why it is a workflow gap:** `.claude/rules/plan-compute-sizing.md` § Activation-capture HBM sizing is MUST-level but has no mechanical check; c26 covers machine-costing WARN only, not the HBM-intent mismatch.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ CAPTURE_PAT = re.compile(r"(hidden.state|activation (store|capture|extract)|extract_store)", re.I)
+ INTENT_PAT = re.compile(r"--intent\s+(eval|debug)\b")
+ if CAPTURE_PAT.search(plan_text) and INTENT_PAT.search(plan_text) and re.search(r"7B", plan_text):
+     fail("c<N>: 7B activation-capture phase booked on --intent eval/debug (L4); use capture-7b (>=40 GB HBM) per plan-compute-sizing")
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'capture-7b\|Activation-capture HBM' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. Consider a canonical N/A escape phrase for plans whose
  eval-intent booking is legitimate (no capture phase, or a <7B model).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- Beware false positives: the check must not FAIL a plan that merely MENTIONS
  extraction while booking eval-intent for a non-capture phase — key the check
  on the launch-command fence + a capture-phase design section, and provide a
  canonical escape phrase (see verify_plan.py's existing N/A conventions).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: f7436ef0e447

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_plan.py
bug_observed: Plan v17 booked `--intent eval` (L4) for a 7B all-layer activation-capture phase — the exact #666/#744 OOM configuration — and sailed through verify_plan PASS (0 FAIL/0 WARN); only manual critic review caught it.
why_workflow_gap: plan-compute-sizing § Activation-capture HBM sizing is MUST-level but has no mechanical check; c26 covers machine-costing WARN only, not the HBM-intent mismatch.
proposed_change: Add a check that flags a plan whose design/§9 text indicates hidden-state/activation extraction on a ≥7B model while the launch command books `--intent eval` or `--intent debug`, pointing at `capture-7b`.
diff_sketch: |
  + CAPTURE_PAT = re.compile(r"(hidden.state|activation (store|capture|extract)|extract_store)", re.I)
  + INTENT_PAT = re.compile(r"--intent\s+(eval|debug)\b")
  + if CAPTURE_PAT.search(plan_text) and INTENT_PAT.search(plan_text) and re.search(r"7B", plan_text):
  +     fail("c<N>: 7B activation-capture phase booked on --intent eval/debug (L4); use capture-7b (>=40 GB HBM) per plan-compute-sizing")
confidence: medium
related_task: #825
<!-- /workflow-fix-candidate -->
